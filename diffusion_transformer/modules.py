import math

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from ..diffusion_transformer.activations import get_activation
from ..diffusion_transformer.layers import InputEmbedding
from ..diffusion_transformer.layers import FiLM
from ..diffusion_transformer.layers import SequentialWithFiLM
from ..diffusion_transformer.layers import WNConv1d


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.var_eps = eps

    def forward(self, x):
        """Returns root mean square normalized version of input `x`
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known
        # as Root Mean Square Layer Normalization https://arxiv.org/abs/1910.07467
        # thus varience is calculated w/o mean and there is no bias
        Parameters
        ----------
        x : Tensor[B x T x D]
        Returns
        -------
        Tensor[B x T x D]
        """
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.var_eps)

        return self.weight * x


class FeedForward(nn.Module):
    def __init__(
        self, d_model: int = 512, dropout: float = 0.1, activation: str = "geglu"
    ):
        super().__init__()
        factor = 2 if activation == "geglu" else 1
        self.w_1 = nn.Linear(d_model, d_model * 4, bias=False)
        self.w_2 = nn.Linear(d_model * 4 // factor, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.act = get_activation(activation)()

    def forward(self, x):
        """Computes position-wise feed-forward layer
        Parameters
        ----------
        x : Tensor[B x T x D]
        Returns
        -------
        Tensor[B x T x D]
        """
        x = self.w_1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.w_2(x)
        return x


class MultiHeadRelativeAttention(nn.Module):
    def __init__(
        self,
        n_head: int = 8,
        d_model: int = 512,
        dropout: float = 0.1,
        bidirectional: bool = True,
        has_relative_attention_bias: bool = True,
        attention_num_buckets: int = 32,
        attention_max_distance: int = 128,
    ):
        super().__init__()
        d_head = d_model // n_head
        self.n_head = n_head
        self.d_head = d_head
        self.bidirectional = bidirectional
        self.has_relative_attention_bias = has_relative_attention_bias
        self.attention_num_buckets = attention_num_buckets
        self.attention_max_distance = attention_max_distance

        # Create linear query, key, value projections
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)

        # Create linear final output projection
        self.fc = nn.Linear(d_model, d_model, bias=False)

        # Dropout for attention output weights
        self.dropout = nn.Dropout(dropout)

        # Create relative positional embeddings (if turned on)
        if has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(attention_num_buckets, n_head)

    def _relative_position_bucket(self, relative_position):
        """Converts unbounded relative position into bounded set of buckets
        with half "exact" buckets (1 position = 1 bucket) and half "log-spaced"
        buckets
        Parameters
        ----------
        relative_position : Tensor[T_q x T_kv]
            Relative positions between queries and key_value items
        Returns
        -------
        Tensor[T_q x T_kv]
            Input relative positions converted into buckets
        """
        relative_buckets = 0
        num_buckets = self.attention_num_buckets
        max_distance = self.attention_max_distance

        # Convert relative position for (-inf, inf) to [0, inf]
        # Negative relative positions correspond to past
        # Positive relative positions correspond to future
        if self.bidirectional:
            # use half buckets for each side (past / future)
            num_buckets //= 2

            # Shift the position positions by `num_buckets` to wrap around
            # negative positions
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            # If not bidirectional, ignore positive positions and wrap
            # negative positions to positive
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position)
            )

        # Allocate half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in
        # positions up to `max_distance`
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)

        # Clip the max relative position to `num_buckets - 1`
        relative_postion_if_large = torch.min(
            relative_postion_if_large,
            torch.full_like(relative_postion_if_large, num_buckets - 1),
        )

        # Choose relative buckets based on small or large positions
        relative_buckets += torch.where(
            is_small, relative_position, relative_postion_if_large
        )

        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Computes a position bias scalar for each index in query_length x key_length
        Parameters
        ----------
        query_length : int
        key_length : int
        Returns
        -------
        Tensor[heads x 1 x T_q x T_kv]
            Position bias to be applied on attention logits
        """

        query_position = torch.arange(query_length, dtype=torch.long)[:, None]
        key_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = key_position - query_position

        # Convert relative position to buckets
        relative_position_bucket = self._relative_position_bucket(relative_position)
        relative_position_bucket = relative_position_bucket.to(
            self.relative_attention_bias.weight.device
        )

        # Index attention bias values
        values = self.relative_attention_bias(relative_position_bucket)
        values = rearrange(values, "q k h -> h 1 q k")

        return values

    def forward(self, q, k, v, mask=None, position_bias=None):
        """Computes attention over (keys, values) for every timestep in query
        Parameters
        ----------
        q : Tensor[B x T_q x d_model]
            Query vectors
        k : Tensor[B x T_kv x d_model]
            Key vectors to compute attention over
        v : Tensor[B x T_kv x d_model]
            Value vectors corresponding to the keys
        mask : Tensor[B x T_q x T_kv], optional
        position_bias: Tensor[head x 1 x T_q x T_kv]
        Returns
        -------
        Tensor[B x T_q x d_model]
            Outputs after attending (key, value) using queries
        """
        # Compute query, key, value projections
        q = rearrange(self.w_qs(q), "b l (head k) -> head b l k", head=self.n_head)
        k = rearrange(self.w_ks(k), "b t (head k) -> head b t k", head=self.n_head)
        v = rearrange(self.w_vs(v), "b t (head k) -> head b t k", head=self.n_head)

        # Compute attention matrix
        attn = torch.einsum("hblk,hbtk->hblt", [q, k]) / np.sqrt(q.shape[-1])

        # Add relative position bias to attention scores
        if position_bias is None:
            if self.has_relative_attention_bias:
                position_bias = self.compute_bias(q.size(-2), k.size(-2))
            else:
                position_bias = torch.zeros_like(attn)
        attn += position_bias

        # Apply mask to attention scores to prevent looking up invalid locations
        if mask is not None:
            attn = attn.masked_fill(mask[None] == 0, -1e9)

        # Normalize attention scores and add dropout
        attn = torch.softmax(attn, dim=3)
        attn = self.dropout(attn)

        # Compute attended outputs (product of attention matrix and values)
        output = torch.einsum("hblt,hbtv->hblv", [attn, v])
        output = rearrange(output, "head b l v -> b l (head v)")
        output = self.fc(output)

        return output, position_bias


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        d_cond: int = 64,
        n_heads: int = 8,
        bidirectional: bool = True,
        is_decoder: bool = False,
        has_relative_attention_bias: bool = False,
        flash_attn: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Store args
        self.is_decoder = is_decoder

        # Create self-attention layer
        self.norm_1 = RMSNorm(d_model)
        self.film_1 = FiLM(d_cond, d_model)
        self.flash_attn = flash_attn

        if flash_attn:
            from flash_attn.flash_attention import FlashMHA
            self.self_attn = FlashMHA(
                embed_dim=d_model,
                num_heads=n_heads,
                attention_dropout=dropout,
                causal=False,
            )
        else:
            self.self_attn = MultiHeadRelativeAttention(
                n_heads, d_model, dropout, bidirectional, has_relative_attention_bias
            )

        # (Optional) Create cross-attention layer
        if is_decoder:
            self.norm_2 = RMSNorm(d_model)
            self.film_2 = FiLM(d_cond, d_model)
            self.cross_attn = MultiHeadRelativeAttention(
                n_heads,
                d_model,
                dropout,
                bidirectional=True,
                has_relative_attention_bias=False,
            )

        # Create last feed-forward layer
        self.norm_3 = RMSNorm(d_model)
        self.film_3 = FiLM(d_cond, d_model)
        self.feed_forward = FeedForward(d_model=d_model, dropout=dropout)

        # Create dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        x_mask,
        cond,
        src=None,
        src_mask=None,
        position_bias=None,
        encoder_decoder_position_bias=None,
    ):
        """Computes one transformer layer consisting of self attention, (op) cross attention
        and feedforward layer
        Parameters
        ----------
        x : Tensor[B x T_q x D]
        x_mask : Tensor[B x T_q]
        src : Tensor[B x T_kv x D], optional
        src_mask : Tensor[B x T_kv x D], optional
        position_bias : Tensor[heads x B x T_q x T_q], optional
            Relative position bias for self attention layer
        encoder_decoder_position_bias : Tensor[heads x B x T_q x T_kv], optional
            Relative position bias for cross attention layer
        Returns
        -------
        Tensor[B x T_q x D]
        """
        y = self.norm_1(x)
        y = self.film_1(y.permute(0, 2, 1), cond).permute(0, 2, 1)
        if self.flash_attn:
            with torch.autocast(y.device.type, dtype=torch.bfloat16):
                y = self.self_attn(y)[0]
        else:
            y, position_bias = self.self_attn(y, y, y, x_mask, position_bias)
        x = x + self.dropout(y)

        if self.is_decoder:
            y = self.norm_2(x)
            y = self.film_2(y.permute(0, 2, 1), cond).permute(0, 2, 1)
            y, encoder_decoder_position_bias = self.cross_attn(
                y, src, src, src_mask, encoder_decoder_position_bias
            )
            x = x + self.dropout(y)

        y = self.norm_3(x)
        y = self.film_3(
            y.permute(
                0,
                2,
                1,
            ),
            cond,
        ).permute(0, 2, 1)
        y = self.feed_forward(y)
        x = x + self.dropout(y)

        return x, position_bias, encoder_decoder_position_bias


class TransformerStack(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        d_cond: int = 64,
        n_heads: int = 8,
        n_layers: int = 8,
        last_layer: bool = True,
        bidirectional: bool = True,
        flash_attn: bool = False,
        is_decoder: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Store args
        self.bidirectional = bidirectional
        self.is_decoder = is_decoder

        # Create transformer layers
        # In T5, relative attention bias is shared by all layers in the stack
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_model,
                    d_cond,
                    n_heads,
                    bidirectional,
                    is_decoder,
                    has_relative_attention_bias=True if (i == 0) else False,
                    flash_attn=flash_attn,
                    dropout=dropout,
                )
                for i in range(n_layers)
            ]
        )

        # Perform last normalization
        self.norm = RMSNorm(d_model) if last_layer else None

    def subsequent_mask(self, size):
        return torch.ones(1, size, size).tril().bool()

    def forward(self, x, x_mask, cond=None, src=None, src_mask=None):
        """Computes a full transformer stack
        Parameters
        ----------
        x : Tensor[B x T_q x D]
        x_mask : Tensor[B x T_q]
        src : Tensor[B x T_kv x D], optional
        src_mask : Tensor[B x T_kv], optional
        Returns
        -------
        Tensor[B x T_q x D]
        """

        # Convert `src_mask` to (B x T_q x T_kv) shape for cross attention masking
        if self.is_decoder:
            src_mask = x_mask.unsqueeze(-1) * src_mask.unsqueeze(-2)

        # Convert `x_mask` to (B x T_q x T_q) shape for self attention masking
        x_mask = x_mask.unsqueeze(-2)
        if not self.bidirectional:
            x_mask = x_mask * self.subsequent_mask(x.size(1)).to(x_mask.device)

        # Initialize position biases
        position_bias = None
        encoder_decoder_position_bias = None

        # Compute transformer layers
        for layer in self.layers:
            x, position_bias, encoder_decoder_position_bias = layer(
                x=x,
                x_mask=x_mask,
                cond=cond,
                src=src,
                src_mask=src_mask,
                position_bias=position_bias,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
            )

        return self.norm(x) if self.norm is not None else x
    

class VampNet(nn.Module):
    def __init__(
        self,
        n_heads: int = 20,
        n_layers: int = 16,
        embedding_dim: int = 1280,
        vocab_size: int = 1024,
        n_classes: int = 6,
        flash_attn: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.flash_attn = flash_attn

        self.embedding = InputEmbedding(
            n_input=n_classes, 
            n_emb=vocab_size, 
            emb_dim=embedding_dim
        )

        self.transformer = TransformerStack(
            d_model=embedding_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            d_cond=0,
            last_layer=True,
            bidirectional=True,
            flash_attn=flash_attn,
            is_decoder=False,
            dropout=dropout,
        )

        # Add final conv layer
        self.classifier = SequentialWithFiLM(
            WNConv1d(
                embedding_dim,
                vocab_size,
                kernel_size=1,
                padding="same",
            ),
        )

    def forward(self, x):
        x = self.embedding(x)
        x_mask = torch.ones_like(x, dtype=torch.bool)[:, :1, :].squeeze(1)

        x = rearrange(x, "b d n -> b n d")
        out = self.transformer(x=x, x_mask=x_mask)
        out = rearrange(out, "b n d -> b d n")

        out = self.classifier(out, None) # no cond here!

        return out