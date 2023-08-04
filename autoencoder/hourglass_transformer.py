import torch
from torch import nn, einsum
from einops import rearrange
import math

def FeedForward(dim, out_dim=None, mult=4, dropout=0.):
    if out_dim is None:
        out_dim = dim

    hidden_dim = dim * mult

    net = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim)
    )

    def _init_weights(layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)

    net.apply(_init_weights)

    return net


class LinearDownsample(nn.Module):
    def __init__(self, dim, shorten_factor):
        super().__init__()
        self.proj = nn.Linear(dim * shorten_factor, dim)
        self.shorten_factor = shorten_factor

    def forward(self, x):
        x = rearrange(x, 'b (n s) d -> b n (s d)', s=self.shorten_factor)
        return self.proj(x)


class LinearUpsample(nn.Module):
    def __init__(self, dim, shorten_factor):
        super().__init__()
        self.proj = nn.Linear(dim, dim * shorten_factor)
        self.shorten_factor = shorten_factor

    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, 'b n (s d) -> b (n s) d', s=self.shorten_factor)


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def forward(self, x):
        h = self.heads
        kv_input = x

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)
    
    def init_weights(self):
        def _init_weights(layer):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                
        self.apply(_init_weights)


class Transformer(nn.Module):
    def __init__(
        self,
        dim=512,
        depth=8,
        heads=8,
        dropout=0.,
        ff_mult=2,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormResidual(dim, Attention(dim, heads=heads, dim_head=dim // heads, dropout=dropout)),
                PreNormResidual(dim, FeedForward(dim, mult=ff_mult, dropout=dropout))
            ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)

        return self.norm(x)
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=16, num_hiddens=512):
        super().__init__()
        if not isinstance(patch_size, (list, tuple)):
            patch_size = (patch_size, patch_size)

        self.conv = nn.LazyConv2d(num_hiddens, kernel_size=patch_size,
                                  stride=patch_size)

    def forward(self, X):
        return self.conv(X).flatten(2).transpose(1, 2)