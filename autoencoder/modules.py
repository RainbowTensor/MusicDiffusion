import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
from einops import rearrange

from .vq import VectorQuantizer
from .nn import conv_nd, avg_pool_nd


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels, add_conv=False, num_groups=32):   
    return nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


class FeedForward(nn.Module):
    def __init__(self, dim, out_dim=None, dropout=0.):
        super().__init__()
        if out_dim is None:
            out_dim = dim

        hidden_dim = dim * 2

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, embedding_dim, seq_len]``
        """
        x = x.permute(0, 2, 1)
        return self.net(x).permute(0, 2, 1)
    

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
    

class PatchMerger(nn.Module):
    def __init__(self, dim, num_tokens_out):
        super().__init__()
        self.scale = dim ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.queries = nn.Parameter(torch.randn(num_tokens_out, dim))

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, embedding_dim, seq_len]``
        """
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        sim = torch.matmul(self.queries, x.transpose(-1, -2)) * self.scale
        attn = sim.softmax(dim = -1)

        return torch.matmul(attn, x).permute(0, 2, 1)


class AttnBlock(nn.Module):
    def __init__(self, in_channels, add_conv=False, dim=2):
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim

        self.norm = Normalize(in_channels, add_conv=add_conv)
        self.q = conv_nd(
            dim, in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = conv_nd(
            dim, in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = conv_nd(
            dim, in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = conv_nd(
            dim, in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x   # b, c, w
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
    
        if self.dim == 2:
            b, c, h, w = q.shape
        else:
            b, c, w = q.shape
            h = 1
    
        # compute attention
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))

        if torch.isinf(w_).any():
            clamp_value = torch.finfo(w_.dtype).max - 1000
            w_ = torch.clamp(w_, min=-clamp_value, max=clamp_value)

        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        
        if self.dim == 2:
            h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Encoder(nn.Module):
    def __init__(
        self,
        img_height=128,
        seq_len=256,
        in_channels=2,
        embedding_channels=16,
        block_out_channels=[128, 256],
        attn_per_block=2,
        z_channels=32,
        dropout=0.1,
    ):
        super().__init__()

        self.embed = nn.Sequential(
            conv_nd(2, in_channels, embedding_channels // 2, 3, padding=1),
            nn.GELU(),
            conv_nd(2, embedding_channels // 2, embedding_channels, 3, padding=1),
        )

        self.flatten_proj = nn.Conv1d(embedding_channels * img_height, block_out_channels[0] // 2, 3, padding=1)
        self.pos_encoding = PositionalEncoding(block_out_channels[0] // 2)

        num_tokens_out = seq_len
        down_modules = []
        for block_channels in block_out_channels:
            block_in_channels = block_channels // 2
            num_tokens_out = num_tokens_out // 2
            down_modules.append(
                PatchMerger(block_in_channels, num_tokens_out=num_tokens_out)
            )

            down_modules.append(
                FeedForward(
                    block_in_channels,
                    out_dim=block_channels,
                    dropout=dropout
                )
            )

            down_modules.extend(
                [
                    AttnBlock(in_channels=block_channels, add_conv=False, dim=1)
                    for _ in range(attn_per_block)
                ]
            )

        self.down_modules = nn.Sequential(*down_modules)

        self.norm_out = Normalize(block_out_channels[-1])
        self.conv_out = self.conv_out = conv_nd(1, block_out_channels[-1], z_channels, 3, padding=1)

    def forward(self, x):
        #onset, duration = torch.chunk(x, 2, dim=1)
        #onset, duration = onset.squeeze(1).permute(0, 2, 1), duration.squeeze(1).permute(0, 2, 1)

        #h_onset = self.onset_conv_in(onset)
        #h_duration = self.duration_conv_in(onset)

        x = x.permute(0, 1, 3, 2)
        h = self.embed(x)

        B, C, H, W = h.shape
        h = h.reshape([B, C * H, W])

        h = self.flatten_proj(h)
        h = self.pos_encoding(h.permute(2, 0, 1))
        
        h = self.down_modules(h.permute(1, 2, 0))

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class Decoder(nn.Module):
    def __init__(
        self,
        img_height=128,
        seq_len=256,
        block_out_channels=[256, 128],
        attn_per_block=2,
        z_channels=32,
        dropout=0.1,
    ):
        super().__init__()
        
        seq_len_downsampled = seq_len / 2 ** len(block_out_channels)
        self.conv_in = conv_nd(1, z_channels, block_out_channels[0], 3, padding=1)

        num_tokens_out = seq_len_downsampled
        up_modules = []
        for i, block_channels in enumerate(block_out_channels):
            block_in_channels = block_channels if i == 0 else block_channels * 2
            num_tokens_out = num_tokens_out * 2

            up_modules.append(
                PatchMerger(block_in_channels, num_tokens_out=num_tokens_out)
            )

            up_modules.append(
                FeedForward(
                    block_in_channels,
                    out_dim=block_channels,
                    dropout=dropout
                )
            )

            up_modules.extend(
                [
                    AttnBlock(in_channels=block_channels, add_conv=False, dim=1)
                    for _ in range(attn_per_block)
                ]
            )

        self.up_modules = nn.Sequential(*up_modules)

        self.norm_out = Normalize(block_out_channels[-1], add_conv=False)
        self.conv_out_onset = FeedForward(block_out_channels[-1], out_dim=img_height, dropout=dropout)
        self.conv_out_duration = FeedForward( block_out_channels[-1], out_dim=img_height, dropout=dropout)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.up_modules(h)

        h = self.norm_out(h)
        h = nonlinearity(h)

        out_onset = self.conv_out_onset(h)
        out_dur = self.conv_out_duration(h)

        return out_onset, out_dur
    

class Autoencoder1D(nn.Module):
    def __init__(
        self,
        img_height=128,
        in_channels=2,
        embedding_channels=16,
        block_out_channels=[128, 256],
        attn_per_block=2,
        z_channels=32,
        n_embed=2048,
        embed_dim=32,
        dropout=0.1,
    ):
        super().__init__()

        self.encoder = Encoder(
            img_height=img_height,
            in_channels=in_channels,
            embedding_channels=embedding_channels,
            block_out_channels=block_out_channels,
            attn_per_block=attn_per_block,
            z_channels=z_channels,
            dropout=dropout,
        )

        self.decoder = Decoder(
            img_height=img_height,
            block_out_channels=list(reversed(block_out_channels)),
            attn_per_block=attn_per_block,
            z_channels=z_channels,
            dropout=dropout,
        )

        self.quantize = VectorQuantizer(
            n_embed, embed_dim, beta=0.25, remap=None, sane_index_shape=False
        )

        self.quant_conv = conv_nd(1, z_channels, embed_dim, 1)
        self.post_quant_conv = conv_nd(1, embed_dim, z_channels, 1)

    def encode(self, x):
        encoded = F.normalize(self.encoder(x))
        encoded_conv = self.quant_conv(encoded)

        quant, emb_loss, info = self.quantize(encoded_conv[:, :, None, :])
        quant = quant.squeeze(2)

        return encoded, quant, emb_loss

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        batch_size = code_b.shape[0]
        quant = self.quantize.embedding(code_b.flatten())
        grid_size = int((quant.shape[0] // batch_size) ** 0.5)
        quant = quant.view((1, 32, 32, 4))
        quant = rearrange(quant, "b h w c -> b c h w").contiguous()
        print(quant.shape)
        quant2 = self.post_quant_conv(quant)
        dec = self.decoder(quant2, quant)
        return dec

    def forward(self, input):
        _, quant, _ = self.encode(input)
        dec = self.decode(quant)
        return dec