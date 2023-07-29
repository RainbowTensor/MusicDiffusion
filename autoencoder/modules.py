import torch
from torch import nn, einsum
import torch.nn.functional as F
import math
import torchlayers as tl
from einops import rearrange

from .vq import VectorQuantizer
from .nn import conv_nd, avg_pool_nd


def nonlinearity(x):
    return F.gelu(x)


def Normalize(in_channels, add_conv=False, num_groups=32):   
    return nn.BatchNorm1d(in_channels)


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


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
        add_conv=False,
        dim=2
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, add_conv=add_conv)
        self.conv1 = conv_nd(
            dim, in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels, add_conv=add_conv)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = conv_nd(
            dim, out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = conv_nd(
                    dim, in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = conv_nd(
                    dim, in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb=None):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x.permute(0, 2, 1)
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return (x + h).permute(0, 2, 1)


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


class Encoder(nn.Module):
    def __init__(
        self,
        img_height=128,
        in_channels=2,
        embedding_channels=16,
        block_out_channels=[128, 256],
        attn_per_block=2,
        z_channels=32,
        dropout=0.1,
        ff_mult=2
    ):
        super().__init__()

        self.embed = nn.Sequential(
            conv_nd(2, in_channels, embedding_channels // 4, 3, padding=1),
            nn.GELU(),
            conv_nd(2, embedding_channels // 4, embedding_channels // 2, 3, padding=1),
            nn.GELU(),
            conv_nd(2, embedding_channels // 2, embedding_channels, 3, padding=1),
        )

        self.flatten_proj = conv_nd(1, embedding_channels * img_height, block_out_channels[0] // 2, 3, padding=1)
        self.pos_encoding = PositionalEncoding(block_out_channels[0] // 2)

        down_modules = []
        for block_channels in block_out_channels:
            block_in_channels = block_channels // 2
            down_modules.append(
                LinearDownsample(dim=block_in_channels, shorten_factor=2)
            )

            down_modules.append(
                ResnetBlock(
                    in_channels=block_in_channels,
                    out_channels=block_channels,
                    dim=1,
                    dropout=dropout
                )
            )

            down_modules.append(
                Transformer(
                    dim=block_channels,
                    depth=attn_per_block,
                    ff_mult=ff_mult,
                    dropout=dropout
                )
            )

        self.down_modules = nn.Sequential(*down_modules)

        self.norm_out = Normalize(block_out_channels[-1])
        self.to_out = FeedForward(dim=block_out_channels[-1], out_dim=z_channels, mult=ff_mult)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, channels, width, height]``
        """

        x = x.permute(0, 1, 3, 2)
        h = self.embed(x)

        B, C, H, W = h.shape
        h = h.reshape([B, C * H, W])#.permute(0, 2, 1)   # B, C, W

        h = self.flatten_proj(h).permute(0, 2, 1)   # B, W, C
        h = self.pos_encoding(h.permute(1, 0, 2))   # W, B, C
        
        h = self.down_modules(h.permute(1, 0, 2))

        # end
        h = self.norm_out(h.permute(0, 2, 1))
        h = nonlinearity(h.permute(0, 2, 1))
        h = self.to_out(h)

        return h

class Decoder(nn.Module):
    def __init__(
        self,
        img_height=128,
        block_out_channels=[256, 128],
        attn_per_block=2,
        z_channels=32,
        dropout=0.1,
        ff_mult=2,
        weight_decay=1e-4
    ):
        super().__init__()
        self.proj_in = FeedForward(dim=z_channels, out_dim=block_out_channels[0], mult=ff_mult)

        up_modules = []
        for i, block_channels in enumerate(block_out_channels):
            block_in_channels = block_channels if i == 0 else int(block_channels * 2)

            up_modules.append(
                LinearUpsample(dim=block_in_channels, shorten_factor=2)
            )

            up_modules.append(
                ResnetBlock(
                    in_channels=block_in_channels,
                    out_channels=block_channels,
                    dim=1,
                    dropout=dropout
                )
            )

            up_modules.append(
                Transformer(
                    dim=block_channels,
                    depth=attn_per_block,
                    ff_mult=ff_mult,
                    dropout=dropout
                )
            )

        self.up_modules = nn.Sequential(*up_modules)

        self.norm_out = Normalize(block_out_channels[-1], add_conv=False)
        self.conv_out_onset = tl.L1(FeedForward(dim=block_out_channels[-1], out_dim=img_height, mult=ff_mult), weight_decay=weight_decay)
        self.conv_out_duration = tl.L1(FeedForward(dim=block_out_channels[-1], out_dim=img_height, mult=ff_mult), weight_decay=weight_decay)

    def forward(self, z):
        """
        Arguments:
            z: Tensor, shape ``[batch_size, width, channels]``
        """
        h = self.proj_in(z)
        h = self.up_modules(h)

        h = self.norm_out(h.permute(0, 2, 1))
        h = nonlinearity(h.permute(0, 2, 1))

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
        ff_mult=2,
        dropout=0.1,
        weight_decay=1e-4
    ):
        super().__init__()

        self.encoder = Encoder(
            img_height=img_height,
            in_channels=in_channels,
            embedding_channels=embedding_channels,
            block_out_channels=block_out_channels,
            attn_per_block=attn_per_block,
            z_channels=z_channels,
            ff_mult=ff_mult,
            dropout=dropout,
        )

        self.decoder = Decoder(
            img_height=img_height,
            block_out_channels=list(reversed(block_out_channels)),
            attn_per_block=attn_per_block,
            z_channels=z_channels,
            ff_mult=ff_mult,
            dropout=dropout,
            weight_decay=weight_decay
        )

        self.quantize = VectorQuantizer(
            n_embed, embed_dim, beta=0.25, remap=None, sane_index_shape=False
        )

        self.quant_conv = conv_nd(1, z_channels, embed_dim, 1)
        self.post_quant_conv = conv_nd(1, embed_dim, z_channels, 1)

    def encode(self, x):
        encoded = self.encoder(x)
        encoded_conv = self.quant_conv(encoded.permute(0, 2, 1))

        quant, emb_loss, info = self.quantize(encoded_conv[:, :, None, :])
        quant = quant.squeeze(2)

        return encoded, quant, emb_loss

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant.permute(0, 2, 1))
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