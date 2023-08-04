import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from .vq import VectorQuantizer
from .nn import conv_nd, avg_pool_nd, unpatchify
from .regularization import L1
from .hourglass_transformer import (
    Transformer, FeedForward, PositionalEncoding, PatchEmbedding
)


def nonlinearity(x):
    return F.gelu(x)


def Normalize(in_channels, add_conv=False, num_groups=8):   
    return nn.BatchNorm2d(in_channels)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv, dim=2):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = conv_nd(
                dim, in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv, dim=2):
        super().__init__()
        self.with_conv = with_conv
        self.dim = dim
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = conv_nd(
                dim, in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

        self.avg_pool = avg_pool_nd(
            dim, kernel_size=2, stride=2
        )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1) if self.dim == 1 else (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = self.avg_pool(x)
        return x


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
        h = self.norm1(x)
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

        return x + h


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=2,
        embedding_channels=16,
        transformer_channels=512,
        block_out_channels=[128, 256],
        attn_per_block=2,
        z_channels=32,
        patch_size=4,
        dropout=0.1,
        ff_mult=2
    ):
        super().__init__()

        self.embed = ResnetBlock(in_channels=in_channels, out_channels=embedding_channels, dim=2, dropout=dropout)

        down_block = []
        block_in_channels = embedding_channels
        for block_channels in block_out_channels:
            down_block.extend([
                ResnetBlock(in_channels=block_in_channels, out_channels=block_channels, dim=2, dropout=dropout),
                Downsample(block_channels, with_conv=True)
            ])

            block_in_channels = block_channels

        self.down_block = nn.Sequential(*down_block)

        self.patchify = PatchEmbedding(patch_size, transformer_channels)
        self.pos_encoding = PositionalEncoding(transformer_channels)

        self.trnaformer_block = Transformer(
            dim=transformer_channels, depth=attn_per_block, ff_mult=ff_mult, dropout=dropout
        )

        self.norm_out = nn.LayerNorm(transformer_channels)
        self.to_out = FeedForward(dim=transformer_channels, out_dim=z_channels, mult=ff_mult)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, channels, width, height]``
        """
        h = self.embed(x)
        h = self.down_block(h)

        h = self.patchify(h)
        h = self.pos_encoding(h.permute(1, 0, 2))   # L, B, C
        
        h = self.trnaformer_block(h.permute(1, 0, 2))

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.to_out(h)

        return h

class Decoder(nn.Module):
    def __init__(
        self,
        transformer_channels=512,
        block_out_channels=[256, 128],
        attn_per_block=2,
        z_channels=32,
        patch_size=4,
        dropout=0.1,
        ff_mult=2,
        img_size=(256, 128),
    ):
        super().__init__()

        w, h = img_size
        self.w = (w // 2**len(block_out_channels)) // patch_size
        self.h = h // 2**len(block_out_channels) // patch_size

        self.patch_size = patch_size
        self.proj_in = FeedForward(dim=z_channels, out_dim=transformer_channels, mult=ff_mult)

        self.trnaformer_block = Transformer(
            dim=transformer_channels, depth=attn_per_block, ff_mult=ff_mult, dropout=dropout
        )
        self.to_patchify = nn.Linear(transformer_channels, (patch_size**2) * transformer_channels)

        up_block = []
        block_in_channels = transformer_channels
        for block_channels in block_out_channels:
            up_block.extend([
                ResnetBlock(in_channels=block_in_channels, out_channels=block_channels, dim=2, dropout=dropout),
                Upsample(block_channels, with_conv=True)
            ])

            block_in_channels = block_channels

        self.up_block = nn.Sequential(*up_block)

        self.norm_out = Normalize(block_channels, add_conv=False)
        self.conv_out_onset = ResnetBlock(in_channels=block_channels, out_channels=1, dim=2, dropout=dropout)
        self.conv_out_duration = ResnetBlock(in_channels=block_channels, out_channels=1, dim=2, dropout=dropout)

    def forward(self, z):
        """
        Arguments:
            z: Tensor, shape ``[batch_size, width, channels]``
        """
        h = self.proj_in(z)

        h = self.trnaformer_block(h)

        h = self.to_patchify(h)
        h = unpatchify(h, self.patch_size, self.h, self.w)
        h = self.up_block(h)

        h = self.norm_out(h)
        h = nonlinearity(h)

        out_onset = self.conv_out_onset(h).squeeze(1).permute(0, 2, 1)
        out_dur = self.conv_out_duration(h).squeeze(1).permute(0, 2, 1)

        return out_onset, out_dur
    

class Autoencoder1D(nn.Module):
    def __init__(
        self,
        in_channels=2,
        embedding_channels=16,
        transformer_channels=512,
        block_out_channels=[128, 256],
        attn_per_block=2,
        z_channels=32,
        n_embed=2048,
        embed_dim=32,
        patch_size=4,
        img_size=(256, 128),
        ff_mult=2,
        dropout=0.1,
    ):
        super().__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            transformer_channels=transformer_channels,
            embedding_channels=embedding_channels,
            block_out_channels=block_out_channels,
            attn_per_block=attn_per_block,
            z_channels=z_channels,
            patch_size=patch_size,
            ff_mult=ff_mult,
            dropout=dropout,
        )

        self.decoder = Decoder(
            transformer_channels=transformer_channels,
            block_out_channels=list(reversed(block_out_channels)),
            attn_per_block=attn_per_block,
            z_channels=z_channels,
            ff_mult=ff_mult,
            patch_size=patch_size,
            img_size=img_size,
            dropout=dropout,
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