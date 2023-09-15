import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm


# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def recurse_children(module, fn):
    for child in module.children():
        if isinstance(child, nn.ModuleList):
            for c in child:
                yield recurse_children(c, fn)
        if isinstance(child, nn.ModuleDict):
            for c in child.values():
                yield recurse_children(c, fn)

        yield recurse_children(child, fn)
        yield fn(child)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class SequentialWithFiLM(nn.Module):
    """
    handy wrapper for nn.Sequential that allows FiLM layers to be
    inserted in between other layers.
    """

    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    @staticmethod
    def has_film(module):
        mod_has_film = any(
            [res for res in recurse_children(module, lambda c: isinstance(c, FiLM))]
        )
        return mod_has_film

    def forward(self, x, cond):
        for layer in self.layers:
            if self.has_film(layer):
                x = layer(x, cond)
            else:
                x = layer(x)
        return x


class FiLM(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if input_dim > 0:
            self.beta = nn.Linear(input_dim, output_dim)
            self.gamma = nn.Linear(input_dim, output_dim)

    def forward(self, x, r):
        if self.input_dim == 0:
            return x
        else:
            beta, gamma = self.beta(r), self.gamma(r)
            beta, gamma = (
                beta.view(x.size(0), self.output_dim, 1),
                gamma.view(x.size(0), self.output_dim, 1),
            )
            x = x * (gamma + 1) + beta
        return x


class InputEmbedding(nn.Module):
    def __init__(self, n_input=6, n_emb=2048, emb_dim=32):
        super().__init__()

        self.n_emb = n_emb
        self.emb_dim = emb_dim

        self.emb_layers = nn.ModuleList([nn.Embedding(n_emb, emb_dim) for _ in range(n_input)])
        self.norm = nn.LayerNorm(emb_dim, elementwise_affine=False, eps=1e-6)
        self.conv = nn.Conv2d(emb_dim, emb_dim, 3, padding=1)

    def forward(self, x):
        embs = []
        for i, layer in enumerate(self.emb_layers):
            embs.append(layer(x[:, i, :]))

        embs = torch.stack(embs, dim=1)
        embs = self.norm(embs)
        embs = rearrange(embs, "b i l c -> b c i l")
        out = self.conv(embs)

        return out
