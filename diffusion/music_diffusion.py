import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from ..data.utils import generate_source, generate_target


class MusicDiffusion(nn.Module):
    def __init__(self, autoencoder, diffusion_model):
        super().__init__()

        self.autoencoder = autoencoder
        self.diffusion_model = diffusion_model

    def forward(self, image, timestep, instr_labels):
        return self.diffusion_model(image, timestep, instr_labels)

    def training_step(self, batch):
        pianoroll_batched = batch
        input_shape = pianoroll_batched.shape
        B, I, _, _, _ = input_shape

        note_count_instr = pianoroll_batched[:, :, 0, :, :].sum(-1).sum(-1)
        valid_instr = torch.where(
            note_count_instr > 0,
            torch.ones_like(note_count_instr),
            torch.zeros_like(note_count_instr)
        )

        pianoroll_merge = rearrange(pianoroll_batched, "b i c w h -> (b i) c w h")

        with torch.no_grad():
            _, _, indices, _ = self.autoencoder.encode(pianoroll_merge)
        indices_batched = indices.reshape(B, I, -1)

        target, target_mask = generate_target(indices_batched, valid_instr)
        source, _ = generate_source(indices_batched, target_mask, valid_instr)

        instr_labels = target_mask.sum(-1).bool().float().argmax(-1)

        timestep = 1 - torch.rand(source.shape[0], device=source.device)
        noised_source, _ = self.diffusion_model.add_noise(source, timestep)
        pred = self(noised_source, timestep, instr_labels)

        loss = F.cross_entropy(pred, target)

        return loss