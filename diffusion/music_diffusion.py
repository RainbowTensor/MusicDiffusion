import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from data.utils import generate_source, generate_target
from data.consts import EMPTY_TOKEN, MASK_TOKEN


class MusicDiffusion(nn.Module):
    def __init__(self, autoencoder, diffusion_model):
        super().__init__()

        self.autoencoder = autoencoder
        self.diffusion_model = diffusion_model
        self.criterion = nn.CrossEntropyLoss(
            reduction='none', label_smoothing=0.1,
        )

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

        pianoroll_merge = rearrange(
            pianoroll_batched, "b i c w h -> (b i) c w h")

        with torch.no_grad():
            _, _, indices, _ = self.autoencoder.encode(pianoroll_merge)
        indices_batched = indices.reshape(B, I, -1)

        target, target_mask = generate_target(indices_batched, valid_instr)
        source, source_mask = generate_source(
            indices_batched, target_mask, valid_instr)

        instr_labels = target_mask.sum(-1).bool().float().argmax(-1)

        timestep = 1 - torch.rand(source.shape[0], device=source.device)
        noised_source, mask = self.diffusion_model.add_noise(source, timestep)

        input = (source * (1 - target_mask)) + (target_mask * noised_source)
        pred = self(input, timestep, instr_labels)

        mask = torch.where(
            target == EMPTY_TOKEN,
            torch.zeros_like(mask),
            mask
        ).sum(1)

        target = torch.where(
            target == EMPTY_TOKEN,
            torch.zeros_like(target),
            target
        ).sum(1)

        loss_weight = self.diffusion_model.get_loss_weight(timestep, mask)
        loss = self.criterion(pred, target).mean()
        loss = (
            (loss * loss_weight).sum(dim=[1, 2]) / loss_weight.sum(dim=[1, 2])).mean()

        return loss
