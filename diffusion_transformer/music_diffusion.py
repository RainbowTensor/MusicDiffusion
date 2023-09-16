from scipy.ndimage import label
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from .mask import generate_mask, apply_mask
from data.utils import generate_source, generate_target, flatten_tensor
from data.consts import EMPTY_TOKEN, MASK_TOKEN


class MusicDiffusion(nn.Module):
    def __init__(self, autoencoder, diffusion_model):
        super().__init__()

        self.autoencoder = autoencoder
        self.diffusion_model = diffusion_model
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=0.1,
        )

    def forward(self, image, target_condition):
        return self.diffusion_model(image, target_condition)

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
        source, source_mask = generate_source(indices_batched, target_mask, valid_instr)

        timestep = 1 - torch.rand(source.shape[0], device=source.device)
        mask = generate_mask(source, timestep)
        masked_input, _ = apply_mask(source, mask, MASK_TOKEN)
        target_condition = source_mask - target_mask

        input = (source * (1 - target_mask)) + (target_mask * masked_input)
        input = flatten_tensor(input)
        target_condition = flatten_tensor(target_condition)
        target = flatten_tensor(target)

        pred = self(input, target_condition)

        loss = self.criterion(pred, target)

        return loss