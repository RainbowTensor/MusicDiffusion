import torch
from torch import nn
import torch.nn.functional as F


class MusicDiffusion(nn.Module):
    def __init__(self, autoencoder, diffusion_model, noise_scheduler):
        super().__init__()

        self.autoencoder = autoencoder
        self.diffusion_model = diffusion_model
        self.noise_scheduler = noise_scheduler

    def forward(self, image, timestep, class_labels):
        return self.diffusion_model(image, timestep, class_labels)

    def training_step(self, batch):
        images, class_labels = batch

        timestep = 1-torch.rand(images.size(0), device=images.device)
        latents, _, _ = self.autoencoder.encode(images)
        noised_latents, noise = self.noise_scheduler.diffuse(latents, timestep)

        pred_noise = self(noised_latents, timestep, class_labels)
        loss = F.mse_loss(pred_noise, noise)

        return loss