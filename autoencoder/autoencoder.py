import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import Autoencoder1D
from .discriminator import NLayerDiscriminator
import numpy as np


class Autoencoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        self.config = config
        self.model = Autoencoder1D(**config["model"])
        self.discriminator = NLayerDiscriminator()

    def encode(self, x):
        encoded, quantized, commit_loss = self.model.encode(x)

        self.assert_not_nan(encoded, "encoded")
        self.assert_not_nan(quantized, "quantized")

        return encoded, quantized, commit_loss

    def decode(self, x, sigmoid):
        dec = self.model.decode(x)
        self.assert_not_nan(dec, "decoded")

        if sigmoid:
            return F.sigmoid(dec)

        return dec

    def from_latent(self, quant):
        dec = self.decode(quant, sigmoid=True)

        return self._treshold_result(dec)

    def forward(self, x, inference=False):
        encoded, quantized, commit_loss = self.encode(x)

        if inference:
            return self.from_latent(quantized)

        dec = self.decode(quantized, sigmoid=False)

        return dec, commit_loss

    def assert_not_nan(self, x, where):
        if x.isnan().any():
            raise Exception(f"NaN found in {where}")

    def training_step(self, batch):
        images, labes = batch

        dec, commit_loss = self(images)
        ce_loss, disct_loss, g_loss = self.compute_loss(images, dec, use_weight=True)
        loss = ce_loss + commit_loss + g_loss * 0.2

        return loss, ce_loss, commit_loss, disct_loss, g_loss

    def compute_loss(self, label, pred, use_weight=False):
        pred = pred[:, None, :, :]
        ce_loss = self._compute_bce_loss_with_weight(
            pred, label, use_weight
        )

        # discr_loss, g_loss = self._compute_discriminator_loss(pred, label)
        discr_loss = 0
        g_loss = 0

        return ce_loss, discr_loss, g_loss

    def _compute_discriminator_loss(self, pred, label):
        discr_real = self.discriminator(label)
        discr_fake = self.discriminator(pred.sigmoid())

        discr_real_loss = torch.mean(F.relu(1 - discr_real))
        discr_fake_loss = torch.mean(F.relu(1 + discr_fake))

        discr_loss = (discr_real_loss + discr_fake_loss) * 0.5
        g_loss = -torch.mean(discr_fake)

        return discr_loss, g_loss

    def _compute_bce_loss_with_weight(self, pred, label, use_weight=False):
        sample_weight = 1

        if use_weight:
            sample_weight = torch.where(
                label != 0,
                4,
                2
            )

        loss = F.mse_loss(pred.sigmoid(), label, reduction='none')

        return (loss * sample_weight).mean()

    def _treshold_result(self, predicted):
        predicted_np = predicted.detach().cpu().numpy()
        # predicted_np = (predicted_np + 1) / 2

        recon_np = predicted_np[:, None, :, :]

        recon_np = np.where(
            recon_np > 0.87,
            np.ones_like(recon_np),
            np.zeros_like(recon_np)
        )

        return predicted_np