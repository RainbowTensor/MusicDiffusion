import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import Autoencoder1D
import numpy as np


class Autoencoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        self.config = config
        self.model = Autoencoder1D(**config["model"])

    def encode(self, x):
        encoded, quantized, commit_loss, indices = self.model.encode(x)

        self.assert_not_nan(encoded, "encoded")
        self.assert_not_nan(quantized, "quantized")

        return encoded, quantized, indices, commit_loss

    def decode(self, x, sigmoid):
        dec_onset, dec_duration = self.model.decode(x)
        self.assert_not_nan(dec_onset, "dec_onset")
        self.assert_not_nan(dec_duration, "dec_duration")

        if sigmoid:
            return F.sigmoid(dec_onset), F.sigmoid(dec_duration)

        return dec_onset, dec_duration

    def from_latent(self, quant):
        dec = self.decode(quant, sigmoid=True)

        return self._treshold_result(dec)

    def forward(self, x, inference=False):
        encoded, quantized, _, commit_loss = self.encode(x)

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
        ce_loss, mse_loss = self.compute_loss(images, dec, use_weight=True)
        loss = ce_loss + commit_loss + mse_loss

        return loss, ce_loss, mse_loss, commit_loss

    def compute_loss(self, label, pred, use_weight=False):
        pred_onset, pred_duration = pred
        pred_onset = pred_onset[:, None, :, :]
        pred_duration = pred_duration[:, None, :, :]

        label_onset = torch.where(
            label > 0,
            torch.ones_like(label),
            torch.zeros_like(label),
        )

        ce_loss = self._compute_loss_with_weight(
            pred_onset, label_onset, loss_fn=F.binary_cross_entropy_with_logits, use_weight=use_weight
        )

        mse_loss = self._compute_loss_with_weight(
            pred_duration.sigmoid(), label, loss_fn=F.mse_loss, use_weight=use_weight
        )

        return ce_loss, mse_loss * 4

    def _compute_loss_with_weight(self, pred, label, loss_fn=F.mse_loss, use_weight=False):
        sample_weight = 1

        if use_weight:
            sample_weight = torch.where(
                label != 0,
                3,
                3
            )

        loss = loss_fn(pred, label, reduction='none')

        return (loss * sample_weight).mean()

    def _treshold_result(self, predicted):
        predicted_onset, predicted_duration = predicted
        predicted_np = predicted_onset.detach().cpu().numpy()
        # predicted_np = (predicted_np + 1) / 2

        recon_np = predicted_np[:, None, :, :]

        recon_np = np.where(
            recon_np > 0.87,
            np.ones_like(recon_np),
            np.zeros_like(recon_np)
        )

        return predicted_np