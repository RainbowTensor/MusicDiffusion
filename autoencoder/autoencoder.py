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
        encoded, quantized, commit_loss = self.model.encode(x)

        self.assert_not_nan(encoded, "encoded")
        self.assert_not_nan(quantized, "quantized")

        return encoded, quantized, commit_loss

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
        encoded, quantized, commit_loss = self.encode(x)

        if inference:
            return self.from_latent(quantized)

        dec_onset, dec_duration = self.decode(quantized, sigmoid=False)

        return (dec_onset, dec_duration), commit_loss

    def assert_not_nan(self, x, where):
        if x.isnan().any():
            raise Exception(f"NaN found in {where}")

    def training_step(self, batch):
        images, labes = batch

        dec, commit_loss = self(images)
        loss = self.compute_loss(images, dec, use_weight=True)
        loss = loss + commit_loss

        return loss, commit_loss

    def compute_loss(self, input, output, use_weight=False):
        pred_onsets, pred_durations = output

        onsets_label = input[:, 0, :, :]
        duration_label = input[:, 1, :, :]

        ce_onset = self._compute_bce_loss_with_weight(
            pred_onsets, onsets_label, use_weight
        )
        ce_duration = self._compute_bce_loss_with_weight(
            pred_durations, duration_label, use_weight
        )

        return (ce_onset + ce_duration)

    def _compute_bce_loss_with_weight(self, pred, label, use_weight=False):
        sample_weight = 1

        if use_weight:
            sample_weight = torch.where(
                label == 1,
                1.5,
                0.5
            )

        loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')

        return (loss * sample_weight).mean()

    def _treshold_result(self, predicted):
        onset, dur = predicted

        onset_np = onset.detach().cpu().numpy()
        dur_np = dur.detach().cpu().numpy()

        recon_np = np.stack(
            [onset_np[:, None, :, :], dur_np[:, None, :, :]], 
            axis=1
        )

        recon_np = np.where(
            recon_np > 0.87,
            np.ones_like(recon_np),
            np.zeros_like(recon_np)
        )

        return onset_np
