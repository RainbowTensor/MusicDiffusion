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
        dec, dec_onset = self.model.decode(x)
        self.assert_not_nan(dec, "dec")
        self.assert_not_nan(dec_onset, "dec_onset")

        if sigmoid:
            return F.sigmoid(dec), F.sigmoid(dec_onset)

        return dec, dec_onset

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
        images, targets = batch

        dec, commit_loss = self(images)
        mse_loss, ce_loss = self.compute_loss(targets, dec, use_weight=False)
        loss = mse_loss + (0.2 * ce_loss) + commit_loss

        return loss, mse_loss, ce_loss, commit_loss

    def compute_loss(self, label, pred, use_weight=False):
        pred, pred_onset = pred
        pred_onset = pred_onset[:, None, :, :]
        pred = pred[:, None, :, :]

        label_onset = torch.where(
            label > 0,
            torch.ones_like(label),
            torch.zeros_like(label),
        )

        ce_loss = self._compute_loss_with_weight(
            pred_onset, label_onset, loss_fn=F.binary_cross_entropy_with_logits, use_weight=use_weight
        )

        mse_loss = self._compute_loss_with_weight(
            pred.sigmoid(), label, loss_fn=F.l1_loss, use_weight=use_weight
        )

        return mse_loss, ce_loss

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

    def inverse_huber_loss(self, output, target):
        absdiff = torch.abs(output - target)
        C = 0.2 * torch.max(absdiff).item()

        return torch.where(
            absdiff < C,
            absdiff,
            (absdiff * absdiff + C * C) / (2 * C)
        )

    def _treshold_result(self, predicted):
        pred, pred_onset = predicted

        pred_onset_np = pred_onset.squeeze().detach().cpu().numpy()
        # predicted_np = (predicted_np + 1) / 2

        recon_np = pred_onset_np[:, None, :, :]

        # recon_np = np.where(
        #     recon_np > 0.87,
        #     np.ones_like(recon_np),
        #     np.zeros_like(recon_np)
        # )

        return pred_onset_np
