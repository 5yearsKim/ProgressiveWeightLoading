import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

from .swap_net import SwapNet


@dataclass
class DistillerOutput(ModelOutput):
    """
    Custom output for your distiller, so Trainer still
    knows to pick .loss and .logits, but you also expose
    loss_hard / loss_soft / loss_feat on outputs.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None

    loss_hard: torch.FloatTensor | None = None
    loss_soft: torch.FloatTensor | None = None
    loss_feat_sync: torch.FloatTensor | None = None
    loss_feat_recon: torch.FloatTensor | None = None
    loss_cross: torch.FloatTensor | None = None


BlockedModuleT = nn.Module


class FeatureDistiller(nn.Module):
    def __init__(
        self,
        swapnet: SwapNet,
        temp: float = 2.0,
        alpha: float = 0.4,
        w_sync: float = 2,
        w_recon: float = 2,
        w_cross: float = 0.5,
    ):
        """
        temp: temperature for KD
        alpha: weight for hard-label CE loss
        beta: weight for feature distillation loss
              (remaining (1-alpha-beta) goes to soft KD loss)
        """
        super().__init__()
        self.swapnet = swapnet
        self.swapnet.teacher.eval()
        self.swapnet.student.train()
        self.swapnet.encoders.train()
        self.swapnet.decoders.train()

        # Loss fns
        self.temp = temp
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.alpha = alpha
        self.w_sync = w_sync
        self.w_recon = w_recon
        self.w_cross = w_cross

    def forward(
        self, pixel_values: torch.Tensor, labels: torch.LongTensor
    ) -> DistillerOutput:
        # --- Student forward ---
        logits_s, features_s = self.swapnet.infer_student(
            pixel_values, return_features=True
        )  # [B, num_classes]

        # --- Teacher forward ---
        with torch.no_grad():
            logits_t, features_t = self.swapnet.infer_teacher(
                pixel_values, return_features=True
            )  # [B, num_classes]

        # 1) Hard-label CE
        loss_hard = self.alpha * self.ce(logits_s, labels)

        # 2) Soft-label KL
        T = self.temp
        p_s = F.log_softmax(logits_s / T, dim=-1)
        p_t = F.softmax(logits_t.detach() / T, dim=-1)
        loss_soft = (
            (1 - self.alpha) * F.kl_div(p_s, p_t, reduction="batchmean") * (T * T)
        )

        # 3) Feature-MSE between each tapped layer
        loss_feat_sync = 0.0
        loss_feat_recon = 0.0

        assert (
            len(features_s) == len(features_t) == self.swapnet.num_feat
        ), "Number of features in student and teacher must be the same"

        for i in range(self.swapnet.num_feat):
            feat_s, feat_t = features_s[i], features_t[i]
            # encoder, decoder = self.encoders[i], self.decoders[i]
            encoder, decoder = self.swapnet.encoders[i], self.swapnet.decoders[i]

            feat_target = encoder(feat_t)
            feat_recon = decoder(feat_target)

            loss_feat_sync += self.mse(feat_s, feat_target)
            loss_feat_recon += self.mse(feat_t, feat_recon)

        loss_feat_sync = self.w_sync * (loss_feat_sync / max(self.swapnet.num_feat, 1))
        loss_feat_recon = self.w_recon * (
            loss_feat_recon / max(self.swapnet.num_feat, 1)
        )

        while True:
            from_teachers = [
                random.choice([True, False]) for _ in range(self.swapnet.num_blocks)
            ]
            # ensure at least one True and one False
            if any(from_teachers) and not all(from_teachers):
                break

        cross_logits = self.swapnet(pixel_values, from_teachers)

        loss_cross = self.w_cross * self.ce(cross_logits, labels)

        # --- Combine all losses ---
        total_loss = (
            loss_hard + loss_soft + loss_feat_sync + loss_feat_recon + loss_cross
        )

        return DistillerOutput(
            logits=logits_s,
            loss=total_loss,
            loss_hard=loss_hard,
            loss_soft=loss_soft,
            loss_feat_sync=loss_feat_sync,
            loss_feat_recon=loss_feat_recon,
            loss_cross=loss_cross,
        )
