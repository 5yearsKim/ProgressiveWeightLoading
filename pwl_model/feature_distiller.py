from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput
from .layers.feature_converter import FeatureConverter


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


class FeatureDistiller(nn.Module):
    def __init__(
        self,
        student,
        teacher,
        input_shape: tuple = (3, 32, 32),
        student_ir: list[nn.Module] = [],
        teacher_ir: list[nn.Module] = [],
        temp: float = 2.0,
        alpha: float = 0.5,
        w_sync: float = 0.3,
        w_recon: float = 0.3,


    ):
        """
        temp: temperature for KD
        alpha: weight for hard-label CE loss
        beta: weight for feature distillation loss
              (remaining (1-alpha-beta) goes to soft KD loss)
        """
        super().__init__()
        # --- Teacher setup (frozen) ---
        self.teacher = teacher
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # --- Student setup ---
        self.student = student
        self.student.train()

        assert len(teacher_ir) == len(student_ir)
        self.num_ir = len(teacher_ir)

        # Storage for hooked features
        self._feat_t: list[torch.Tensor] = [None] * self.num_ir
        self._feat_s: list[torch.Tensor] = [None] * self.num_ir

        

        def _get_hook(storage: list, idx: int) -> callable:
            def hook(module, inp, out):
                # out: Tensor of shape [B, C, H, W]
                storage[idx] = out

            return hook

        # Register hooks
        for i, mod in enumerate(teacher_ir):
            mod.register_forward_hook(_get_hook(self._feat_t, i))
        for i, mod in enumerate(student_ir):
            mod.register_forward_hook(_get_hook(self._feat_s, i))

        with torch.no_grad():
            device = next(self.teacher.parameters()).device
            dummy = torch.zeros((1, *input_shape)).to(device)
            _ = self.teacher(dummy)
            _ = self.student(dummy)

        self.encoders = nn.ModuleList([])
        self.decoders = nn.ModuleList([])

        for feat_s, feat_t in zip(self._feat_s, self._feat_t):
            chan_t, chan_s = feat_t.shape[1], feat_s.shape[1]

            if chan_t == chan_s:
                self.encoders.append(nn.Identity())
                self.decoders.append(nn.Identity())
            elif chan_t > chan_s:
                self.encoders.append(FeatureConverter(chan_t, chan_s))
                self.decoders.append(FeatureConverter(chan_s, chan_t))
            else:
                raise ValueError(f"chan_s {chan_s} should be less than chan_t {chan_t}")
            
        # Loss fns
        self.temp = temp
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.alpha = alpha
        self.w_sync = w_sync
        self.w_recon = w_recon

    def forward(
        self, pixel_values: torch.Tensor, labels: torch.LongTensor
    ) -> DistillerOutput:
        # --- Student forward ---
        out_s = self.student(pixel_values)
        logits_s = out_s.logits  # [B, num_classes]

        # --- Teacher forward ---
        with torch.no_grad():
            out_t = self.teacher(pixel_values)
        logits_t = out_t.logits  # [B, num_classes]

        # 1) Hard-label CE
        loss_hard = self.alpha * self.ce(logits_s.detach(), labels)

        # 2) Soft-label KL
        T = self.temp
        p_s = F.log_softmax(logits_s / T, dim=-1)
        p_t = F.softmax(logits_t / T, dim=-1)
        loss_soft = (
            (1 - self.alpha) * F.kl_div(p_s, p_t, reduction="batchmean") * (T * T)
        )

        # 3) Feature-MSE between each tapped layer
        loss_feat_sync = 0.0
        loss_feat_recon = 0.0

        for i in range(self.num_ir):
            feat_s, feat_t = self._feat_s[i], self._feat_t[i]
            encoder, decoder = self.encoders[i], self.decoders[i]

            feat_target = encoder(feat_t)
            feat_recon = decoder(feat_target)

            loss_feat_sync += self.mse(feat_s, feat_target)
            loss_feat_recon += self.mse(feat_t, feat_recon)

        loss_feat_sync = self.w_sync * (loss_feat_sync / (self.num_ir if self.num_ir >= 2 else 1))
        loss_feat_recon = self.w_recon * (loss_feat_recon / (self.num_ir if self.num_ir >= 2 else 1))

        # --- Combine all losses ---
        total_loss = loss_hard + loss_soft + loss_feat_sync + loss_feat_recon

        return DistillerOutput(
            logits=logits_s,
            loss=total_loss,
            loss_hard=loss_hard,
            loss_soft=loss_soft,
            loss_feat_sync=loss_feat_sync,
            loss_feat_recon=loss_feat_recon,
        )
