import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ResNetConfig, ResNetForImageClassification


class ResNetFeatureDistiller(nn.Module):
    def __init__(
        self,
        student: ResNetForImageClassification,
        teacher: ResNetForImageClassification,
        temp: float = 4.0,
        alpha: float = 0.5,
        beta: float = 0.5,
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

        # Storage for hooked features
        self._feat_t: dict[str, torch.Tensor] = {}
        self._feat_s: dict[str, torch.Tensor] = {}

        # Choose which layers to distill; here we pick two ResNet “stages”
        teacher_layers = {
            "stage0": teacher.resnet.encoder.stages[0],
            "stage1": teacher.resnet.encoder.stages[1],
            "stage2": teacher.resnet.encoder.stages[2],
            "stage3": teacher.resnet.encoder.stages[3],
        }
        student_layers = {
            "stage0": student.resnet.encoder.stages[0],
            "stage1": student.resnet.encoder.stages[1],
            "stage2": student.resnet.encoder.stages[2],
            "stage3": student.resnet.encoder.stages[3],
        }

        # Register hooks
        for name, mod in teacher_layers.items():
            mod.register_forward_hook(self._get_hook(self._feat_t, name))
        for name, mod in student_layers.items():
            mod.register_forward_hook(self._get_hook(self._feat_s, name))

        # Loss fns
        self.temp = temp
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.alpha = alpha
        self.beta = beta

    def _get_hook(self, storage: dict, name: str):
        def hook(module, inp, out):
            # out: Tensor of shape [B, C, H, W]
            storage[name] = out

        return hook

    def forward(self, pixel_values, labels: torch.LongTensor):
        # --- Student forward ---
        out_s = self.student(pixel_values)
        logits_s = out_s.logits  # [B, num_classes]

        # --- Teacher forward ---
        with torch.no_grad():
            out_t = self.teacher(pixel_values)
        logits_t = out_t.logits  # [B, num_classes]

        # 1) Hard-label CE
        loss_hard = self.ce(logits_s, labels)

        # 2) Soft-label KL
        T = self.temp
        p_s = F.log_softmax(logits_s / T, dim=-1)
        p_t = F.softmax(logits_t / T, dim=-1)
        loss_soft = F.kl_div(p_s, p_t, reduction="batchmean") * (T * T)

        # 3) Feature-MSE between each tapped layer
        loss_feat = 0.0
        for name in self._feat_s:
            f_s = self._feat_s[name]
            f_t = self._feat_t[name]
            # If shapes differ (e.g. student narrower), you may need a 1×1 conv to align dims
            if f_s.shape != f_t.shape:
                # simple fallback: interpolate spatially and pad channels
                f_t_resized = F.interpolate(
                    f_t, size=f_s.shape[-2:], mode="bilinear", align_corners=False
                )
                if f_t_resized.shape[1] != f_s.shape[1]:
                    # zero-pad or project; here: truncate or pad zeros
                    minC = min(f_s.shape[1], f_t_resized.shape[1])
                    f_t_resized = torch.cat(
                        [f_t_resized[:, :minC], torch.zeros_like(f_s[:, minC:])], dim=1
                    )
                f_t = f_t_resized
            loss_feat += self.mse(f_s, f_t)
        loss_feat = loss_feat / len(self._feat_s)

        # --- Combine all losses ---
        loss = (
            self.alpha * loss_hard
            + (1 - self.alpha - self.beta) * loss_soft
            + self.beta * loss_feat
        )
        return {"loss": loss, "logits": logits_s}


from transformers import ResNetConfig, ResNetForImageClassification


def create_resnet_blocks(config: ResNetConfig) -> tuple[list[nn.Module], int]:
    resnet_blocks = [
        nn.Conv2d(config.in_channels, config.cnn_channels[0], kernel_size=7, stride=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Conv2d(
            config.cnn_channels[0], config.cnn_channels[1], kernel_size=3, stride=1
        ),
        nn.ReLU(),
        nn.Conv2d(
            config.cnn_channels[1], config.cnn_channels[2], kernel_size=3, stride=1
        ),
        nn.ReLU(),
    ]
    return resnet_blocks, config.fc_sizes[-1]
