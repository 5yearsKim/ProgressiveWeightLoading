import torch
import torch.nn as nn

from .block_net import BlockModule, BlockNetMixin
from .feature_converter import FeatureConverter


class SwapNet(nn.Module):
    def __init__(
        self,
        teacher: BlockNetMixin,
        student: BlockNetMixin,
        input_shape: tuple = (3, 32, 32),
        channel_last: bool = False,
    ):

        super().__init__()

        self.teacher = teacher
        self.student = student

        self.channel_last = channel_last

        assert len(self.student.blocks) == len(
            self.teacher.blocks
        ), "Teacher and student must have the same number of blocks"

        for p in self.teacher.parameters():
            p.requires_grad = False

        self.num_blocks = len(self.student.blocks)
        self.num_feat = self.num_blocks - 1

        # Storage for hooked features
        self._feat_t: list[torch.Tensor] = []
        self._feat_s: list[torch.Tensor] = []

        with torch.no_grad():
            x_s = torch.zeros((1, *input_shape))
            x_t = torch.zeros((1, *input_shape))

            x_s = self.student.embedder(x_s)
            x_t = self.teacher.embedder(x_t)

            for i, (s_block, t_block) in enumerate(
                zip(self.student.blocks, self.teacher.blocks)
            ):
                if i == self.num_blocks - 1:
                    break # Last block is not tracked
                x_s = s_block(x_s)
                x_t = t_block(x_t)
                self._feat_s.append(x_s)
                self._feat_t.append(x_t)

        assert (
            len(self._feat_s) == len(self._feat_t) == self.num_feat
        ), "Number of features in student and teacher must be the same"

        self.encoders = nn.ModuleList([])
        self.decoders = nn.ModuleList([])

        for i, (feat_s, feat_t) in enumerate(zip(self._feat_s, self._feat_t)):

            if self.channel_last:
                chan_t, chan_s = feat_t.shape[-1], feat_s.shape[-1]
            else:
                chan_t, chan_s = feat_t.shape[1], feat_s.shape[1]

            print(
                f"Block {i}: teacher {chan_t} -> student {chan_s} ({feat_t.shape} -> {feat_s.shape})"
            )

            if chan_t == chan_s:
                self.encoders.append(nn.Identity())
                self.decoders.append(nn.Identity())
            elif chan_t > chan_s:
                self.encoders.append(
                    FeatureConverter(chan_t, chan_s, channel_last=channel_last)
                )
                self.decoders.append(
                    FeatureConverter(chan_s, chan_t, channel_last=channel_last)
                )
            else:
                raise ValueError(f"chan_s {chan_s} should be less than chan_t {chan_t}")

    def forward(self, x, from_teachers: list[bool]) -> torch.Tensor:
        assert (
            len(from_teachers) == self.num_blocks
        ), "from_teacher must be a list of booleans with the same length as the number of blocks"

        embedder = (
            self.teacher.embedder if from_teachers[0] else self.student.embedder
        )
        x = embedder(x)

        for i in range(self.num_blocks):
            is_teacher = from_teachers[i]
            block = self.teacher.blocks[i] if is_teacher else self.student.blocks[i]
            # converter is None if last block or if next block is from teacher is same as current block
            converter: None | FeatureConverter = (
                None
                if i == self.num_blocks - 1 or from_teachers[i] == from_teachers[i + 1]
                else (self.encoders[i] if is_teacher else self.decoders[i])
            )
            x = block(x)
            if converter is not None:
                x = converter(x)

        classifier = (
            self.teacher.classifier if from_teachers[-1] else self.student.classifier
        )
        logits = classifier(x)

        return logits

    def cross_forward(
        self, feature: torch.Tensor, idx: int, from_teacher: bool
    ) -> torch.Tensor:
        """
        Forward pass with cross feature.
        """
        assert (
            idx < self.num_feat
        ), f"idx {idx} should be less than num_blocks {self.num_feat} - 1"

        converter = self.encoders[idx] if from_teacher else self.decoders[idx]
        blocks = (
            self.student.blocks[idx + 1 :]
            if from_teacher
            else self.teacher.blocks[idx + 1 :]
        )
        classifier = (
            self.student.classifier if from_teacher else self.teacher.classifier
        )

        x = converter(feature)
        for block in blocks:
            x = block(x)

        logits = classifier(x)

        return logits

    def infer_teacher(
        self, x, return_features: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        return self._infer_block(
            x,
            self.teacher.embedder,
            self.teacher.blocks,
            self.teacher.classifier,
            return_features=return_features,
        )

    def infer_student(
        self, x, return_features: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        return self._infer_block(
            x,
            self.student.embedder,
            self.student.blocks,
            self.student.classifier,
            return_features=return_features,
        )

    def _infer_block(
        self,
        x,
        embedder: nn.Module,
        blocks: list[BlockModule],
        classifier: nn.Module,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if return_features:
            features = []

        x = embedder(x)

        for i, block in enumerate(blocks):
            x = block(x)

            if return_features and i < self.num_feat:
                features.append(x)

        logits = classifier(x)

        if return_features:
            return logits, features
        else:
            return logits
