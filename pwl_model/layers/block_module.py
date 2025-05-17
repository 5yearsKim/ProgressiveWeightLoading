import torch
import torch.nn as nn
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention)

BlockModule = nn.Sequential


class BlockModelForImageClassification(nn.Module):
    def __init__(
        self, blocks: list[BlockModule], last_out_dim: int, num_labels: int
    ) -> None:
        super().__init__()

        self.num_labels = num_labels
        self.classifier = nn.Sequential(
            nn.Flatten(),
            (nn.Linear(last_out_dim, num_labels) if num_labels > 0 else nn.Identity()),
        )
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self, pixel_values: torch.FloatTensor, labels: torch.LongTensor | None = None
    ) -> BaseModelOutputWithPoolingAndNoAttention:
        """
        Forward pass of the model.
        """
        x = pixel_values
        for block in self.blocks:
            x = block(x)

        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return ImageClassifierOutputWithNoAttention(loss=loss, logits=logits)
