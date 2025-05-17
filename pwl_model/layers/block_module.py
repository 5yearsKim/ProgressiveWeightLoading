import torch
import torch.nn as nn
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention)


class BlockModule(nn.Module):
    def __init__(self, blocks: nn.Module | list[nn.Module]):
        """
        Wrap either a single block or a list of blocks,
        and expose them as one nn.Module in a Sequential‐style.
        """
        super().__init__()
        # If it's already a single Module, just use it directly
        if isinstance(blocks, nn.Module):
            self.seq = blocks
        # If it's a list/tuple of Modules, pack into a Sequential
        elif isinstance(blocks, (list, tuple)):
            self.seq = nn.Sequential(*blocks)
        else:
            raise ValueError(
                f"blocks must be nn.Module or list of them, got {type(blocks)}"
            )

    def forward(self, *args, **kwargs):
        # Just forward everything to the underlying module
        return self.seq(*args, **kwargs)


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
