import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutput, ImageClassifierOutputWithNoAttention)

BlockModule = nn.Sequential


class BlockNetConfig(PretrainedConfig):
    model_type = "block_net"

    def __init__(
        self,
        in_shape: tuple,
        out_shape: tuple,
        num_labels: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.in_shape = in_shape
        self.out_shape = out_shape
        assert all(
            isinstance(i, int) for i in in_shape
        ), "in_shape must be a tuple of ints"
        assert all(
            isinstance(i, int) for i in out_shape
        ), "out_shape must be a tuple of ints"


class BlockNetMixin(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.blocks = nn.ModuleList(self.get_blocks())

    @abstractmethod
    def get_blocks(self) -> list[BlockModule]:
        raise NotImplementedError("get_blocks() must be implemented in subclasses")

    def forward(
        self,
        x: torch.FloatTensor,
        output_hidden_states: bool | None = None,
    ) -> BaseModelOutput:
        """
        Forward pass of the model.
        """
        hidden_states = () if output_hidden_states else None

        for block in self.blocks:
            x = block(x)
            if output_hidden_states:
                hidden_states += (x,)

        return BaseModelOutput(last_hidden_state=x, hidden_states=hidden_states)


class BlockNetPretrainedModel(PreTrainedModel):
    config_class = BlockNetConfig
    base_model_prefix = "block_model"


class BlockNet(ABC, PreTrainedModel):
    config_class = BlockNetConfig
    base_model_prefix = "block_model"

    def __init__(self, config: BlockNetConfig) -> None:
        super().__init__(config)
        self.config = config

        self.blocks = nn.ModuleList(self.get_blocks())

        # Initialize weights and apply final processing
        self.post_init()

    @abstractmethod
    def get_blocks(self) -> list[BlockModule]:
        raise NotImplementedError("get_blocks() must be implemented in subclasses")


class BlockNetForImageClassification(BlockNetPretrainedModel):
    def __init__(
        self,
        config: BlockNetConfig,
    ) -> None:
        super().__init__()

        self.config = config

        assert config.num_labels is not None, "num_labels must be set in the config"

        self.encoder = BlockNet(config)

        last_out_dim = math.prod(x for x in config.out_shape if x > 0)
        num_labels = config.num_labels

        self.classifier = nn.Sequential(
            nn.Flatten(),
            (nn.Linear(last_out_dim, num_labels) if num_labels > 0 else nn.Identity()),
        )

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: torch.LongTensor | None = None,
        output_hidden_states: bool | None = None,
    ) -> ImageClassifierOutputWithNoAttention:
        """
        Forward pass of the model.
        """
        x = pixel_values

        outputs = self.encoder(x, output_hidden_states=output_hidden_states)

        logits = self.classifier(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return ImageClassifierOutputWithNoAttention(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states
        )
