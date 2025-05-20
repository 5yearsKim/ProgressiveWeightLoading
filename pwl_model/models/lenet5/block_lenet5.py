import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutput, ImageClassifierOutputWithNoAttention)

from pwl_model.core.block_net import BlockModule, BlockNetMixin


class BlockLeNet5Config(PretrainedConfig):
    model_type = "block_lenet5"

    def __init__(
        self,
        in_channels: int = 3,
        num_labels: int = 10,
        cnn_channels: list[int] = [6, 16],
        fc_sizes: list[int] = [400, 120, 84],
        **kwargs,
    ):

        assert len(cnn_channels) == 2
        assert len(fc_sizes) == 3
        assert cnn_channels[-1] * 5 * 5 == fc_sizes[0]

        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.num_labels = num_labels
        self.cnn_channels = cnn_channels
        self.fc_sizes = fc_sizes


class BlockLeNet5PreTrainedModel(PreTrainedModel):
    config_class = BlockLeNet5Config
    base_model_prefix = "block_lenet5"


class BlockLeNet5Model(BlockNetMixin, BlockLeNet5PreTrainedModel):
    def __init__(self, config: BlockLeNet5Config) -> None:
        super().__init__(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_blocks(self) -> list[BlockModule]:
        config = self.config
        lenet5_blocks = [
            BlockModule(
                nn.Conv2d(config.in_channels, config.cnn_channels[0], kernel_size=5),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
            ),
            BlockModule(
                nn.Conv2d(
                    config.cnn_channels[0], config.cnn_channels[1], kernel_size=5
                ),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
            ),
            BlockModule(
                nn.Flatten(),
                nn.Linear(config.fc_sizes[0], config.fc_sizes[1]),
                nn.ReLU(),
            ),
            BlockModule(
                nn.Linear(config.fc_sizes[1], config.fc_sizes[2]),
                nn.ReLU(),
            ),
        ]

        return lenet5_blocks


class BlockLeNet5ForImageClassification(BlockLeNet5PreTrainedModel):
    def __init__(self, config: BlockLeNet5Config) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.lenet = BlockLeNet5Model(config)
        # classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            (
                nn.Linear(config.fc_sizes[-1], config.num_labels)
                if config.num_labels > 0
                else nn.Identity()
            ),
        )
        # initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: torch.LongTensor | None = None,
        output_hidden_states: bool = False,
    ):
        outputs = self.lenet(pixel_values, output_hidden_states=output_hidden_states)

        logits = self.classifier(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return ImageClassifierOutputWithNoAttention(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states
        )
