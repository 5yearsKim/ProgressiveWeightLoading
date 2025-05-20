import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention
from transformers.models.resnet.modeling_resnet import (ResNetConfig,
                                                        ResNetEmbeddings,
                                                        ResNetStage)

from pwl_model.core.block_net import BlockModule, BlockNetMixin


class BlockResNetConfig(ResNetConfig):
    model_type = "block_resnet"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class BlockResNetPreTrainedModel(PreTrainedModel):
    config_class = BlockResNetConfig
    base_model_prefix = "block_resnet"


class BlockResNetModel(BlockNetMixin, BlockResNetPreTrainedModel):
    def __init__(self, config: BlockResNetConfig) -> None:
        super().__init__(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_blocks(self) -> list[BlockModule]:
        config = self.config
        blocks = [
            BlockModule(
                ResNetEmbeddings(config),
                ResNetStage(
                    config,
                    config.embedding_size,
                    config.hidden_sizes[0],
                    stride=2 if config.downsample_in_first_stage else 1,
                    depth=config.depths[0],
                ),
            ),
        ]
        # ex) config.hidden_sizes = [64, 128, 256, 512]
        # -> in_out_channels = [(64, 128), (128, 256), (256, 512)]
        in_out_channels = zip(config.hidden_sizes, config.hidden_sizes[1:])

        for (in_channels, out_channels), depth in zip(
            in_out_channels, config.depths[1:]
        ):
            blocks.append(
                BlockModule(
                    ResNetStage(
                        config,
                        in_channels,
                        out_channels,
                        depth=depth,
                    ),
                )
            )

        return blocks


class BlockResNetForImageClassification(BlockResNetPreTrainedModel):
    def __init__(self, config: BlockResNetConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.resnet = BlockResNetModel(config)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            (
                nn.Linear(config.hidden_sizes[-1], config.num_labels)
                if config.num_labels > 0
                else nn.Identity()
            ),
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: torch.LongTensor | None = None,
        output_hidden_states: bool = False,
    ) -> torch.FloatTensor:
        outputs = self.resnet(pixel_values, output_hidden_states=output_hidden_states)

        logits = self.classifier(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return ImageClassifierOutputWithNoAttention(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states
        )
