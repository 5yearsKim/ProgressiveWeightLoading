import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention
from transformers.models.resnet.modeling_resnet import (ResNetConfig,
                                                        ResNetConvLayer,
                                                        ResNetEmbeddings,
                                                        ResNetStage)

from pwl_model.core.block_net import BlockModule, BlockNetMixin


class BlockResNetConfig(ResNetConfig):
    model_type = "block_resnet"

    def __init__(
        self,
        embedder_kernel_size=7,
        embedder_kernel_stride=2,
        embedder_use_pooler=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embedder_kernel_size = embedder_kernel_size
        self.embedder_kernel_stride = embedder_kernel_stride
        self.embedder_use_pooler = embedder_use_pooler


class BlockResNetPreTrainedModel(PreTrainedModel):
    config_class = BlockResNetConfig
    base_model_prefix = "block_resnet"


class BlockResNetEmbeddings(ResNetEmbeddings):
    def __init__(self, config: BlockResNetConfig):
        super().__init__(config)
        self.embedder = ResNetConvLayer(
            config.num_channels,
            config.embedding_size,
            kernel_size=config.embedder_kernel_size,
            stride=config.embedder_kernel_stride,
            activation=config.hidden_act,
        )
        self.pooler = (
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            if config.embedder_use_pooler
            else nn.Identity()
        )


class BlockResNetModel(BlockNetMixin, BlockResNetPreTrainedModel):
    def __init__(self, config: BlockResNetConfig) -> None:
        super().__init__(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_embedder(self) -> nn.Module:
        return BlockResNetEmbeddings(self.config)

    def get_blocks(self) -> list[BlockModule]:
        config = self.config
        blocks = [
            BlockModule(
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

    @property
    def blocks(self) -> list[BlockModule]:
        return self.resnet.blocks

    @property
    def embedder(self) -> nn.Module:
        return self.resnet.embedder

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
