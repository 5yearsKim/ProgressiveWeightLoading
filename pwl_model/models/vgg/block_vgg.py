import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention

from pwl_model.core.block_net import BlockModule, BlockNetMixin


class BlockVGGConfig(PretrainedConfig):
    model_type = "block_vgg"

    def __init__(
        self,
        in_channels: int = 3,
        hidden_sizes: list[int] = [64, 128, 256, 512, 512],
        depths: list[int] = [2, 2, 3, 3, 3],
        pool_kernel: int = 2,
        pool_stride: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.hidden_sizes = hidden_sizes
        self.depths = depths
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride


class BlockVGGPreTrainedModel(PreTrainedModel):
    config_class = BlockVGGConfig
    base_model_prefix = "block_vgg"


class BlockVGGModel(BlockNetMixin, BlockVGGPreTrainedModel):
    def __init__(self, config: BlockVGGConfig) -> None:
        super().__init__(config)
        self.post_init()

    def get_embedder(self) -> nn.Module:
        return nn.Identity()  # VGG does not have a separate embedder

    def get_blocks(self) -> list[BlockModule]:
        cfg = self.config
        blocks: list[BlockModule] = []
        in_channels = cfg.in_channels

        # Build each VGG stage as a BlockModule
        for out_channels, num_convs in zip(cfg.hidden_sizes, cfg.depths):
            convs = []
            # sequence of conv -> relu
            for _ in range(num_convs):
                convs.append(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
                )
                convs.append(nn.BatchNorm2d(out_channels))        # ← add this
                convs.append(nn.ReLU(inplace=True))
                in_channels = out_channels
            # pooling layer
            convs.append(
                nn.MaxPool2d(kernel_size=cfg.pool_kernel, stride=cfg.pool_stride)
            )

            block_seq = nn.Sequential(*convs)
            blocks.append(BlockModule(block_seq))

        return blocks


class BlockVGGForImageClassification(BlockVGGPreTrainedModel):
    def __init__(self, config: BlockVGGConfig) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.vgg = BlockVGGModel(config)

        # classification head: global avg pool + linear
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            (
                nn.Linear(config.hidden_sizes[-1], config.num_labels)
                if config.num_labels > 0
                else nn.Identity()
            ),
        )

        self.post_init()

    @property
    def blocks(self) -> list[BlockModule]:
        return self.vgg.blocks

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: torch.LongTensor | None = None,
        output_hidden_states: bool = False,
    ) -> ImageClassifierOutputWithNoAttention:
        outputs = self.vgg(pixel_values, output_hidden_states=output_hidden_states)
        logits = self.classifier(outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
