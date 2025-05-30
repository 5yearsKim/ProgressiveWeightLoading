import torch
import torch.nn as nn
from transformers.modeling_outputs import ImageClassifierOutput
from transformers.models.vit.modeling_vit import (ViTConfig, ViTEmbeddings,
                                                  ViTLayer, ViTModel,
                                                  ViTPooler,
                                                  ViTPreTrainedModel)

from pwl_model.core.block_net import BlockModule, BlockNetMixin


class FirstOutput(nn.Module):
    """
    A wrapper that runs `module(x)` and if the result is a tuple,
    returns its first element; otherwise returns it unchanged.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x, *args, **kwargs):
        out = self.module(x, *args, **kwargs)
        return out[0] if isinstance(out, tuple) else out


class OutputPoller(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x, *args, **kwargs):
        pooled = x[:, 0]
        return pooled


class BlockViTConfig(ViTConfig):
    model_type = "block_vit"

    def __init__(
        self,
        layers_per_block: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layers_per_block = layers_per_block

        assert (
            self.num_hidden_layers % self.layers_per_block == 0
        ), f"num_hidden_layers ({self.num_hidden_layers}) must be divisible by layers_per_block ({self.layers_per_block})"


class BlockViTPreTrainedModel(ViTPreTrainedModel):
    config_class = BlockViTConfig
    base_model_prefix = "block_vit"


class BlockViTModel(BlockNetMixin, BlockViTPreTrainedModel):
    def __init__(
        self,
        config: BlockViTConfig,
        add_pooling_layer: bool = True,
        use_mask_token: bool = False,
    ):
        # load vanilla ViT components
        self.add_pooling_layer = add_pooling_layer
        self.use_mask_token = use_mask_token

        super().__init__(config)
        self.post_init()

    def get_embedder(self) -> nn.Module:
        return ViTEmbeddings(self.config, use_mask_token=self.use_mask_token)

    def get_blocks(self) -> list[BlockModule]:
        # Wrap each transformer layer as a BlockModule
        blocks = []

        num_blocks = self.config.num_hidden_layers // self.config.layers_per_block

        for i in range(num_blocks):
            layers = [
                FirstOutput(
                    ViTLayer(self.config),
                ) for _ in range(self.config.layers_per_block)
            ]
            blocks.append(
                BlockModule(*layers)
            )

        last_block = blocks[-1]

        last_block.append(
            OutputPoller(
                nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps),
            )
        )

        return blocks


class BlockViTForImageClassification(BlockViTPreTrainedModel):
    def __init__(self, config: BlockViTConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.vit = BlockViTModel(config, add_pooling_layer=True)
        # classification head
        self.classifier = (
            nn.Linear(config.hidden_size, config.num_labels)
            if config.num_labels > 0
            else nn.Identity()
        )
        self.post_init()

    @property
    def blocks(self) -> list[BlockModule]:
        return self.vit.blocks

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: torch.LongTensor | None = None,
        **kwargs
    ):
        output = self.vit(pixel_values, **kwargs)
        logits = self.classifier(output.last_hidden_state)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
        )
