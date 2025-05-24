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


class BlockViTConfig(ViTConfig):
    model_type = "block_vit"


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

    def get_blocks(self) -> list[BlockModule]:
        # Wrap each transformer layer as a BlockModule
        blocks = [
            BlockModule(
                ViTEmbeddings(self.config, use_mask_token=self.use_mask_token),
                FirstOutput(
                    ViTLayer(self.config),
                ),
            )
        ]
        for i in range(self.config.num_hidden_layers - 1):
            blocks.append(
                BlockModule(
                    FirstOutput(
                        ViTLayer(self.config),
                    ),
                )
            )

        last_block = blocks[-1]

        last_block.append(
            nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps),
        )
        # last_block.append(
        #     ViTPooler(self.config)
        # )

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
        outputs = self.vit(pixel_values, **kwargs)
        pooled = (
            outputs.pooler_output
            if hasattr(outputs, "pooler_output")
            else outputs.last_hidden_state[:, 0]
        )
        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
