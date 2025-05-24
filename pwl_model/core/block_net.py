import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

BlockModule = nn.Sequential


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
        **kwargs,
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
