import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureConverter(nn.Module):
    def __init__(self, c_in: int, c_out: int, channel_last: bool = True):
        super().__init__()
        self.c_in, self.c_out = c_in, c_out
        self.linear = nn.Linear(c_in, c_out)
        self.channel_last = channel_last

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, D1, D2, ..., Dk)
        returns y: (B, C_out, D1, D2, ..., Dk)
        """
        if self.channel_last:
            # Move the channel dimension to the last position
            # e.g. for 4D: (B, D1, D2, C_in) → (B, D1, D2, C_out)
            y = self.linear(x)
            return y
        # 1) move C_in to the last dim
        #    e.g. for 4D: (B, H, W, C_in)
        dims = list(range(x.dim()))

        # new order: [0] + [2,3,…,k+1] + [1]
        permute_order = [0] + dims[2:] + [1]
        x_perm = x.permute(*permute_order)  # (..., C_in)

        # 2) apply linear → shape (..., C_out)
        y_perm = self.linear(x_perm)  # x @ W^T + b

        # 3) permute back: move C_out to dim=1
        #    invert the original permutation
        #    original dims = len(dims)+1 after linear, call that D = x.dim()
        D = x.dim()
        # build inverse permutation
        inv = [None] * D
        for i, p in enumerate(permute_order):
            inv[p] = i
        y = y_perm.permute(*inv)
        return y
