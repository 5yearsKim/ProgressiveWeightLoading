import re
from collections import OrderedDict

import torch

StateDictT = OrderedDict[str, torch.Tensor]


def convert_hf_to_block_resnet(hf_sd: StateDictT) -> StateDictT:
    """
    Convert a HuggingFace-style ResNet state_dict (hf_sd)
    into the corresponding BlockResNet state_dict.
    """
    block_sd: StateDictT = OrderedDict()

    for hf_key, tensor in hf_sd.items():
        # 1) embedder → blocks.0.0.embedder
        if hf_key.startswith("resnet.embedder.embedder"):
            rest = hf_key[len("resnet.embedder.embedder") :]
            block_key = f"resnet.blocks.0.0.embedder{rest}"

        # 2) encoder stages → blocks.{stage}.1.{rest}
        elif hf_key.startswith("resnet.encoder.stages."):
            m = re.match(r"^resnet\.encoder\.stages\.(\d)\.(.+)", hf_key)
            if not m:
                continue
            stage, rest = m.group(1), m.group(2)
            block_key = f"resnet.blocks.{stage}.1.{rest}"

        # 3) classifier.1 → classifier.2
        elif hf_key.startswith("classifier.1."):
            block_key = hf_key.replace("classifier.1.", "classifier.2.")

        else:
            # skip anything that doesn't map
            continue

        block_sd[block_key] = tensor

    return block_sd
