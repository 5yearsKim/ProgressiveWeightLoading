import re
from collections import OrderedDict

import torch

StateDictT = OrderedDict[str, torch.Tensor]


def convert_hf_to_block_vit(hf_sd: StateDictT, convert_classifier=True) -> StateDictT:
    """
    Convert a HuggingFace-style ResNet state_dict (hf_sd)
    into the corresponding BlockResNet state_dict.
    """
    blk_sd: StateDictT = {}

    for k, v in hf_sd.items():
        # 1) embeddings → blocks[0][0]
        if k.startswith("vit.embeddings."):
            suffix = k[len("vit.embeddings.") :]
            new_key = f"vit.blocks.0.0.{suffix}"

        # 2) encoder layers → blocks[i]
        elif k.startswith("vit.encoder.layer."):
            m = re.match(r"vit\.encoder\.layer\.(\d+)\.(.*)", k)
            if not m:
                continue
            layer_idx = int(m.group(1))
            suffix = m.group(2)

            # layer 0 is in blocks[0][1]
            if layer_idx == 0:
                new_key = f"vit.blocks.0.1.module.{suffix}"
            else:
                # layers 1–11 are in blocks[1..11][0]
                new_key = f"vit.blocks.{layer_idx}.0.module.{suffix}"

        # 3) post‐encoder layernorm → blocks[11][1]
        elif k.startswith("vit.layernorm."):
            suffix = k[len("vit.layernorm.") :]
            new_key = f"vit.blocks.11.1.module.{suffix}"

        # 4) classifier → same name (optional)
        elif k.startswith("classifier.") and convert_classifier:
            new_key = k

        else:
            print("dropping key: ", k)
            # drop any other keys (e.g. config, unused)
            continue

        blk_sd[new_key] = v

    return blk_sd


def check_weight_same_vit(
    block_sd: StateDictT,
    hf_sd: StateDictT,
    atol: float = 1e-6,
) -> None:
    """
    Check if the weights in the BlockResNet state_dict (block_sd)
    are the same as those in the HuggingFace state_dict (hf_sd).
    """

    # Convert the HF state dict to the block structure
    converted_hf_sd = convert_hf_to_block_vit(hf_sd)

    # Key sets
    block_keys = set(block_sd.keys())
    hf_keys = set(converted_hf_sd.keys())

    # print('------block model-------')
    # print(block_keys)
    # print('------hf model-------')
    # print(hf_keys)

    for b_key, h_key in zip(sorted(block_keys), sorted(hf_keys)):
        # print(b_key, h_key)
        if b_key != h_key:
            raise ValueError(f"Key mismatch: b( {b_key} )vs h ( {h_key}) ")

    # Compare tensor values for common keys
    mismatches = []
    for key in sorted(block_keys):
        t_block = block_sd[key]
        # print("key:", key)
        t_hf = converted_hf_sd[key]
        if not torch.allclose(t_block, t_hf, atol=atol):
            max_diff = (t_block - t_hf).abs().max().item()
            mismatches.append((key, max_diff))

    if mismatches:
        print(f"❌ Found {len(mismatches)} mismatched tensor(s):")
        for key, diff in mismatches[:10]:
            print(f"  {key}: max abs diff = {diff:.3e}")
        if len(mismatches) > 10:
            print(f"  ...and {len(mismatches) - 10} more mismatches.")
    print("Weight comparison complete.")
