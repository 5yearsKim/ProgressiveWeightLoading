import re
from collections import OrderedDict

import torch

StateDictT = OrderedDict[str, torch.Tensor]


def convert_hf_to_block_resnet(hf_sd: StateDictT, convert_classfier=True) -> StateDictT:
    """
    Convert a HuggingFace-style ResNet state_dict (hf_sd)
    into the corresponding BlockResNet state_dict.
    """
    blk_sd: StateDictT = {}

    # 1) Stem (embedder)
    blk_sd["resnet.blocks.0.0.embedder.convolution.weight"] = hf_sd[
        "resnet.embedder.embedder.convolution.weight"
    ]
    for p in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
        blk_sd[f"resnet.blocks.0.0.embedder.normalization.{p}"] = hf_sd[
            f"resnet.embedder.embedder.normalization.{p}"
        ]

    # 2) Stage 0 (first ResNetStage → blocks[0][1])
    for i in (0, 1):
        for j in (0, 1):
            ms_pref = f"resnet.encoder.stages.0.layers.{i}.layer.{j}"
            blk_pref = f"resnet.blocks.0.1.layers.{i}.layer.{j}"
            # conv weight
            blk_sd[f"{blk_pref}.convolution.weight"] = hf_sd[
                f"{ms_pref}.convolution.weight"
            ]
            # batch‐norm params
            for p in (
                "weight",
                "bias",
                "running_mean",
                "running_var",
                "num_batches_tracked",
            ):
                blk_sd[f"{blk_pref}.normalization.{p}"] = hf_sd[
                    f"{ms_pref}.normalization.{p}"
                ]

    # 3) Stages 1–3 (all go to blocks[n][0])
    for n in (1, 2, 3):
        for i in (0, 1):
            ms_layer = f"resnet.encoder.stages.{n}.layers.{i}"
            blk_layer = f"resnet.blocks.{n}.0.layers.{i}"

            # shortcut only for the first block in each stage
            if i == 0:
                blk_sd[f"{blk_layer}.shortcut.convolution.weight"] = hf_sd[
                    f"{ms_layer}.shortcut.convolution.weight"
                ]
                for p in (
                    "weight",
                    "bias",
                    "running_mean",
                    "running_var",
                    "num_batches_tracked",
                ):
                    blk_sd[f"{blk_layer}.shortcut.normalization.{p}"] = hf_sd[
                        f"{ms_layer}.shortcut.normalization.{p}"
                    ]

            # two conv+BN sublayers per basic layer
            for j in (0, 1):
                ms_pref = f"{ms_layer}.layer.{j}"
                blk_pref = f"{blk_layer}.layer.{j}"
                blk_sd[f"{blk_pref}.convolution.weight"] = hf_sd[
                    f"{ms_pref}.convolution.weight"
                ]
                for p in (
                    "weight",
                    "bias",
                    "running_mean",
                    "running_var",
                    "num_batches_tracked",
                ):
                    blk_sd[f"{blk_pref}.normalization.{p}"] = hf_sd[
                        f"{ms_pref}.normalization.{p}"
                    ]

    if convert_classfier:
        # 4) Classifier head
        # HF’s classifier: (0)=Flatten, (1)=Linear
        # Block’s classifier: (0)=AdaptiveAvgPool, (1)=Flatten, (2)=Linear
        blk_sd["classifier.2.weight"] = hf_sd["classifier.1.weight"]
        blk_sd["classifier.2.bias"] = hf_sd["classifier.1.bias"]

    return blk_sd


def check_weight_same_resnet(
    block_sd: StateDictT,
    hf_sd: StateDictT,
    atol: float = 1e-6,
) -> None:
    """
    Check if the weights in the BlockResNet state_dict (block_sd)
    are the same as those in the HuggingFace state_dict (hf_sd).
    """

    # Convert the HF state dict to the block structure
    converted_hf_sd = convert_hf_to_block_resnet(hf_sd)

    # Key sets
    block_keys = set(block_sd.keys())
    hf_keys = set(converted_hf_sd.keys())

    # print('------block model-------')
    # print(block_keys)
    # print('------hf model-------')
    # print(hf_keys)

    for b_key, h_key in zip(sorted(block_keys), sorted(hf_keys)):
        print(b_key, h_key)
        if b_key != h_key:
            raise ValueError(f"Key mismatch: b( {b_key} )vs h ( {h_key}) ")

    # Compare tensor values for common keys
    mismatches = []
    for key in sorted(block_keys):
        t_block = block_sd[key]
        print("key:", key)
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
