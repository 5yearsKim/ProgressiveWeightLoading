import re
from collections import OrderedDict

import torch

StateDictT = OrderedDict[str, torch.Tensor]


def convert_hf_to_block_vit(
    hf_sd: StateDictT,
    convert_classifier: bool = True
) -> StateDictT:
    """
    Convert a HuggingFace ViTForImageClassification state_dict (hf_sd)
    into the corresponding BlockViTForImageClassification layout.
    """
    blk_sd: StateDictT = OrderedDict()

    for k, v in hf_sd.items():
        # 1) Embeddings → vit.embedder
        if k.startswith("vit.embeddings."):
            suffix = k[len("vit.embeddings."):]
            new_key = f"vit.embedder.{suffix}"

        # 2) Patch-proj & norm under embeddings
        elif k.startswith("vit.embeddings.patch_embeddings.projection."):
            suffix = k[len("vit.embeddings.patch_embeddings.projection."):]
            new_key = f"vit.embedder.patch_embeddings.projection.{suffix}"
        elif k.startswith("vit.embeddings.patch_embeddings.norm."):
            suffix = k[len("vit.embeddings.patch_embeddings.norm."):]
            new_key = f"vit.embedder.patch_embeddings.norm.{suffix}"

        # 3) Encoder layers → blocks[i//2].[i%2].module
        elif k.startswith("vit.encoder.layer."):
            m = re.match(r"vit\.encoder\.layer\.(\d+)\.(.*)", k)
            if not m:
                continue
            layer_idx = int(m.group(1))
            suffix = m.group(2)
            blk_idx = layer_idx // 2
            in_blk  = layer_idx % 2
            new_key = f"vit.blocks.{blk_idx}.{in_blk}.module.{suffix}"

        # 4) Post-encoder LayerNorm → blocks[5][2].module
        elif k.startswith("vit.layernorm."):
            suffix  = k[len("vit.layernorm."):]
            new_key = f"vit.blocks.5.2.module.{suffix}"

        # 5) Classifier → passthrough
        elif convert_classifier and k.startswith("classifier."):
            new_key = k

        else:
            # drop everything else
            continue

        blk_sd[new_key] = v

    return blk_sd


def check_weight_same_vit(
    block_sd: StateDictT,
    hf_sd: StateDictT,
    atol: float = 1e-6,
) -> None:
    """
    Verify that block_sd matches hf_sd under the block layout.
    Raises if any keys mismatch or any tensor differs by > atol.
    """
    # 1) convert HF → block layout
    converted = convert_hf_to_block_vit(hf_sd, convert_classifier=True)

    # 2) compare key sets
    b_keys = set(block_sd.keys())
    h_keys = set(converted.keys())
    if b_keys != h_keys:
        missing    = b_keys - h_keys
        unexpected = h_keys - b_keys
        raise ValueError(
            f"Key mismatch:\n"
            f"  Missing in converted:   {missing}\n"
            f"  Unexpected in converted: {unexpected}"
        )

    # 3) compare values
    diffs = []
    for k in sorted(b_keys):
        t1 = block_sd[k]
        t2 = converted[k]
        if not torch.allclose(t1, t2, atol=atol):
            diffs.append((k, (t1 - t2).abs().max().item()))

    if diffs:
        msg = "\n".join(f"{k}: max diff = {d:.3e}" for k, d in diffs[:10])
        more = f"... and {len(diffs)-10} more" if len(diffs) > 10 else ""
        raise RuntimeError(f"{len(diffs)} tensors differ:\n{msg}\n{more}")

    print("✅ All weights match within atol!")
