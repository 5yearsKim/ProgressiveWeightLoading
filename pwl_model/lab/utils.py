from pathlib import Path

from transformers import PretrainedConfig, PreTrainedModel


def looks_like_checkpoint_dir(path: str) -> bool:
    p = Path(path)
    if not p.is_dir():
        return False
    # look for any .bin or .safetensors files in the top level
    has_bin = any(p.glob("*.bin"))
    has_safe = any(p.glob("*.safetensors"))
    return has_bin or has_safe


def load_block_model(
    model_from: str | None,
    model_for_image_classification: PreTrainedModel,
    model_config: PretrainedConfig,
) -> PreTrainedModel | None:
    if model_from is None:
        return None

    try:
        return model_for_image_classification.from_pretrained(model_from)
    except:
        try:
            config = model_config.from_pretrained(Path(model_from) / "config.json")
            return model_for_image_classification(config)
        except:
            raise ValueError(f"Invalid model_from value: {model_from}.")
