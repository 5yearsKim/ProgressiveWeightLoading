from pathlib import Path


def looks_like_checkpoint_dir(path: str) -> bool:
    p = Path(path)
    if not p.is_dir():
        return False
    # look for any .bin or .safetensors files in the top level
    has_bin = any(p.glob("*.bin"))
    has_safe = any(p.glob("*.safetensors"))
    return has_bin or has_safe
