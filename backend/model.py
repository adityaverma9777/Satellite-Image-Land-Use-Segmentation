from __future__ import annotations

from pathlib import Path
from typing import Dict

import segmentation_models_pytorch as smp
import torch

NUM_CLASSES = 6
ENCODER_NAME = "resnet50"


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    clean_state: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            clean_state[key[len("module.") :]] = value
        else:
            clean_state[key] = value
    return clean_state


def _extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            nested = checkpoint.get(key)
            if isinstance(nested, dict):
                return _strip_module_prefix(nested)

        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return _strip_module_prefix(checkpoint)

    raise ValueError("Could not find a valid state_dict in checkpoint.")


def build_model() -> torch.nn.Module:
    return smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
    )


def load_model(weights_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(weights_path, map_location=device)

    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint.to(device)
        model.eval()
        return model

    model = build_model().to(device)
    state_dict = _extract_state_dict(checkpoint)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"[model] Warning: missing keys while loading checkpoint: {missing_keys[:10]}")
    if unexpected_keys:
        print(f"[model] Warning: unexpected keys while loading checkpoint: {unexpected_keys[:10]}")

    model.eval()
    return model
