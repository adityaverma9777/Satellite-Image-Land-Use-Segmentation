from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


def _extract_logits(model_output: object) -> torch.Tensor:
    if isinstance(model_output, torch.Tensor):
        return model_output
    if isinstance(model_output, dict):
        if "out" in model_output:
            return model_output["out"]
        first_key = next(iter(model_output.keys()))
        return model_output[first_key]
    if isinstance(model_output, (tuple, list)):
        return model_output[0]
    raise TypeError("Unsupported model output type for segmentation logits.")


def _prepare_batch(batch_tiles: Sequence[np.ndarray], device: torch.device) -> torch.Tensor:
    array = np.stack(batch_tiles).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2)
    return tensor.to(device)


@torch.inference_mode()
def direct_inference(
    image_rgb: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    input_size: int = 256,
) -> np.ndarray:
    original_h, original_w = image_rgb.shape[:2]
    resized = np.array(Image.fromarray(image_rgb).resize((input_size, input_size), Image.Resampling.BILINEAR))

    inputs = _prepare_batch([resized], device)
    logits = _extract_logits(model(inputs))
    pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)

    upscaled = np.array(
        Image.fromarray(pred, mode="L").resize((original_w, original_h), Image.Resampling.NEAREST)
    )
    return upscaled


@torch.inference_mode()
def tiled_inference(
    image_rgb: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    tile_size: int = 256,
    batch_size: int = 8,
) -> np.ndarray:
    if tile_size <= 0:
        raise ValueError("tile_size must be a positive integer.")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    h, w = image_rgb.shape[:2]
    pad_h = (tile_size - (h % tile_size)) % tile_size
    pad_w = (tile_size - (w % tile_size)) % tile_size

    padded = np.pad(image_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    padded_h, padded_w = padded.shape[:2]

    coordinates: List[Tuple[int, int]] = []
    for y in range(0, padded_h, tile_size):
        for x in range(0, padded_w, tile_size):
            coordinates.append((y, x))

    pred_mask = np.zeros((padded_h, padded_w), dtype=np.uint8)

    for start in range(0, len(coordinates), batch_size):
        chunk = coordinates[start : start + batch_size]

        tiles = [
            padded[y : y + tile_size, x : x + tile_size]
            for (y, x) in chunk
        ]
        inputs = _prepare_batch(tiles, device)
        logits = _extract_logits(model(inputs))
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.uint8)

        for idx, (y, x) in enumerate(chunk):
            pred_mask[y : y + tile_size, x : x + tile_size] = preds[idx]

    return pred_mask[:h, :w]
