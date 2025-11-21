from __future__ import annotations

from contextlib import nullcontext
from typing import List, Sequence, Tuple

import numpy as np
import torch


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

    # Channels-last typically improves convolution throughput on CUDA.
    if device.type == "cuda":
        tensor = tensor.contiguous(memory_format=torch.channels_last)
        return tensor.to(device, non_blocking=True)

    return tensor.to(device)


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _tile_positions(length: int, tile_size: int, stride: int) -> List[int]:
    if length <= tile_size:
        return [0]

    positions = list(range(0, (length - tile_size) + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def _blend_window(tile_size: int, overlap: int, device: torch.device) -> torch.Tensor:
    if overlap <= 0:
        return torch.ones((1, tile_size, tile_size), dtype=torch.float32, device=device)

    edge = min(overlap, tile_size // 2)
    ramp = torch.linspace(0.2, 1.0, steps=edge, dtype=torch.float32, device=device)

    horizontal = torch.ones((tile_size,), dtype=torch.float32, device=device)
    horizontal[:edge] = ramp
    horizontal[-edge:] = torch.flip(ramp, dims=[0])

    window = torch.outer(horizontal, horizontal)
    return window.unsqueeze(0)


@torch.inference_mode()
def direct_inference(
    image_rgb: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    input_size: int = 384,
    confidence_threshold: float = 0.5,
    background_class: int = 0,
) -> np.ndarray:
    # Keep the direct path for API compatibility, but run the same full-resolution tiled logic.
    return tiled_inference(
        image_rgb=image_rgb,
        model=model,
        device=device,
        tile_size=input_size,
        batch_size=1,
        overlap=0,
        confidence_threshold=confidence_threshold,
        background_class=background_class,
    )


@torch.inference_mode()
def tiled_inference(
    image_rgb: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    tile_size: int = 384,
    batch_size: int = 8,
    overlap: int = 48,
    confidence_threshold: float = 0.5,
    background_class: int = 0,
) -> np.ndarray:
    if tile_size <= 0:
        raise ValueError("tile_size must be a positive integer.")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if overlap < 0:
        raise ValueError("overlap must be >= 0.")
    if overlap >= tile_size:
        raise ValueError("overlap must be smaller than tile_size.")
    if not (0.0 <= confidence_threshold <= 1.0):
        raise ValueError("confidence_threshold must be in [0.0, 1.0].")

    h, w = image_rgb.shape[:2]
    padded_h = max(h, tile_size)
    padded_w = max(w, tile_size)
    pad_h = padded_h - h
    pad_w = padded_w - w
    stride = tile_size - overlap

    padded = np.pad(image_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    y_positions = _tile_positions(padded_h, tile_size, stride)
    x_positions = _tile_positions(padded_w, tile_size, stride)

    coordinates: List[Tuple[int, int]] = [
        (y, x)
        for y in y_positions
        for x in x_positions
    ]

    blend_window = _blend_window(tile_size=tile_size, overlap=overlap, device=device)
    accum_probs: torch.Tensor | None = None
    accum_weights = torch.zeros((padded_h, padded_w), dtype=torch.float32, device=device)

    for start in range(0, len(coordinates), batch_size):
        chunk = coordinates[start : start + batch_size]

        tiles = [
            padded[y : y + tile_size, x : x + tile_size]
            for (y, x) in chunk
        ]
        inputs = _prepare_batch(tiles, device)
        with _autocast_context(device):
            logits = _extract_logits(model(inputs))

        probs = torch.softmax(logits.float(), dim=1)
        if accum_probs is None:
            num_classes = int(probs.shape[1])
            if background_class < 0 or background_class >= num_classes:
                raise ValueError(
                    f"background_class={background_class} is out of range for {num_classes} classes."
                )
            accum_probs = torch.zeros((num_classes, padded_h, padded_w), dtype=torch.float32, device=device)

        confidence, _ = torch.max(probs, dim=1)
        low_conf_mask = (confidence < confidence_threshold).unsqueeze(1)
        if low_conf_mask.any():
            background_probs = torch.zeros_like(probs)
            background_probs[:, background_class : background_class + 1, :, :] = 1.0
            probs = torch.where(low_conf_mask, background_probs, probs)

        for idx, (y, x) in enumerate(chunk):
            weighted_probs = probs[idx] * blend_window
            accum_probs[:, y : y + tile_size, x : x + tile_size] += weighted_probs
            accum_weights[y : y + tile_size, x : x + tile_size] += blend_window[0]

    if accum_probs is None:
        raise RuntimeError("No tiles were processed during inference.")

    normalized = accum_probs / accum_weights.clamp_min(1e-6).unsqueeze(0)
    pred_mask = torch.argmax(normalized, dim=0).to(torch.uint8).detach().cpu().numpy()
    return pred_mask[:h, :w]
