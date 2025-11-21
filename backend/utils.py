from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import rasterio
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.warp import transform_bounds

CLASS_LABELS = {
    0: "Class 0",
    1: "Class 1",
    2: "Class 2",
    3: "Class 3",
}

# RGB colors for each class index 0..3
CLASS_COLORS = np.array(
    [
        [255, 0, 0],      # Class 0 -> Red
        [0, 255, 0],      # Class 1 -> Green
        [0, 0, 255],      # Class 2 -> Blue
        [255, 255, 0],    # Class 3 -> Yellow
    ],
    dtype=np.uint8,
)


def _normalize_to_uint8(band: np.ndarray) -> np.ndarray:
    if band.dtype == np.uint8:
        return band

    finite = np.isfinite(band)
    if not finite.any():
        return np.zeros_like(band, dtype=np.uint8)

    values = band[finite].astype(np.float32)
    low = float(np.percentile(values, 2))
    high = float(np.percentile(values, 98))

    if np.isclose(high, low):
        low = float(values.min())
        high = float(values.max())

    if np.isclose(high, low):
        return np.zeros_like(band, dtype=np.uint8)

    scaled = (band.astype(np.float32) - low) / (high - low)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def load_image_and_geo_metadata(file_bytes: bytes, filename: str) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    suffix = Path(filename or "uploaded").suffix.lower()

    if suffix in {".tif", ".tiff", ".geotiff"}:
        with MemoryFile(file_bytes) as memfile:
            with memfile.open() as src:
                data = src.read()

                if data.ndim != 3:
                    raise ValueError("Unsupported GeoTIFF shape for RGB inference.")

                if data.shape[0] == 1:
                    rgb = np.repeat(data, 3, axis=0)
                elif data.shape[0] >= 3:
                    rgb = data[:3]
                else:
                    raise ValueError("GeoTIFF has no usable bands.")

                rgb = np.moveaxis(rgb, 0, -1)
                rgb_uint8 = np.zeros_like(rgb, dtype=np.uint8)
                for c in range(3):
                    rgb_uint8[..., c] = _normalize_to_uint8(rgb[..., c])

                geo_meta = {
                    "profile": src.profile.copy(),
                    "crs": str(src.crs) if src.crs else None,
                }

                if src.crs:
                    bounds = src.bounds
                    if str(src.crs).upper() == "EPSG:4326":
                        west, south, east, north = bounds
                    else:
                        west, south, east, north = transform_bounds(
                            src.crs,
                            "EPSG:4326",
                            bounds.left,
                            bounds.bottom,
                            bounds.right,
                            bounds.top,
                            densify_pts=21,
                        )
                    geo_meta["bounds_epsg4326"] = [[south, west], [north, east]]

                return rgb_uint8, geo_meta

    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return np.array(image), None


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("Mask must be 2D [H, W].")
    if mask.min() < 0 or mask.max() >= len(CLASS_COLORS):
        raise ValueError("Mask values are outside expected class index range 0..3.")

    return CLASS_COLORS[mask]


def blend_images(original_rgb: np.ndarray, colored_mask_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    base = original_rgb.astype(np.float32)
    overlay = colored_mask_rgb.astype(np.float32)
    blended = ((1.0 - alpha) * base) + (alpha * overlay)
    return blended.clip(0, 255).astype(np.uint8)


def save_png(path: Path, array: np.ndarray, mode: Optional[str] = None) -> None:
    image = Image.fromarray(array, mode=mode) if mode else Image.fromarray(array)
    image.save(path, format="PNG")


def png_base64(array: np.ndarray, mode: Optional[str] = None) -> str:
    image = Image.fromarray(array, mode=mode) if mode else Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def save_mask_as_geotiff(mask: np.ndarray, output_path: Path, geo_meta: Optional[Dict[str, Any]]) -> bool:
    if geo_meta is None:
        return False

    profile = dict(geo_meta.get("profile", {}))
    if not profile:
        return False

    profile.update(
        driver="GTiff",
        dtype=rasterio.uint8,
        count=1,
        height=mask.shape[0],
        width=mask.shape[1],
        compress="lzw",
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)

    return True


def load_label_mask(file_bytes: bytes) -> np.ndarray:
    mask = Image.open(io.BytesIO(file_bytes))
    array = np.array(mask)

    if array.ndim == 3:
        array = array[..., 0]

    return array.astype(np.uint8)


def compute_miou(pred_mask: np.ndarray, true_mask: np.ndarray, num_classes: int = 4) -> Dict[str, Any]:
    if pred_mask.shape != true_mask.shape:
        raise ValueError("Prediction and ground-truth masks must have the same shape.")

    class_ious: Dict[int, Optional[float]] = {}
    valid_ious = []

    for cls in range(num_classes):
        pred_cls = pred_mask == cls
        true_cls = true_mask == cls

        intersection = int(np.logical_and(pred_cls, true_cls).sum())
        union = int(np.logical_or(pred_cls, true_cls).sum())

        if union == 0:
            class_ious[cls] = None
            continue

        iou = float(intersection / union)
        class_ious[cls] = iou
        valid_ious.append(iou)

    miou = float(np.mean(valid_ious)) if valid_ious else 0.0
    return {
        "miou": miou,
        "class_ious": class_ious,
    }
