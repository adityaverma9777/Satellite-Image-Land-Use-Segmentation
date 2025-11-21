from __future__ import annotations

import io
import math
import sqlite3
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
MAP_DATA_DIR = BASE_DIR / "map-data"

LAYER_FILES = {
    "satellite": "satellite.mbtiles",
    "terrain": "terrain-rgb.mbtiles",
}


def available_map_layers() -> Dict[str, bool]:
    return {
        layer: (MAP_DATA_DIR / filename).exists()
        for layer, filename in LAYER_FILES.items()
    }


def _resolve_layer_path(layer: str) -> Path:
    if layer not in LAYER_FILES:
        raise ValueError(f"Unsupported map layer '{layer}'.")
    return MAP_DATA_DIR / LAYER_FILES[layer]


def _xyz_to_tms_y(z: int, y: int) -> int:
    return (1 << z) - 1 - y


def _detect_tile_mime(tile_data: bytes) -> str:
    if tile_data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if tile_data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    return "application/octet-stream"


def read_mbtiles_tile(layer: str, z: int, x: int, y: int) -> Tuple[bytes, str]:
    layer_path = _resolve_layer_path(layer)
    if not layer_path.exists():
        raise FileNotFoundError(f"Offline layer file not found: {layer_path}")

    tms_y = _xyz_to_tms_y(z, y)
    conn = sqlite3.connect(layer_path)
    try:
        cursor = conn.execute(
            """
            SELECT tile_data
            FROM tiles
            WHERE zoom_level = ? AND tile_column = ? AND tile_row = ?
            LIMIT 1
            """,
            (z, x, tms_y),
        )
        row = cursor.fetchone()
    finally:
        conn.close()

    if row is None:
        raise LookupError("Tile is not available in offline dataset.")

    tile_data = bytes(row[0])
    return tile_data, _detect_tile_mime(tile_data)


def _clip_lat(lat: float) -> float:
    return max(-85.05112878, min(85.05112878, lat))


def _lon_lat_to_global_pixel(lon: float, lat: float, zoom: int) -> Tuple[float, float]:
    lat = _clip_lat(lat)
    world_size = (1 << zoom) * 256.0

    x = (lon + 180.0) / 360.0 * world_size
    sin_lat = math.sin(math.radians(lat))
    y = (0.5 - math.log((1.0 + sin_lat) / (1.0 - sin_lat)) / (4.0 * math.pi)) * world_size
    return x, y


def _load_tile_rgb(layer: str, z: int, x: int, y: int) -> Image.Image:
    try:
        tile_data, _ = read_mbtiles_tile(layer=layer, z=z, x=x, y=y)
    except LookupError:
        return Image.new("RGB", (256, 256), (28, 34, 40))

    return Image.open(io.BytesIO(tile_data)).convert("RGB")


def build_satellite_rgb_from_bbox(
    *,
    south: float,
    west: float,
    north: float,
    east: float,
    zoom: int,
    max_output_size: Optional[int] = 1536,
) -> np.ndarray:
    if north <= south:
        raise ValueError("north must be greater than south.")
    if east <= west:
        raise ValueError("east must be greater than west. Dateline crossing is not supported.")
    if not (0 <= zoom <= 17):
        raise ValueError("zoom must be between 0 and 17.")

    layers = available_map_layers()
    if not layers.get("satellite"):
        raise FileNotFoundError(
            f"Missing offline base map: {MAP_DATA_DIR / LAYER_FILES['satellite']}"
        )

    x0, y0 = _lon_lat_to_global_pixel(west, north, zoom)
    x1, y1 = _lon_lat_to_global_pixel(east, south, zoom)

    if x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid bbox produced an empty pixel window.")

    max_tile_index = (1 << zoom) - 1

    tile_x_min = max(0, min(max_tile_index, int(math.floor(x0 / 256.0))))
    tile_y_min = max(0, min(max_tile_index, int(math.floor(y0 / 256.0))))
    tile_x_max = max(0, min(max_tile_index, int(math.floor((x1 - 1) / 256.0))))
    tile_y_max = max(0, min(max_tile_index, int(math.floor((y1 - 1) / 256.0))))

    tiles_w = tile_x_max - tile_x_min + 1
    tiles_h = tile_y_max - tile_y_min + 1

    stitched = Image.new("RGB", (tiles_w * 256, tiles_h * 256))
    for ty in range(tile_y_min, tile_y_max + 1):
        for tx in range(tile_x_min, tile_x_max + 1):
            tile = _load_tile_rgb("satellite", zoom, tx, ty)
            stitched.paste(tile, ((tx - tile_x_min) * 256, (ty - tile_y_min) * 256))

    crop_left = int(round(x0 - tile_x_min * 256.0))
    crop_top = int(round(y0 - tile_y_min * 256.0))
    crop_right = int(round(x1 - tile_x_min * 256.0))
    crop_bottom = int(round(y1 - tile_y_min * 256.0))

    cropped = stitched.crop((crop_left, crop_top, crop_right, crop_bottom))

    if max_output_size is not None and max_output_size > 0:
        width, height = cropped.size
        largest_dim = max(width, height)
        if largest_dim > max_output_size:
            scale = max_output_size / float(largest_dim)
            resized = (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            )
            cropped = cropped.resize(resized, Image.Resampling.BILINEAR)

    return np.array(cropped)
