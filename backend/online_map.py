from __future__ import annotations

import io
import math
import os
import urllib.error
import urllib.request
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

ONLINE_TILE_URLS = {
    # Free public basemap; check provider terms and attribution for your usage.
    "satellite": "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    # Free global terrain RGB tiles (terrarium encoding) for client-side terrain rendering.
    "terrain": "https://elevation-tiles-prod.s3.amazonaws.com/terrarium/{z}/{x}/{y}.png",
}

ONLINE_ATTRIBUTION = {
    "satellite": "Esri World Imagery",
    "terrain": "Mapzen Terrain (Terrarium)",
}


def online_fallback_enabled() -> bool:
    value = os.getenv("MAP_ONLINE_FALLBACK", "true").strip().lower()
    return value in {"1", "true", "yes", "on"}


def available_online_layers() -> Dict[str, bool]:
    enabled = online_fallback_enabled()
    return {layer: enabled for layer in ONLINE_TILE_URLS}


def online_terrain_encoding() -> str:
    return "terrarium"


def _detect_tile_mime(tile_data: bytes) -> str:
    if tile_data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if tile_data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    return "application/octet-stream"


def fetch_online_tile(layer: str, z: int, x: int, y: int, timeout_sec: int = 15) -> Tuple[bytes, str]:
    if layer not in ONLINE_TILE_URLS:
        raise ValueError(f"Unsupported online map layer '{layer}'.")

    if not online_fallback_enabled():
        raise FileNotFoundError("Online map fallback is disabled by MAP_ONLINE_FALLBACK=false.")

    if z < 0:
        raise ValueError("z must be >= 0.")

    max_index = (1 << z) - 1
    if x < 0 or y < 0 or x > max_index or y > max_index:
        raise LookupError("Requested tile coordinates are outside the valid XYZ range.")

    url = ONLINE_TILE_URLS[layer].format(z=z, x=x, y=y)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "SatelliteLandUseSegmentation/1.0",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            data = response.read()
            content_type = response.headers.get_content_type() or _detect_tile_mime(data)
            return data, content_type
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise LookupError("Online tile was not found for this z/x/y.") from exc
        raise RuntimeError(f"Online tile service error ({exc.code}) for {layer}.") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach online tile service for {layer}: {exc.reason}") from exc


def _clip_lat(lat: float) -> float:
    return max(-85.05112878, min(85.05112878, lat))


def _lon_lat_to_global_pixel(lon: float, lat: float, zoom: int) -> Tuple[float, float]:
    lat = _clip_lat(lat)
    world_size = (1 << zoom) * 256.0

    px = (lon + 180.0) / 360.0 * world_size
    sin_lat = math.sin(math.radians(lat))
    py = (0.5 - math.log((1.0 + sin_lat) / (1.0 - sin_lat)) / (4.0 * math.pi)) * world_size
    return px, py


def _load_online_satellite_tile(z: int, x: int, y: int) -> Image.Image:
    try:
        tile_data, _ = fetch_online_tile(layer="satellite", z=z, x=x, y=y)
        return Image.open(io.BytesIO(tile_data)).convert("RGB")
    except LookupError:
        return Image.new("RGB", (256, 256), (28, 34, 40))


def build_satellite_rgb_from_bbox_online(
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

    if not online_fallback_enabled():
        raise FileNotFoundError("Online map fallback is disabled by MAP_ONLINE_FALLBACK=false.")

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
            tile = _load_online_satellite_tile(zoom, tx, ty)
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
