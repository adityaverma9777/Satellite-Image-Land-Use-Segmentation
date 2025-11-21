from __future__ import annotations

import os
import math
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from backend.inference import tiled_inference
from backend.model import NUM_CLASSES, load_model
from backend.offline_map import (
    MAP_DATA_DIR,
    available_map_layers,
    build_satellite_rgb_from_bbox,
    read_mbtiles_tile,
)
from backend.online_map import (
    ONLINE_ATTRIBUTION,
    available_online_layers,
    build_satellite_rgb_from_bbox_online,
    fetch_online_tile,
    online_fallback_enabled,
    online_terrain_encoding,
)
from backend.utils import (
    CLASS_LABELS,
    blend_images,
    compute_miou,
    colorize_mask,
    load_label_mask,
    load_image_and_geo_metadata,
    png_base64,
    save_mask_as_geotiff,
    save_png,
)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "backend" / "outputs"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Satellite Land-Use Segmentation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

model: Optional[torch.nn.Module] = None
loaded_model_path: Optional[Path] = None


def _resolve_model_path() -> Path:
    configured_path = os.getenv("MODEL_PATH", "").strip()
    if configured_path:
        candidate = Path(configured_path)
        if not candidate.is_absolute():
            candidate = (BASE_DIR / candidate).resolve()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"MODEL_PATH points to a missing file: {candidate}")

    preferred = [
        BASE_DIR / "best_model (3).pth",
        BASE_DIR / "best_model.pth",
        BASE_DIR / "models" / "best_model.pth",
    ]
    for candidate in preferred:
        if candidate.exists():
            return candidate

    discovered = sorted(
        set(BASE_DIR.glob("best_model*.pth")) | set((BASE_DIR / "models").glob("best_model*.pth")),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if discovered:
        return discovered[0]

    raise FileNotFoundError(
        "No model checkpoint found. Provide MODEL_PATH or add one of: "
        "best_model (3).pth, best_model.pth, or models/best_model.pth"
    )


def _resolve_device() -> torch.device:
    device_pref = os.getenv("MODEL_DEVICE", "auto").strip().lower()
    if device_pref == "cpu":
        return torch.device("cpu")

    if device_pref == "cuda":
        if not torch.cuda.is_available():
            print("[startup] MODEL_DEVICE=cuda requested but CUDA is not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _configure_torch_runtime(active_device: torch.device) -> None:
    if active_device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        return

    cpu_count = os.cpu_count() or 1
    preferred_threads = max(1, cpu_count - 1)
    thread_override = os.getenv("TORCH_NUM_THREADS", "").strip()
    if thread_override:
        try:
            preferred_threads = max(1, int(thread_override))
        except ValueError:
            print(f"[startup] Ignoring invalid TORCH_NUM_THREADS={thread_override!r}")

    torch.set_num_threads(preferred_threads)
    interop_threads = min(4, preferred_threads)
    try:
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError:
        pass


device = _resolve_device()

MIN_AREA_ZOOM = 10
MAX_SOURCE_TILES = 144
DEFAULT_TILE_OVERLAP = 48
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_BACKGROUND_CLASS = 0


class AreaPredictionRequest(BaseModel):
    south: float = Field(..., ge=-85.05112878, le=85.05112878)
    west: float = Field(..., ge=-180.0, le=180.0)
    north: float = Field(..., ge=-85.05112878, le=85.05112878)
    east: float = Field(..., ge=-180.0, le=180.0)
    map_source: Literal["auto", "offline", "online"] = "auto"
    zoom: int = Field(12, ge=0, le=17)
    use_tiling: bool = True
    tile_size: int = Field(384, ge=64, le=1024)
    batch_size: int = Field(8, ge=1, le=32)
    tile_overlap: int = Field(DEFAULT_TILE_OVERLAP, ge=0, le=256)
    confidence_threshold: float = Field(DEFAULT_CONFIDENCE_THRESHOLD, ge=0.0, le=1.0)
    background_class: int = Field(DEFAULT_BACKGROUND_CLASS, ge=0, le=255)
    overlay_alpha: float = Field(0.45, ge=0.0, le=1.0)
    include_base64: bool = False


def _clip_lat(lat: float) -> float:
    return max(-85.05112878, min(85.05112878, lat))


def _lon_lat_to_global_pixel(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    lat = _clip_lat(lat)
    world_size = (1 << zoom) * 256.0

    px = (lon + 180.0) / 360.0 * world_size
    sin_lat = math.sin(math.radians(lat))
    py = (0.5 - math.log((1.0 + sin_lat) / (1.0 - sin_lat)) / (4.0 * math.pi)) * world_size
    return px, py


def _validate_area_request_quality(
    *,
    south: float,
    west: float,
    north: float,
    east: float,
    zoom: int,
) -> Optional[str]:
    if zoom < MIN_AREA_ZOOM:
        return (
            f"Selected zoom level ({zoom}) is too low for reliable segmentation. "
            f"Please zoom in to at least {MIN_AREA_ZOOM} and select again."
        )

    x0, y0 = _lon_lat_to_global_pixel(west, north, zoom)
    x1, y1 = _lon_lat_to_global_pixel(east, south, zoom)

    width_px = max(1.0, x1 - x0)
    height_px = max(1.0, y1 - y0)

    tiles_w = int(math.ceil(width_px / 256.0))
    tiles_h = int(math.ceil(height_px / 256.0))
    tile_count = tiles_w * tiles_h
    if tile_count > MAX_SOURCE_TILES:
        return (
            "Selected area is too large for high-quality, responsive inference. "
            "Please zoom in and select a smaller region."
        )

    return None


def _map_capabilities() -> Dict[str, Any]:
    offline_layers = available_map_layers()
    online_layers = available_online_layers()

    satellite_ready = bool(offline_layers.get("satellite") or online_layers.get("satellite"))
    terrain_ready = bool(offline_layers.get("terrain") or online_layers.get("terrain"))

    if offline_layers.get("satellite"):
        active_source = "offline"
    elif online_layers.get("satellite"):
        active_source = "online"
    else:
        active_source = "unavailable"

    if offline_layers.get("terrain"):
        terrain_encoding = "mapbox"
    elif online_layers.get("terrain"):
        terrain_encoding = online_terrain_encoding()
    else:
        terrain_encoding = None

    return {
        "offline_layers": offline_layers,
        "online_layers": online_layers,
        "satellite_ready": satellite_ready,
        "terrain_ready": terrain_ready,
        "active_source": active_source,
        "terrain_encoding": terrain_encoding,
    }


@app.on_event("startup")
def startup_event() -> None:
    global model, loaded_model_path

    loaded_model_path = _resolve_model_path()
    _configure_torch_runtime(device)
    model = load_model(loaded_model_path, device)
    print(f"[startup] Model loaded from: {loaded_model_path}")
    print(f"[startup] Model loaded on device: {device}")


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "Satellite Land-Use Segmentation API",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "device": str(device),
        "model_loaded": model is not None,
        "model_path": str(loaded_model_path) if loaded_model_path else None,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "num_classes": NUM_CLASSES,
    }


@app.get("/classes")
def classes() -> Dict[str, Any]:
    return {
        "classes": [
            {"index": idx, "name": name}
            for idx, name in CLASS_LABELS.items()
        ]
    }


@app.get("/map/status")
def map_status() -> Dict[str, Any]:
    caps = _map_capabilities()
    return {
        "map_data_dir": str(MAP_DATA_DIR.resolve()),
        "layers": {
            "offline": caps["offline_layers"],
            "online": caps["online_layers"],
        },
        "online_fallback_enabled": online_fallback_enabled(),
        "satellite_ready": caps["satellite_ready"],
        "terrain_ready": caps["terrain_ready"],
        "active_source": caps["active_source"],
        "terrain_encoding": caps["terrain_encoding"],
        "satellite_attribution": ONLINE_ATTRIBUTION["satellite"],
        "terrain_attribution": ONLINE_ATTRIBUTION["terrain"],
    }


@app.get("/map/tile/{z}/{x}/{y}")
def map_tile(z: int, x: int, y: int, layer: str = "satellite") -> Response:
    offline_layers = available_map_layers()

    if offline_layers.get(layer):
        try:
            tile_data, media_type = read_mbtiles_tile(layer=layer, z=z, x=x, y=y)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    else:
        try:
            tile_data, media_type = fetch_online_tile(layer=layer, z=z, x=x, y=y)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return Response(
        content=tile_data,
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    use_tiling: bool = Form(True),
    tile_size: int = Form(384),
    batch_size: int = Form(8),
    tile_overlap: int = Form(DEFAULT_TILE_OVERLAP),
    confidence_threshold: float = Form(DEFAULT_CONFIDENCE_THRESHOLD),
    background_class: int = Form(DEFAULT_BACKGROUND_CLASS),
    overlay_alpha: float = Form(0.45),
    export_geotiff: bool = Form(True),
    include_base64: bool = Form(False),
) -> Dict[str, Any]:
    global model

    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    payload = file.file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image_rgb, geo_meta = load_image_and_geo_metadata(payload, file.filename or "upload")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse image: {exc}") from exc

    t0 = time.perf_counter()

    try:
        effective_overlap = tile_overlap if use_tiling else 0
        effective_batch_size = batch_size if use_tiling else 1
        pred_mask = tiled_inference(
            image_rgb=image_rgb,
            model=model,
            device=device,
            tile_size=tile_size,
            batch_size=effective_batch_size,
            overlap=effective_overlap,
            confidence_threshold=confidence_threshold,
            background_class=background_class,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    inference_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    colored_mask = colorize_mask(pred_mask)
    overlay = blend_images(image_rgb, colored_mask, alpha=overlay_alpha)

    prediction_id = uuid.uuid4().hex[:12]
    result_dir = OUTPUTS_DIR / prediction_id
    result_dir.mkdir(parents=True, exist_ok=True)

    original_path = result_dir / "original.png"
    prediction_path = result_dir / "prediction.png"
    overlay_path = result_dir / "overlay.png"
    mask_path = result_dir / "mask.png"

    save_png(original_path, image_rgb)
    save_png(prediction_path, colored_mask)
    save_png(overlay_path, overlay)
    save_png(mask_path, pred_mask, mode="L")

    geotiff_saved = False
    geotiff_path = result_dir / "mask.tif"
    if export_geotiff:
        geotiff_saved = save_mask_as_geotiff(pred_mask, geotiff_path, geo_meta)

    images_base64 = None
    if include_base64:
        images_base64 = {
            "original": png_base64(image_rgb),
            "prediction": png_base64(colored_mask),
            "overlay": png_base64(overlay),
            "mask": png_base64(pred_mask, mode="L"),
        }

    return {
        "prediction_id": prediction_id,
        "filename": file.filename,
        "shape": {
            "height": int(image_rgb.shape[0]),
            "width": int(image_rgb.shape[1]),
        },
        "settings": {
            "use_tiling": use_tiling,
            "tile_size": tile_size,
            "batch_size": batch_size,
            "tile_overlap": tile_overlap,
            "confidence_threshold": confidence_threshold,
            "background_class": background_class,
            "overlay_alpha": overlay_alpha,
            "include_base64": include_base64,
        },
        "timing": {
            "inference_ms": inference_ms,
        },
        "classes": [
            {"index": idx, "name": name}
            for idx, name in CLASS_LABELS.items()
        ],
        "downloads": {
            "original_png": f"/outputs/{prediction_id}/original.png",
            "prediction_png": f"/outputs/{prediction_id}/prediction.png",
            "overlay_png": f"/outputs/{prediction_id}/overlay.png",
            "mask_png": f"/outputs/{prediction_id}/mask.png",
            "mask_geotiff": f"/outputs/{prediction_id}/mask.tif" if geotiff_saved else None,
        },
        "geo": {
            "has_metadata": geo_meta is not None,
            "crs": geo_meta.get("crs") if geo_meta else None,
            "bounds_epsg4326": geo_meta.get("bounds_epsg4326") if geo_meta else None,
            "geotiff_saved": geotiff_saved,
        },
        "images_base64": images_base64,
    }


@app.post("/predict/area")
def predict_area(payload: AreaPredictionRequest) -> Dict[str, Any]:
    global model

    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    south = min(payload.south, payload.north)
    north = max(payload.south, payload.north)
    west = min(payload.west, payload.east)
    east = max(payload.west, payload.east)

    if north == south:
        raise HTTPException(
            status_code=400,
            detail="Selected area height is zero. Pick two points with different latitudes.",
        )
    if east == west:
        raise HTTPException(
            status_code=400,
            detail="Selected area width is zero. Pick two points with different longitudes.",
        )

    requested_zoom = int(max(0, min(17, payload.zoom)))
    area_validation_error = _validate_area_request_quality(
        south=south,
        west=west,
        north=north,
        east=east,
        zoom=requested_zoom,
    )
    if area_validation_error:
        raise HTTPException(status_code=400, detail=area_validation_error)

    source_used = "offline"
    try:
        if payload.map_source in {"auto", "offline"}:
            image_rgb = build_satellite_rgb_from_bbox(
                south=south,
                west=west,
                north=north,
                east=east,
                zoom=requested_zoom,
                max_output_size=None,
            )
        else:
            raise FileNotFoundError("Offline source was not selected.")
    except FileNotFoundError:
        if payload.map_source == "offline":
            raise HTTPException(
                status_code=503,
                detail=(
                    "Offline satellite map is not configured. "
                    "Add map-data/satellite.mbtiles or use map_source='online'."
                ),
            ) from None

        try:
            image_rgb = build_satellite_rgb_from_bbox_online(
                south=south,
                west=west,
                north=north,
                east=east,
                zoom=requested_zoom,
                max_output_size=None,
            )
            source_used = "online"
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to read online map data: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read map data: {exc}") from exc

    t0 = time.perf_counter()
    try:
        effective_overlap = payload.tile_overlap if payload.use_tiling else 0
        effective_batch_size = payload.batch_size if payload.use_tiling else 1
        pred_mask = tiled_inference(
            image_rgb=image_rgb,
            model=model,
            device=device,
            tile_size=payload.tile_size,
            batch_size=effective_batch_size,
            overlap=effective_overlap,
            confidence_threshold=payload.confidence_threshold,
            background_class=payload.background_class,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    inference_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    colored_mask = colorize_mask(pred_mask)
    overlay = blend_images(image_rgb, colored_mask, alpha=payload.overlay_alpha)

    prediction_id = f"area-{uuid.uuid4().hex[:10]}"
    result_dir = OUTPUTS_DIR / prediction_id
    result_dir.mkdir(parents=True, exist_ok=True)

    save_png(result_dir / "original.png", image_rgb)
    save_png(result_dir / "prediction.png", colored_mask)
    save_png(result_dir / "overlay.png", overlay)
    save_png(result_dir / "mask.png", pred_mask, mode="L")

    geo_bounds = [
        [south, west],
        [north, east],
    ]

    images_base64 = None
    if payload.include_base64:
        images_base64 = {
            "original": png_base64(image_rgb),
            "prediction": png_base64(colored_mask),
            "overlay": png_base64(overlay),
            "mask": png_base64(pred_mask, mode="L"),
        }

    return {
        "prediction_id": prediction_id,
        "filename": (
            f"{source_used}-area-z{payload.zoom}-"
            f"{south:.4f}-{west:.4f}-{north:.4f}-{east:.4f}"
        ),
        "source": f"{source_used}_satellite",
        "shape": {
            "height": int(image_rgb.shape[0]),
            "width": int(image_rgb.shape[1]),
        },
        "settings": {
            "use_tiling": payload.use_tiling,
            "tile_size": payload.tile_size,
            "batch_size": payload.batch_size,
            "tile_overlap": payload.tile_overlap,
            "confidence_threshold": payload.confidence_threshold,
            "background_class": payload.background_class,
            "overlay_alpha": payload.overlay_alpha,
            "include_base64": payload.include_base64,
            "zoom": requested_zoom,
            "map_source": source_used,
        },
        "timing": {
            "inference_ms": inference_ms,
        },
        "classes": [
            {"index": idx, "name": name}
            for idx, name in CLASS_LABELS.items()
        ],
        "downloads": {
            "original_png": f"/outputs/{prediction_id}/original.png",
            "prediction_png": f"/outputs/{prediction_id}/prediction.png",
            "overlay_png": f"/outputs/{prediction_id}/overlay.png",
            "mask_png": f"/outputs/{prediction_id}/mask.png",
            "mask_geotiff": None,
        },
        "geo": {
            "has_metadata": True,
            "crs": "EPSG:4326",
            "bounds_epsg4326": geo_bounds,
            "geotiff_saved": False,
        },
        "images_base64": images_base64,
    }


@app.post("/metrics/miou")
def evaluate_miou(
    image: UploadFile = File(...),
    ground_truth: UploadFile = File(...),
    use_tiling: bool = Form(True),
    tile_size: int = Form(384),
    batch_size: int = Form(8),
    tile_overlap: int = Form(DEFAULT_TILE_OVERLAP),
    confidence_threshold: float = Form(DEFAULT_CONFIDENCE_THRESHOLD),
    background_class: int = Form(DEFAULT_BACKGROUND_CLASS),
) -> Dict[str, Any]:
    global model

    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    image_bytes = image.file.read()
    gt_bytes = ground_truth.file.read()
    if not image_bytes or not gt_bytes:
        raise HTTPException(status_code=400, detail="Image and ground-truth files are required.")

    try:
        image_rgb, _ = load_image_and_geo_metadata(image_bytes, image.filename or "image")
        gt_mask = load_label_mask(gt_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Input parsing failed: {exc}") from exc

    effective_overlap = tile_overlap if use_tiling else 0
    effective_batch_size = batch_size if use_tiling else 1
    pred_mask = tiled_inference(
        image_rgb=image_rgb,
        model=model,
        device=device,
        tile_size=tile_size,
        batch_size=effective_batch_size,
        overlap=effective_overlap,
        confidence_threshold=confidence_threshold,
        background_class=background_class,
    )

    if gt_mask.shape != pred_mask.shape:
        gt_mask = (
            np.array(
                Image.fromarray(gt_mask, mode="L").resize(
                    (pred_mask.shape[1], pred_mask.shape[0]),
                    Image.Resampling.NEAREST,
                )
            )
            .astype(np.uint8)
        )

    metrics = compute_miou(pred_mask=pred_mask, true_mask=gt_mask, num_classes=NUM_CLASSES)

    class_metrics = [
        {
            "index": idx,
            "name": CLASS_LABELS[idx],
            "iou": metrics["class_ious"].get(idx),
        }
        for idx in CLASS_LABELS
    ]

    return {
        "miou": metrics["miou"],
        "classes": class_metrics,
    }
