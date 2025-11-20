from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from backend.inference import direct_inference, tiled_inference
from backend.model import NUM_CLASSES, load_model
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
MODEL_PATH = BASE_DIR / "models" / "best_model.pth"
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.on_event("startup")
def startup_event() -> None:
    global model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")

    model = load_model(MODEL_PATH, device)
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


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    use_tiling: bool = Form(True),
    tile_size: int = Form(256),
    batch_size: int = Form(8),
    overlay_alpha: float = Form(0.45),
    export_geotiff: bool = Form(True),
) -> Dict[str, Any]:
    global model

    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image_rgb, geo_meta = load_image_and_geo_metadata(payload, file.filename or "upload")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse image: {exc}") from exc

    t0 = time.perf_counter()

    try:
        if use_tiling:
            pred_mask = tiled_inference(
                image_rgb=image_rgb,
                model=model,
                device=device,
                tile_size=tile_size,
                batch_size=batch_size,
            )
        else:
            pred_mask = direct_inference(
                image_rgb=image_rgb,
                model=model,
                device=device,
                input_size=256,
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
            "overlay_alpha": overlay_alpha,
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
        "images_base64": {
            "original": png_base64(image_rgb),
            "prediction": png_base64(colored_mask),
            "overlay": png_base64(overlay),
            "mask": png_base64(pred_mask, mode="L"),
        },
    }


@app.post("/metrics/miou")
async def evaluate_miou(
    image: UploadFile = File(...),
    ground_truth: UploadFile = File(...),
    use_tiling: bool = Form(True),
    tile_size: int = Form(256),
    batch_size: int = Form(8),
) -> Dict[str, Any]:
    global model

    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    image_bytes = await image.read()
    gt_bytes = await ground_truth.read()
    if not image_bytes or not gt_bytes:
        raise HTTPException(status_code=400, detail="Image and ground-truth files are required.")

    try:
        image_rgb, _ = load_image_and_geo_metadata(image_bytes, image.filename or "image")
        gt_mask = load_label_mask(gt_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Input parsing failed: {exc}") from exc

    if use_tiling:
        pred_mask = tiled_inference(
            image_rgb=image_rgb,
            model=model,
            device=device,
            tile_size=tile_size,
            batch_size=batch_size,
        )
    else:
        pred_mask = direct_inference(
            image_rgb=image_rgb,
            model=model,
            device=device,
            input_size=256,
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
