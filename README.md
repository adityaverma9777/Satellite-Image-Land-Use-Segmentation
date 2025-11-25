# Satellite Image Land-Use Segmentation

This project is a full-stack app for land-use segmentation from satellite imagery.
You can either upload an image or select an area directly on the map, then run segmentation and download results.

Tech stack:
- Backend: FastAPI + PyTorch
- Frontend: React + MapLibre
- Deployment: Docker Compose

## Start Here: Model File Placement

Before running anything, place your trained model exactly here:

`models/best_model.pth`

That is the default and recommended location for this repo.

If your file is currently named something else (for example `best_model (3).pth`), rename it to `best_model.pth` and place it in the `models/` folder.

## Quick Start (Docker)

1. Make sure Docker Desktop is running.
2. Confirm model exists at `models/best_model.pth`.
3. Optional for offline map mode: place MBTiles files in `map-data/`:
   - `map-data/satellite.mbtiles`
   - `map-data/terrain-rgb.mbtiles`
4. Run:

```bash
docker compose up --build
```

5. Open:
   - Frontend: http://localhost:5173
   - Backend: http://localhost:8000

Run in detached mode:

```bash
docker compose up -d --build
```

Stop everything:

```bash
docker compose down
```

## Local Dev (Without Docker)

Backend:

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

## Core Features

- Map-first workflow with area selection
- Upload-based segmentation endpoint (legacy but still supported)
- Tiled inference for large satellite crops
- Overlay and color-mask output generation
- Output files served by URL (base64 optional)
- Auto CPU/GPU selection with optional override
- Online map fallback when local MBTiles are unavailable

## Main API Endpoints

- `GET /health`
- `GET /classes`
- `GET /map/status`
- `GET /map/tile/{z}/{x}/{y}?layer=satellite|terrain`
- `POST /predict`
- `POST /predict/area`
- `POST /metrics/miou`
- `GET /outputs/{prediction_id}/...`

## Configuration

Frontend (`frontend/.env`):

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Backend environment variables:

```bash
MODEL_PATH=/app/models/best_model.pth
MODEL_DEVICE=auto
MAP_ONLINE_FALLBACK=true
```

`MODEL_DEVICE` values:
- `auto` (default): CUDA if available, else CPU
- `cuda`: force GPU
- `cpu`: force CPU

Optional Docker build arg for PyTorch wheels:

```bash
TORCH_WHL_CHANNEL=cu121 docker compose up --build
```

Use `TORCH_WHL_CHANNEL=cpu` for a lighter CPU-only image.

## Map Workflow

1. Open dashboard.
2. Zoom in to your area of interest.
3. Click two corners to define a bounding box.
4. Run segmentation.
5. Compare original vs prediction in the slider and download outputs.

Notes:
- Dateline-crossing boxes are not supported.
- Better zoom generally gives better local detail.

## Performance Notes

- Backend supports direct and tiled inference.
- Tiling is safer for larger areas and high-resolution requests.
- On GPU, mixed precision is used where appropriate.
- On CPU, threading is tuned for inference throughput.

## Troubleshooting

Model not loading:
- Verify `models/best_model.pth` exists.
- Check `GET /health` for model path and device info.

Very slow predictions:
- Use smaller selected area or tune tile size/batch size.
- If GPU is available, confirm backend actually started on CUDA from `GET /health`.

Map tiles not loading:
- Confirm internet access for online fallback.
- Or provide local MBTiles in `map-data/`.

Windows DLL policy errors (for local non-Docker runs):
- If you hit `WinError 4551` with local PyTorch DLL loading, use Docker mode.

## Project Structure

```text
.
├── backend/
├── frontend/
├── models/
│   └── best_model.pth
├── map-data/
├── graphs/
├── Kaggle Notebook 1.ipynb
├── Kaggle Notebook 2.ipynb
├── Kaggle Notebook 3 FINAL.ipynb
└── docker-compose.yml
```

## Model Journey

See [MODEL_ANALYSIS.md](MODEL_ANALYSIS.md) for the experiment-by-experiment breakdown of all three Kaggle notebooks, including graph references and what changed between versions.
