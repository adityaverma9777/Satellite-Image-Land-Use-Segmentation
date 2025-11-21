# Satellite Image Land-Use Segmentation

Full-stack AI web application for semantic segmentation of satellite imagery into 4 classes.

## Quick Start (Docker)

1. Clone the repo.
2. Put your trained model at one of these locations:

	- `best_model (3).pth` (highest priority)
	- `best_model.pth`
	- `models/best_model.pth`
3. Optional (for offline/self-hosted maps), add MBTiles files:

	- `map-data/satellite.mbtiles` (offline satellite source)
	- `map-data/terrain-rgb.mbtiles` (offline 3D terrain source)

4. Build and start containers:

```bash
docker compose up --build
```

5. Open:

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`

To run in detached mode:

```bash
docker compose up -d --build
```

To stop the system:

```bash
docker compose down
```

## Project Structure

```text
project/
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   ├── inference.py
│   ├── utils.py
│   ├── requirements.txt
│   └── outputs/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── App.jsx
│   │   └── ...
├── best_model.pth
├── docker-compose.yml
└── README.md
```

## Backend (FastAPI)

### Features

- Loads model once at startup from `best_model.pth`
- Auto-discovers model checkpoint (`MODEL_PATH` env override supported)
- Uses GPU automatically if available
- `/predict` endpoint for image segmentation (legacy upload route)
- `/predict/area` for map-selected area inference
- Tiled inference for large satellite images
- Colorized mask output and blended overlay output
- GeoTIFF mask export preserving source metadata when available
- Free online map fallback (satellite + terrain) when offline MBTiles are missing

### API Endpoints

- `GET /health`
- `GET /classes`
- `GET /map/status`
- `GET /map/tile/{z}/{x}/{y}?layer=satellite|terrain`
- `POST /predict`
- `POST /predict/area`
- `POST /metrics/miou`
- `GET /outputs/{prediction_id}/...` (static files)

## World Map Mode

- The dashboard is map-first and does not require image upload in the main flow.
- Click two points on the map to define a bounding box, then run segmentation on that selected area.
- Map is location-aware and includes geolocate controls.
- By default, map tiles come from free online services via backend proxy.
- If MBTiles are present in `map-data/`, backend prefers those for offline/self-hosted operation.
- If `terrain-rgb.mbtiles` is available, terrain is rendered in 3D from local data.
- Dateline-crossing boxes are currently not supported.

## Frontend (React + MapLibre)

### Features

- Offline world map area selection (click two corners)
- Inference controls (tiling, tile size, batch size, alpha)
- Before/After interactive slider
- 3D satellite + terrain map rendering with location-aware controls
- Layer toggles (original vs prediction)
- Download PNG outputs

### Install and Run

```bash
cd frontend
npm install
npm run dev
```

Frontend default URL: `http://localhost:5173`
Backend default URL: `http://127.0.0.1:8000`

## Environment Configuration

Frontend API base URL can be changed in `.env`:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Backend map fallback can be controlled with environment variable:

```bash
MAP_ONLINE_FALLBACK=true
```

Model file and runtime device can be controlled with:

```bash
MODEL_PATH=/app/best_model\ \(3\).pth
MODEL_DEVICE=auto   # auto | cpu | cuda
```

To build Docker backend with CUDA-enabled PyTorch wheels:

```bash
TORCH_WHL_CHANNEL=cu121 docker compose up --build
```

Use `TORCH_WHL_CHANNEL=cpu` for smallest CPU-only images.

- `true` (default): use free online satellite/terrain when offline MBTiles are unavailable
- `false`: require local MBTiles only

## Notes

- If the uploaded input is georeferenced (GeoTIFF), the backend extracts bounds and preserves spatial metadata for GeoTIFF export.
- If georeference is unavailable, map overlay uses approximate fallback bounds for visualization only.
- Model architecture expected by backend loader: U-Net with ResNet-50 encoder and 4 output classes.
- Inference path is configured for 384x384 model input.
- API responses now return file download URLs by default; inline base64 images are optional (`include_base64=true`).
- If a Windows App Control policy blocks local PyTorch DLLs (`WinError 4551`), use Docker mode.
