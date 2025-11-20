# Satellite Image Land-Use Segmentation

Full-stack AI web application for semantic segmentation of satellite imagery into 6 land-use classes:
Urban, Agriculture, Rangeland, Forest, Water, and Barren.

## Quick Start (Docker)

1. Clone the repo.
2. Put your trained model at `models/best_model.pth`.
3. Build and start containers:

```bash
docker compose up --build
```

4. Open:

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
├── models/
│   └── best_model.pth
├── docker-compose.yml
└── README.md
```

## Backend (FastAPI)

### Features

- Loads model once at startup from `models/best_model.pth`
- Uses GPU automatically if available
- `/predict` endpoint for image segmentation
- Tiled inference for large satellite images
- Colorized mask output and blended overlay output
- GeoTIFF mask export preserving source metadata when available
- Optional `/metrics/miou` endpoint for evaluation against uploaded ground-truth mask

### API Endpoints

- `GET /health`
- `GET /classes`
- `POST /predict`
- `POST /metrics/miou`
- `GET /outputs/{prediction_id}/...` (static files)

## Frontend (React + Leaflet)

### Features

- Drag-and-drop image upload with preview
- Inference controls (tiling, tile size, batch size, alpha)
- Before/After interactive slider
- Leaflet map overlay with zoom, pan, and opacity control
- Layer toggles (original vs prediction)
- Download PNG and GeoTIFF outputs

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

## Notes

- If the uploaded input is georeferenced (GeoTIFF), the backend extracts bounds and preserves spatial metadata for GeoTIFF export.
- If georeference is unavailable, map overlay uses approximate fallback bounds for visualization only.
- Model architecture expected by backend loader: U-Net with ResNet-50 encoder and 6 output classes.
- If a Windows App Control policy blocks local PyTorch DLLs (`WinError 4551`), use Docker mode.
