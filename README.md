# Satellite Image Land-Use Segmentation

Full-stack AI web application for semantic segmentation of satellite imagery into 6 land-use classes:
Urban, Agriculture, Rangeland, Forest, Water, and Barren.

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

### Install and Run

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cd ..
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

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
