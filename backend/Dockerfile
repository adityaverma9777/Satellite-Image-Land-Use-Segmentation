FROM python:3.10-slim

ARG TORCH_WHL_CHANNEL=cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY backend/requirements.txt /tmp/requirements.txt

RUN apt-get update && \
    apt-get install -y --no-install-recommends libexpat1 && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip && \
        if [ "$TORCH_WHL_CHANNEL" = "cu121" ]; then \
            pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r /tmp/requirements.txt; \
        elif [ "$TORCH_WHL_CHANNEL" = "cpu" ]; then \
            pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt; \
        else \
            pip install -r /tmp/requirements.txt; \
        fi

COPY backend /app/backend
COPY best_model.pth /app/best_model.pth
COPY map-data /app/map-data

RUN mkdir -p /app/backend/outputs

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
