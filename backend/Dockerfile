# Builds the API image. Build context is the repo root (see docker-compose.yml), because the app's
# path math (backend/app/config.py's ROOT) and the campus config in config/ both live above backend/.
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \n    && pip install --no-cache-dir -r backend/requirements.txt \
    && python -m spacy download en_core_web_sm

COPY . .

# Bakes ~0.7GB of ML weights into the image at build time (yolo11n, CLIP, MiniLM) -- keeps the
# container self-contained after build, no shared volume or build-time network needed on the VPS.
# The real secret comes from the environment at runtime; this is only so the script can import Settings.
RUN JWT_SECRET=download-only python scripts/download_weights.py

RUN chmod +x backend/docker-entrypoint.sh

EXPOSE 8010
ENTRYPOINT ["backend/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8010", "--workers", "1"]
