# Minimal runtime image for the FastAPI model server
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (optional); keep base light
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install runtime Python deps
COPY model_server/requirements.txt /app/model_server/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/model_server/requirements.txt

# Copy application code
COPY model_server /app/model_server

EXPOSE 8000

# Simple healthcheck using Python stdlib to avoid curl/wget deps
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python - << 'PY' || exit 1
import json, urllib.request
try:
    with urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3) as r:
        body = json.loads(r.read().decode('utf-8'))
        if not body.get('ok'): raise SystemExit(1)
except Exception:
    raise SystemExit(1)
PY

# Default environment can be overridden at runtime
ENV HOST=0.0.0.0 \
    PORT=8000 \
    MODEL_PATH="./models/gemma-2-2b-it.Q4_K_M.gguf"

CMD ["uvicorn", "model_server.server:app", "--host", "0.0.0.0", "--port", "8000"]

