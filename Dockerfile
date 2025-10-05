# Minimal runtime image for the FastAPI model server
ARG PYTHON_VERSION=3.11-slim
FROM python:${PYTHON_VERSION} AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Optional GPU build for llama-cpp-python: set at build time
#   docker build --build-arg ENABLE_CUBLAS=1 ...
ARG ENABLE_CUBLAS=0
ENV ENABLE_CUBLAS=${ENABLE_CUBLAS}

FROM base AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY model_server/requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip
# If GPU build requested, propagate CMAKE_ARGS to build cublas variant
RUN if [ "$ENABLE_CUBLAS" = "1" ]; then \
      export CMAKE_ARGS="-DLLAMA_CUBLAS=1"; \
      python -m pip install --no-cache-dir -r /tmp/requirements.txt; \
    else \
      python -m pip install --no-cache-dir -r /tmp/requirements.txt; \
    fi \
    && python -c "import site,shutil,sys; d=site.getsitepackages()[0]; shutil.make_archive('/wheels','zip',d)"

FROM base AS runtime
# Copy site-packages from builder
COPY --from=builder /wheels.zip /wheels.zip
RUN python - << 'PY'
import zipfile, sys, site, os
dst = site.getsitepackages()[0]
with zipfile.ZipFile('/wheels.zip') as z:
    z.extractall(dst)
os.remove('/wheels.zip')
PY

# Copy application code (server only; data/models mounted at runtime)
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

# Create a non-root user
RUN useradd -m -u 10001 appuser
USER appuser

CMD ["uvicorn", "model_server.server:app", "--host", "0.0.0.0", "--port", "8000"]
