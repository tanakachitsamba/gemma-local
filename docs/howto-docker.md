How to Build and Run with Docker

Why containers?
- Reproducible runtime, small surface area, fast onboarding.
- Optional multi-arch builds (amd64/arm64) and GHCR publishing via CI.

Local build (CPU)
- `docker build -t gemma-local:latest .`

Local run
- `docker run --rm -p 8000:8000 \
   -v %CD%/models:/app/models \
   -e MODEL_PATH="/app/models/gemma-2-2b-it.Q4_K_M.gguf" \
   gemma-local:latest`

Compose (recommended)
- `docker compose up --build`
- Mounts `./models` to `/app/models` and reads `./config.yaml` if present.
- To enable optional cuBLAS build for `llama-cpp-python` (GPU-aware):
  - `docker compose build --build-arg ENABLE_CUBLAS=1`
  - Run with GPU (varies by host): `docker run --gpus=all ...` or compose GPU stanza.

GHCR images via CI
- Workflow `docker.yml` builds and pushes multi-arch images to GHCR on push to `main` and on tags.
- Pull latest:
  - `docker pull ghcr.io/<your-org-or-user>/gemma-local:latest`

Notes
- The container runs as a non‑root user and exposes 8000.
- GPU builds require compatible CUDA libraries on host and inside image; keep CPU unless needed.
- Set `MODEL_PATH` (and other envs) at runtime; see `docs/howto-run-server.md` for settings.
