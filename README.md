
# Gemma Local Runner (Go + Python)

This project glues together a FastAPI wrapper around llama.cpp and a thin Go proxy that streams OpenAI-compatible responses. Bring your own UI, point it at the proxy, and you get /v1/chat/completions with streaming on your workstation.

## What's here
- FastAPI service that embeds llama-cpp-python and exposes OpenAI-style chat completions with optional SSE streaming.
- Minimal Go sidecar that forwards requests to the Python service and replays the stream for browser clients.
- PowerShell helpers for standing up a virtualenv and keeping model files organised.

## Prerequisites
- Windows, macOS, or Linux.
- Python 3.10 or newer.
- Go 1.21 or newer if you want the proxy.
- A Gemma or compatible GGUF checkpoint. Examples:
  - bartowski/gemma-2-2b-it-GGUF
  - TheBloke/gemma-2b-it-GGUF
  - Any other GGUF that works with llama.cpp (NeoX conversions work, just mind the RAM cost).

Download the .gguf you want to run and drop it under gemma-local/models/. Accept the Gemma licence where required.

## Python model server quickstart
1. Create a virtual environment and install deps:

    python -m venv .venv
    .\.venv\Scripts\pip install --upgrade pip
    .\.venv\Scripts\pip install -r model_server/requirements.txt

2. Configure the environment (adjust for your box):

    set MODEL_PATH=./models/gemma-2-2b-it.Q4_K_M.gguf
    set N_CTX=4096

3. Start the server (dev):

    python -m pip install -r model_server/requirements-dev.txt
    uvicorn model_server.server:app --reload --host 127.0.0.1 --port 8000

4. Try a request (non-stream):

    curl -s http://127.0.0.1:8000/v1/chat/completions \
      -H "content-type: application/json" \
      -d '{"model":"gemma","stream":false,"messages":[{"role":"user","content":"Hello"}]}'

## Testing

Python tests use fakes and don’t require llama-cpp:

    python -m pip install -r model_server/requirements-dev.txt
    pytest -q model_server/tests

## Experiments: fine-tune and evaluate

This repo includes small, pragmatic scaffolding:
- `training/` — LoRA fine-tuning via PEFT/Transformers
- `configs/train_lora.example.yaml` — copy & tweak for your run
- `data/` — dataset notes and tiny JSONL samples
- `scripts/evaluate.py` — send prompts to the server and save generations

Suggested flow:
1) Collect or craft a small dataset (JSONL). Keep samples short and clean.
2) Train a LoRA adapter with `training/train_lora.py`.
3) Merge/export, convert to GGUF, and quantize for llama.cpp.
4) Serve locally and compare base vs. fine-tuned outputs.

## Notes

- The Python server falls back from chat-completions to plain completions for base models without a chat template.
- Concurrency is conservative: a single `Llama` instance and a generation lock keep things stable on consumer hardware.

## How-To Guides

- `docs/howto-install.md` — install Python deps and set up your environment
- `docs/howto-run-server.md` — run the API and stream outputs
- `docs/howto-tests.md` — run tests and view coverage
- `docs/howto-finetune.md` — fine-tune with LoRA (PEFT)
- `docs/howto-evaluate.md` — evaluate base vs. fine‑tuned outputs

## Docker

Build locally:

    docker build -t gemma-local:latest .

Run (mount models directory and set MODEL_PATH accordingly):

    docker run --rm -p 8000:8000 \
      -v %CD%/models:/app/models \
      -e MODEL_PATH="/app/models/gemma-2-2b-it.Q4_K_M.gguf" \
      gemma-local:latest

GitHub Actions publishes images to GHCR on pushes to `main` and tags. Pull via:

    docker pull ghcr.io/<your-org-or-user>/gemma-local:latest

Compose (recommended for local):

    docker compose up --build

- Mounts `./models` at `/app/models` and reads `./config.yaml` if present.
- To build with CUDA/cuBLAS support for `llama-cpp-python` (advanced), pass an ARG:

    docker compose build --build-arg ENABLE_CUBLAS=1

Then run with GPU access (varies by setup):
- Docker CLI: `docker run --gpus=all ...`
- Compose profiles or runtime flags depending on your NVIDIA Docker version.

Notes
- GPU builds require CUDA libraries on the host and inside the container. Prefer CPU builds unless you know you need GPU.
- Keep one worker process; the server uses a generation lock to stay stable on small machines.
