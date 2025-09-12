# Gemma Local Runner (Go + Python)

A minimal, local-first setup to run a small Gemma model on your machine with a Python FastAPI model server (llama.cpp backend) and a lightweight Go HTTP proxy that streams responses in an OpenAI-compatible way. You provide the UI in a separate folder; the Go proxy exposes the same `/v1/chat/completions` endpoint with streaming for easy integration.

## Overview
- Python model server: wraps `llama-cpp-python` to run a Gemma GGUF model locally and serve OpenAI-style Chat Completions with optional streaming (SSE).
- Go proxy: forwards requests to the Python server and streams responses to clients. Useful for process separation and future orchestration.
- Streaming: Server-Sent Events (`text/event-stream`) compatible with OpenAI’s `stream: true` behavior.

## Prerequisites
- Windows, macOS, or Linux. For Windows, Python wheels for `llama-cpp-python` are available.
- Python 3.10+ recommended.
- Go 1.21+.
- A Gemma small model in GGUF format (e.g., Gemma 2 2B-Instruct, Q4_K_M quant).

Important: Gemma weights require accepting the license. Use an official or community GGUF conversion such as:
- `bartowski/gemma-2-2b-it-GGUF` (Gemma 2 Instruct, multiple quantizations)
- `TheBloke/gemma-2b-it-GGUF` (Gemma 1 series)
 - For Gemma 3, look for `gemma-3-*-it-GGUF` variants with appropriate quantization.

For GPT-NeoX 20B style models (often referred to as “GPT-NeoX-20B”), look for GGUF conversions (e.g., `TheBloke/GPT-NeoX-20B-GGUF`). Be aware these are large models; even quantized they may require >12 GB RAM/VRAM and will be slow on CPU.

Place the chosen `*.gguf` file under `gemma-local/models/` and set `MODEL_PATH` accordingly.

## Repo Structure
- `model_server/` — FastAPI app using `llama-cpp-python`
- `scripts/` — helper scripts (PowerShell) and examples

## Setup: Python model server
1) Create a virtual environment and install dependencies:

   - CPU-only example:
     pip install --upgrade pip
     pip install -r model_server/requirements.txt

   If you need specific CPU/GPU wheels for `llama-cpp-python`, see:
     https://github.com/abetlen/llama-cpp-python

2) Put your model file, e.g. `gemma-2-2b-it.Q4_K_M.gguf`, into `gemma-local/models/`.

3) Configure environment (copy and edit as needed):

     set MODEL_PATH=./models/gemma-2-2b-it.Q4_K_M.gguf
     set N_CTX=4096
     set N_THREADS=8
     set HOST=127.0.0.1
     set PORT=8000

4) Run the model server (or use `scripts/run_fastapi.ps1` with parameters):

     python -m uvicorn model_server.server:app --host %HOST% --port %PORT%

5) Health check:

     curl http://127.0.0.1:8000/healthz

## Setup: FastAPI model server
Preferred and simplest path, runs locally via `llama-cpp-python`.

## Using the API (OpenAI-style)
Non-streaming example:

  curl -X POST http://127.0.0.1:8081/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"gemma\",\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}"

Streaming example (SSE):

  curl -N -X POST http://127.0.0.1:8081/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"gemma\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}"

The Go proxy will forward to the Python server, which generates OpenAI-style chunks (`data: { ... }`) and a final `data: [DONE]` line.

Models without built-in chat templates (e.g., some GPT-NeoX variants) are supported via a fallback to completion mode with a generic prompt format.

## UI Integration
Point your UI (in a separate folder) at the Go proxy endpoint `http://127.0.0.1:8081/v1/chat/completions`. For streaming, use EventSource or an SSE client.

CORS: the proxy sets permissive CORS headers. Tighten as needed for production.

## Notes
- Gemma requires accepting the license. Ensure you comply with the terms.
- Model size/perf: Start with a 2B Instruct quant like Q4_K_M for CPU-only systems.
- GPU builds of `llama.cpp` / `llama-cpp-python` can dramatically increase speed if available.

## GitHub
This directory is ready to initialize as a git repo and push to GitHub:

  cd gemma-local
  git init
  git add .
  git commit -m "Scaffold Gemma local runner (Go + Python)"
  gh repo create <your-repo> --public --source . --push

## Testing
- Python server tests (uses fakes, no model needed):

  python -m venv .venv
  .\.venv\Scripts\python -m pip install -r model_server/requirements.txt -r model_server/requirements-dev.txt
  .\.venv\Scripts\python -m pytest -q model_server/tests

- Go tests:

  go test ./orchestrator/...
  go test ./model_server_go/...
