How to Run the Server

1) Configure settings
   - Easiest: create `config.yaml` (copy from `configs/server.example.yaml`).
   - Or set environment variables directly.
   - Precedence: env vars > config.yaml > defaults.
   - Minimal required: `MODEL_PATH` to your `.gguf`.
   - Optional tuning (env keys):
     - `N_CTX` (default 4096)
     - `N_THREADS` (defaults to CPU count)
     - `N_GPU_LAYERS` (0 for CPU-only)
     - `SYSTEM_PROMPT` (customize assistant behavior)
     - `MAX_TOKENS_LIMIT` (validation upper bound for `max_tokens`)

2) Start the API
   `uvicorn model_server.server:app --host 127.0.0.1 --port 8000`

3) Call the API (non-stream)
   `curl -s http://127.0.0.1:8000/v1/chat/completions -H "content-type: application/json" -d '{"model":"gemma","stream":false,"messages":[{"role":"user","content":"Hello"}]}'`

4) Streaming example (Python + httpx)
   ```python
   import httpx
   with httpx.Client(timeout=None) as client:
       with client.stream("POST", "http://127.0.0.1:8000/v1/chat/completions", json={
           "model": "gemma",
           "stream": True,
           "messages": [{"role": "user", "content": "Hello"}],
       }) as resp:
           for line in resp.iter_lines():
               if line:
                   print(line)
   ```

Troubleshooting
- If import errors mention `llama-cpp-python`, ensure the wheel matches your OS/GPU.
- For long runs, reduce `max_tokens` or use fewer `N_GPU_LAYERS`.

Run via Docker
- See `docs/howto-docker.md` for building/running images and Compose usage.
