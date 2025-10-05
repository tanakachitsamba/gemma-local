How to Install (Windows/macOS/Linux)

Prereqs
- Python 3.10+ (3.11 recommended)
- A GGUF model file (e.g., Gemma or compatible)

Steps
1) Create a virtual environment
   - Windows: `python -m venv .venv && .\.venv\Scripts\activate`
   - macOS/Linux: `python -m venv .venv && source .venv/bin/activate`

2) Install server dependencies
   `pip install -r model_server/requirements.txt`

3) Optional: dev/test deps
   `pip install -r model_server/requirements-dev.txt`

4) Place your model
   Put your `.gguf` under `./models/` and set `MODEL_PATH` accordingly.

Notes
- If you need GPU acceleration, install a CUDA-capable `llama-cpp-python` wheel.
- Keep the environment minimal; use a dedicated venv per project.
