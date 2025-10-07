Lightweight training (LoRA) scaffold

This directory provides a minimal, practical entry to fine-tune a causal LM
with PEFT/LoRA using Hugging Face Transformers on an instruction-style JSONL.

Why this approach?
- You can train on full-precision or 4-bit (QLoRA) depending on GPU. The
  script automatically disables 4-bit when no CUDA device is available to
  avoid confusing runtime failures.
- It’s easy to export a merged adapter back to a HF model and then convert to
  GGUF for llama.cpp inference.

Files
- train_lora.py — config-driven training script (YAML)
- requirements.txt — training-time dependencies

Quickstart
1) Create and edit a config from the example:
   cp ../configs/train_lora.example.yaml ./train_lora.yaml
2) Install deps in a fresh venv with CUDA-capable PyTorch.
   pip install -r requirements.txt
3) Run training:
   python train_lora.py --config ./train_lora.yaml

Dataset format (JSONL)
- Each line is an object with either:
  {"instruction": "...", "input": "...", "output": "..."}
  or a chat-style schema:
  {"messages": [{"role": "user", "content": "..."}, ...]}

Export and convert
- After training, merge adapters into the base model and save a HF repo.
- Use llama.cpp’s convert.py + quantize to create a GGUF for local inference.

Notes
- Start tiny: a few hundred samples, short sequences, small ranks.
- Keep logs (CSV/JSON) and samples for quick regression checks.
