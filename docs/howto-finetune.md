How to Fine‑Tune with LoRA (PEFT)

Goal
- Start with a base HF model, train a small LoRA adapter on your dataset, and export artifacts for serving.

1) Install training deps
- `pip install -r training/requirements.txt`
- Ensure a CUDA PyTorch build is installed if you train on GPU.

2) Prepare a config
- Copy `configs/train_lora.example.yaml` and edit fields:
  - `base_model`: HF repo (e.g., `meta-llama/Llama-2-7b-hf`)
  - `dataset_path`: JSONL file (instruction or chat schema)
  - `output_dir`: where to save adapters and logs
  - `use_4bit`: enable QLoRA on consumer GPUs

3) Train
- `python training/train_lora.py --config configs/your_config.yaml`
- Artifacts: `output_dir/lora_adapter/` and `train_summary.json`

4) Export & convert (outline)
- Merge adapters with the base model (via PEFT/Transformers) and save HF format.
- Convert to GGUF and quantize using llama.cpp tools for local serving.

Tips
- Keep sequence lengths modest at first (`max_length: 512`).
- Start with low ranks (`lora_r: 8`) and short runs to validate setup.
