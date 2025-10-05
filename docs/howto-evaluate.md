How to Evaluate Base vs. Fine‑Tuned

1) Start the server on the target model
- Base or fine‑tuned GGUF.

2) Prepare a small eval set
- JSONL in either instruction or chat format.

3) Run the HTTP evaluator
- `python scripts/evaluate.py --dataset path/to/eval.jsonl --base_url http://127.0.0.1:8000 --max_samples 50`
- Outputs `eval_outputs.jsonl` with prompts and generations.

4) Compare
- Run the evaluator against base and fine‑tuned servers; diff the outputs.
- For quick scoring, add rules or use a light metric (e.g., exact match for structured tasks).

Notes
- Keep eval sets small and representative.
- For reproducibility, fix `seed` in requests (server supports `seed`).
