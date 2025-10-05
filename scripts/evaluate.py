import argparse
import json
from pathlib import Path
from typing import List, Dict

import httpx


def load_dataset(path: str) -> List[Dict]:
    items: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def as_chat(example: Dict) -> List[Dict[str, str]]:
    if "messages" in example:
        return example["messages"]
    instr = example.get("instruction", "")
    inp = example.get("input", "")
    user = instr if not inp else f"{instr}\n{inp}"
    return [{"role": "user", "content": user}]


def main() -> None:
    ap = argparse.ArgumentParser(description="Quick HTTP evaluator (non-stream)")
    ap.add_argument("--dataset", required=True, help="path to jsonl dataset")
    ap.add_argument("--base_url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="gemma")
    ap.add_argument("--max_samples", type=int, default=50)
    args = ap.parse_args()

    data = load_dataset(args.dataset)[: args.max_samples]
    url = args.base_url.rstrip("/") + "/v1/chat/completions"

    outputs = []
    with httpx.Client(timeout=60) as client:
        for i, ex in enumerate(data):
            payload = {
                "model": args.model,
                "stream": False,
                "messages": as_chat(ex),
                "max_tokens": 128,
            }
            r = client.post(url, json=payload)
            r.raise_for_status()
            j = r.json()
            text = j.get("choices", [{}])[0].get("message", {}).get("content", "")
            outputs.append({"index": i, "prompt": payload["messages"], "output": text})

    out_path = Path("./eval_outputs.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for row in outputs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(outputs)} results to {out_path}")


if __name__ == "__main__":
    main()

