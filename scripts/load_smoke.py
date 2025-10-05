import argparse
import asyncio
import time
from typing import List

import httpx


async def worker(name: int, url: str, payload: dict, reps: int) -> float:
    latencies: List[float] = []
    async with httpx.AsyncClient(timeout=30) as client:
        for _ in range(reps):
            t0 = time.perf_counter()
            r = await client.post(url, json=payload)
            r.raise_for_status()
            _ = r.json()
            latencies.append(time.perf_counter() - t0)
    return sum(latencies) / max(1, len(latencies))


async def main() -> None:
    ap = argparse.ArgumentParser(description="Quick load smoke test")
    ap.add_argument("--base_url", default="http://127.0.0.1:8000")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": "gemma",
        "stream": False,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 16,
    }

    tasks = [worker(i, url, payload, args.reps) for i in range(args.concurrency)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    vals = [r for r in results if isinstance(r, (int, float))]
    if not vals:
        print("No successful requests.")
        return
    avg = sum(vals) / len(vals)
    print(f"avg_latency={avg:.3f}s across {len(vals)*args.reps} requests at c={args.concurrency}")


if __name__ == "__main__":
    asyncio.run(main())

