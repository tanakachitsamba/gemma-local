from typing import List

import json
from fastapi.testclient import TestClient
from hypothesis import given, strategies as st

from model_server import server


class GenStream:
    def __init__(self, parts: List[str]):
        self.parts = parts

    def create_chat_completion(self, messages, temperature, max_tokens, stream, **kwargs):
        assert stream is True

        def gen():
            for p in self.parts:
                yield {
                    "choices": [
                        {"delta": {"content": p}, "index": 0, "finish_reason": None}
                    ]
                }

        return gen()


def _extract_deltas(raw: bytes) -> List[str]:
    # Split by double newline (SSE events) and parse JSON lines
    text = raw.decode("utf-8", errors="ignore")
    chunks = []
    for block in text.split("\n\n"):
        if not block.startswith("data: "):
            continue
        body = block[len("data: ") :].strip()
        if body == "[DONE]":
            continue
        try:
            j = json.loads(body)
        except Exception:
            continue
        try:
            chunks.append(j["choices"][0]["delta"].get("content", ""))
        except Exception:
            chunks.append("")
    return chunks


@given(st.lists(st.text(min_size=0, max_size=5), min_size=1, max_size=6))
def test_stream_chunk_order_preserved(parts):
    monkey = server
    client = TestClient(server.app)

    # Patch llama to yield the generated parts in order
    monkey.get_llama = lambda: GenStream(parts)  # type: ignore

    payload = {
        "model": "gemma",
        "stream": True,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 32,
    }
    with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
        assert resp.status_code == 200
        data = b"".join(resp.iter_raw())

    observed = _extract_deltas(data)
    assert observed == parts

