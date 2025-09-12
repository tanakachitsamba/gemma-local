from fastapi.testclient import TestClient

from model_server import server


class FakeLlamaStream:
    def create_chat_completion(self, messages, temperature, max_tokens, stream, **kwargs):
        assert stream is True
        # Yield OpenAI-like chunks
        yield {
            "choices": [
                {
                    "delta": {"role": "assistant", "content": "Hel"},
                    "index": 0,
                    "finish_reason": None,
                }
            ]
        }
        yield {
            "choices": [
                {
                    "delta": {"content": "lo"},
                    "index": 0,
                    "finish_reason": None,
                }
            ]
        }


def test_chat_completion_streaming(monkeypatch):
    monkeypatch.setattr(server, "get_llama", lambda: FakeLlamaStream())
    client = TestClient(server.app)

    payload = {
        "model": "gemma",
        "stream": True,
        "messages": [
            {"role": "user", "content": "Hi"},
        ],
    }

    with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
        assert resp.status_code == 200
        assert resp.headers.get("content-type", "").startswith("text/event-stream")
        chunks = b"".join(resp.iter_raw())
        # Expect at least two data lines and a [DONE]
        text = chunks.decode("utf-8", errors="ignore")
        assert "data: {" in text
        assert "[DONE]" in text
