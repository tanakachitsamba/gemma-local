from fastapi.testclient import TestClient

from model_server import server


class FakeLlamaCompletionOnly:
    def create_chat_completion(self, *a, **kw):
        raise RuntimeError("no chat format")

    def create_completion(self, prompt, temperature, max_tokens, stream, stop=None, seed=None):
        assert stream is True
        # Stream two text chunks and then finish
        yield {"choices": [{"text": "Hel"}]}
        yield {"choices": [{"text": "lo"}]}


def test_streaming_fallback_to_completion(monkeypatch):
    monkeypatch.setattr(server, "get_llama", lambda: FakeLlamaCompletionOnly())
    client = TestClient(server.app)

    payload = {
        "model": "gpt-neox-20b",
        "stream": True,
        "messages": [
            {"role": "user", "content": "Hi"},
        ],
    }
    with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
        assert resp.status_code == 200
        data = b"".join(resp.iter_raw())
        text = data.decode("utf-8", errors="ignore")
        assert "data: {" in text
        assert "[DONE]" in text
