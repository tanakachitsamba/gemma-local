from fastapi.testclient import TestClient

from model_server import server


class FakeLlamaCompletionOnly:
    def create_chat_completion(self, *a, **kw):
        raise RuntimeError("no chat format")

    def create_completion(self, prompt, temperature, max_tokens, stream, stop=None, seed=None):
        assert stream is False
        return {"choices": [{"text": "Hello from completion", "finish_reason": "stop"}]}


def test_nonstreaming_fallback_to_completion(monkeypatch):
    monkeypatch.setattr(server, "get_llama", lambda: FakeLlamaCompletionOnly())
    client = TestClient(server.app)

    payload = {
        "model": "gpt-neox-20b",
        "stream": False,
        "messages": [{"role": "user", "content": "Hi"}],
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert "Hello" in body["choices"][0]["message"]["content"]

