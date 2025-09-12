from fastapi.testclient import TestClient
import types

from model_server import server


class FakeLlama:
    def create_chat_completion(self, messages, temperature, max_tokens, stream, **kwargs):
        assert stream is False
        # Return a minimal OpenAI-style completion
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gemma",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
        }


def test_chat_completion_non_streaming(monkeypatch):
    # Patch the server to use our fake llama
    monkeypatch.setattr(server, "get_llama", lambda: FakeLlama())

    client = TestClient(server.app)
    payload = {
        "model": "gemma",
        "stream": False,
        "messages": [
            {"role": "user", "content": "Hi"},
        ],
    }
    resp = client.post("/v1/chat/completions", json=payload)
    if resp.status_code != 200:
        print("Body:", resp.text)
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "Hello!"
