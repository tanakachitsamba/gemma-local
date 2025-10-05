from fastapi.testclient import TestClient

from model_server import server
from model_server import config


class CaptureMessages:
    def __init__(self):
        self.messages = None

    def create_chat_completion(self, messages, temperature, max_tokens, stream, **kwargs):
        self.messages = messages
        return {
            "id": "ok",
            "object": "chat.completion",
            "created": 0,
            "model": "x",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
            ],
        }


def test_injects_default_system_when_missing(monkeypatch):
    cap = CaptureMessages()
    monkeypatch.setattr(server, "get_llama", lambda: cap)
    client = TestClient(server.app)

    payload = {
        "model": "gemma",
        "stream": False,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert cap.messages is not None
    assert cap.messages[0]["role"] == "system"
    assert cap.messages[0]["content"] == config.SYSTEM_PROMPT


def test_respects_existing_system_prompt(monkeypatch):
    cap = CaptureMessages()
    monkeypatch.setattr(server, "get_llama", lambda: cap)
    client = TestClient(server.app)

    payload = {
        "model": "gemma",
        "stream": False,
        "messages": [
            {"role": "system", "content": "Custom system"},
            {"role": "user", "content": "Hi"},
        ],
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert cap.messages is not None
    assert cap.messages[0]["role"] == "system"
    assert cap.messages[0]["content"] == "Custom system"

