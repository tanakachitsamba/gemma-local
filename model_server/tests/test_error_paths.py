from fastapi.testclient import TestClient

from model_server import server


def test_chat_returns_500_when_model_unavailable(monkeypatch):
    def boom():
        raise RuntimeError("no llama")

    monkeypatch.setattr(server, "get_llama", boom)
    client = TestClient(server.app)
    payload = {
        "model": "gemma",
        "stream": False,
        "messages": [
            {"role": "user", "content": "Hi"},
        ],
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 500
    assert "no llama" in resp.text

