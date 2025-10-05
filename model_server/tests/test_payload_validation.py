from fastapi.testclient import TestClient

from model_server import server


def test_empty_messages_rejected():
    client = TestClient(server.app)
    payload = {"model": "gemma", "stream": False, "messages": []}
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 422


def test_invalid_stop_type_rejected():
    client = TestClient(server.app)
    payload = {"model": "gemma", "stream": False, "messages": [{"role": "user", "content": "hi"}], "stop": [123]}
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 422


def test_excessive_max_tokens_rejected():
    client = TestClient(server.app)
    payload = {
        "model": "gemma",
        "stream": False,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 999999,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 422

