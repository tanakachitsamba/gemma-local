from fastapi.testclient import TestClient

from model_server import server


class FakeSem:
    def acquire(self, timeout=None):
        return False

    def release(self):
        pass


def test_rate_limit_nonstreaming(monkeypatch):
    monkeypatch.setattr(server, "_req_semaphore", FakeSem())
    client = TestClient(server.app)
    payload = {"model": "gemma", "stream": False, "messages": [{"role": "user", "content": "hi"}]}
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 429


def test_rate_limit_streaming(monkeypatch):
    monkeypatch.setattr(server, "_req_semaphore", FakeSem())
    client = TestClient(server.app)
    payload = {"model": "gemma", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
    with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
        assert resp.status_code == 429

