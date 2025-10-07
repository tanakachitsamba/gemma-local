from fastapi.testclient import TestClient

from model_server.server import app


def test_healthz_ok():
    client = TestClient(app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("ok") is True
    assert "model_path" in body
    assert "hardware" in body
    hardware = body["hardware"]
    assert "cpu_count" in hardware
    assert "gpu_available" in hardware

