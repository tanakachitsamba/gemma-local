import pytest
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


def test_get_llama_errors_when_gpu_layers_requested_without_gpu(monkeypatch):
    class DummyLlama:
        def __init__(self, *args, **kwargs):
            raise AssertionError("should not construct when GPU unavailable")

    monkeypatch.setattr(server, "_llama_instance", None)
    monkeypatch.setattr(server, "Llama", DummyLlama)
    monkeypatch.setattr(server, "_import_err", None)
    monkeypatch.setattr(server.config, "N_GPU_LAYERS", 2)
    monkeypatch.setattr(server.config, "GPU_AVAILABLE", False)
    monkeypatch.setitem(server.config.HARDWARE, "gpu_detection_source", "test")
    monkeypatch.setitem(server.config.HARDWARE, "configured_gpu_layers", 2)
    monkeypatch.setitem(server.config.HARDWARE, "gpu_available", False)

    with TestClient(server.app):
        pass  # ensure FastAPI app doesn't hold onto previous state

    with pytest.raises(RuntimeError) as excinfo:
        server.get_llama()

    assert "no GPU was detected" in str(excinfo.value)

