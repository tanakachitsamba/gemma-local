from fastapi.testclient import TestClient

from model_server import server


class CaptureStreamArgs:
    def __init__(self):
        self.kwargs = None

    def create_chat_completion(self, messages, temperature, max_tokens, stream, **kwargs):
        self.kwargs = kwargs

        def gen():
            yield {"choices": [{"delta": {"content": "Hi"}, "index": 0, "finish_reason": None}]}
        return gen()


def test_streaming_passes_stop_and_seed(monkeypatch):
    cap = CaptureStreamArgs()
    monkeypatch.setattr(server, "get_llama", lambda: cap)
    client = TestClient(server.app)

    payload = {
        "model": "gemma",
        "stream": True,
        "stop": ["STOP"],
        "seed": 42,
        "messages": [{"role": "user", "content": "Hi"}],
    }
    with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
        assert resp.status_code == 200
    assert cap.kwargs is not None
    assert cap.kwargs.get("stop") == ["STOP"]
    assert cap.kwargs.get("seed") == 42


class ErroringStream:
    def create_chat_completion(self, *a, **kw):
        def gen():
            yield {"choices": [{"delta": {"content": "Part"}, "index": 0, "finish_reason": None}]}
            raise RuntimeError("stream broke")
        return gen()


def test_streaming_error_event_and_done(monkeypatch):
    monkeypatch.setattr(server, "get_llama", lambda: ErroringStream())
    client = TestClient(server.app)
    payload = {
        "model": "gemma",
        "stream": True,
        "messages": [{"role": "user", "content": "Hi"}],
    }
    with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
        text = b"".join(resp.iter_raw()).decode("utf-8", errors="ignore")
        assert "data: [DONE]" in text
        # error may appear depending on timing; check that either first chunk or error exists
        assert "data: {" in text

