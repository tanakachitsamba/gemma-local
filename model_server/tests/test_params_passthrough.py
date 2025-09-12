from fastapi.testclient import TestClient

from model_server import server


class CaptureArgs:
    def __init__(self):
        self.kwargs = None

    def create_chat_completion(self, messages, temperature, max_tokens, stream, **kwargs):
        self.kwargs = kwargs
        # minimal response
        return {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "x",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
            ],
        }


def test_passes_stop_and_seed(monkeypatch):
    cap = CaptureArgs()
    monkeypatch.setattr(server, "get_llama", lambda: cap)

    client = TestClient(server.app)
    payload = {
        "model": "gemma",
        "stream": False,
        "stop": ["\nUser:"],
        "seed": 1234,
        "messages": [
            {"role": "user", "content": "Hi"},
        ],
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert cap.kwargs is not None
    assert cap.kwargs.get("stop") == ["\nUser:"]
    assert cap.kwargs.get("seed") == 1234

