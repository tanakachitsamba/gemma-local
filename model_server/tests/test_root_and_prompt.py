from model_server.server import app, _format_prompt
from fastapi.testclient import TestClient


def test_root_info_present():
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("name")
    assert "/healthz" in body.get("endpoints", [])


def test_format_prompt_includes_system_and_assistant_trailer():
    messages = [
        {"role": "user", "content": "Ping"},
        {"role": "assistant", "content": "Pong"},
    ]
    out = _format_prompt(messages)
    assert "User: Ping" in out
    assert "Assistant: Pong" in out
    assert out.rstrip().endswith("Assistant:")


def test_format_prompt_handles_unknown_roles():
    messages = [
        {"role": "critic", "content": "too vague"},
    ]
    out = _format_prompt(messages)
    # Role should be capitalized and preserved
    assert "Critic: too vague" in out

