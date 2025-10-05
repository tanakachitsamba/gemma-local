from fastapi.testclient import TestClient
from hypothesis import given, strategies as st

from model_server.server import app, sse_event


def test_metrics_endpoint_exists():
    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    text = resp.text
    assert "requests_total" in text
    assert "errors_total" in text


@given(st.dictionaries(keys=st.text(min_size=0, max_size=5), values=st.text(max_size=20)))
def test_sse_event_property_always_prefixes_data_and_has_double_newline(d):
    out = sse_event(d)
    assert out.startswith(b"data: ")
    assert out.endswith(b"\n\n")
