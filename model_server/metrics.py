from __future__ import annotations

import threading
import time
from typing import Dict


_lock = threading.Lock()
_counters: Dict[str, float] = {
    "requests_total": 0.0,
    "streaming_requests_total": 0.0,
    "errors_total": 0.0,
    "request_duration_seconds_sum": 0.0,
    "request_duration_seconds_count": 0.0,
}
_last_request_ts: float | None = None


def record_request(start: float, streaming: bool) -> None:
    global _last_request_ts
    dur = time.time() - start
    with _lock:
        _counters["requests_total"] += 1.0
        _counters["request_duration_seconds_sum"] += dur
        _counters["request_duration_seconds_count"] += 1.0
        if streaming:
            _counters["streaming_requests_total"] += 1.0
        _last_request_ts = time.time()


def record_error() -> None:
    with _lock:
        _counters["errors_total"] += 1.0


def prometheus_text() -> bytes:
    lines = [
        "# HELP requests_total Total HTTP requests",
        "# TYPE requests_total counter",
        f"requests_total {_counters['requests_total']}",
        "# HELP streaming_requests_total Streaming HTTP requests",
        "# TYPE streaming_requests_total counter",
        f"streaming_requests_total {_counters['streaming_requests_total']}",
        "# HELP errors_total Total errors",
        "# TYPE errors_total counter",
        f"errors_total {_counters['errors_total']}",
        "# HELP request_duration_seconds Request durations",
        "# TYPE request_duration_seconds summary",
        f"request_duration_seconds_sum {_counters['request_duration_seconds_sum']}",
        f"request_duration_seconds_count {_counters['request_duration_seconds_count']}",
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")

