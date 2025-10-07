import importlib
import sys
from types import SimpleNamespace


def _reload_config(monkeypatch):
    """Reload config after clearing common environment overrides."""

    for key in ("CONFIG_PATH", "N_THREADS", "N_GPU_LAYERS", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(key, raising=False)
    import model_server.config as cfg

    return importlib.reload(cfg)


def test_cpu_detection_and_recommendation(monkeypatch):
    cfg = _reload_config(monkeypatch)

    # When cpu_count() is unavailable we fall back to 1.
    monkeypatch.setattr(cfg.os, "cpu_count", lambda: None)
    assert cfg.detect_cpu_count() == 1

    # Small machines use all available threads, larger ones leave headroom.
    assert cfg.recommend_thread_count(1) == 1
    assert cfg.recommend_thread_count(4) == 4

    monkeypatch.setattr(cfg.os, "cpu_count", lambda: 8)
    assert cfg.recommend_thread_count(cfg.detect_cpu_count()) == 7


def test_gpu_detection_via_environment(monkeypatch):
    cfg = _reload_config(monkeypatch)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    cfg = importlib.reload(cfg)

    assert cfg.HARDWARE["gpu_available"] is True
    assert cfg.HARDWARE["gpu_count"] == 2
    assert cfg.DEFAULT_GPU_LAYERS == 64

    # Reset to defaults for later tests.
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    importlib.reload(cfg)


def test_gpu_detection_via_fake_torch(monkeypatch):
    cfg = _reload_config(monkeypatch)

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 3,
        )
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    cfg = importlib.reload(cfg)

    assert cfg.HARDWARE["gpu_detection_source"] == "torch.cuda"
    assert cfg.HARDWARE["gpu_count"] == 3
    assert cfg.DEFAULT_GPU_LAYERS == 80  # capped at safety limit

    # Clean up for other tests: remove fake torch and reload defaults.
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    importlib.reload(cfg)
