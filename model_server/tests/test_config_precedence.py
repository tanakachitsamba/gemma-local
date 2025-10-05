import os
import importlib


def test_config_reads_from_yaml_file(tmp_path, monkeypatch):
    cfgfile = tmp_path / "config.yaml"
    cfgfile.write_text(
        """
MODEL_PATH: ./models/demo.gguf
N_CTX: 2048
N_THREADS: 1
MAX_TOKENS_LIMIT: 9000
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CONFIG_PATH", str(cfgfile))

    # Reload module to pick up new config
    import model_server.config as cfg

    importlib.reload(cfg)

    assert cfg.N_CTX == 2048
    assert cfg.N_THREADS == 1
    assert cfg.MODEL_PATH.endswith("demo.gguf")
    assert cfg.MAX_TOKENS_LIMIT == 9000


def test_env_overrides_yaml(tmp_path, monkeypatch):
    cfgfile = tmp_path / "config.yaml"
    cfgfile.write_text("N_CTX: 2048\n", encoding="utf-8")
    monkeypatch.setenv("CONFIG_PATH", str(cfgfile))
    monkeypatch.setenv("N_CTX", "7777")

    import model_server.config as cfg

    importlib.reload(cfg)

    assert cfg.N_CTX == 7777

