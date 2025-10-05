import os
from typing import Any, Dict

try:  # optional in tests/CI
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


_CFG: Dict[str, Any] = {}


def _load_yaml_config() -> Dict[str, Any]:
    path = os.getenv("CONFIG_PATH", "./config.yaml")
    if not os.path.isfile(path):
        return {}
    if yaml is None:  # pragma: no cover - exercised via env path in tests
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:  # pragma: no cover - defensive parse fallback
        return {}


_CFG = _load_yaml_config()


def getenv_str(name: str, default: str) -> str:
    # Precedence: env var > config.yaml > default
    if name in os.environ:
        return os.environ[name]
    if name in _CFG:
        val = _CFG[name]
        return str(val)
    return default


def getenv_int(name: str, default: int) -> int:
    try:
        if name in os.environ:
            return int(os.environ[name])
        if name in _CFG:
            return int(_CFG[name])
        return int(default)
    except ValueError:
        return default


MODEL_PATH = getenv_str("MODEL_PATH", "./models/gemma-2-2b-it.Q4_K_M.gguf")
N_CTX = getenv_int("N_CTX", 4096)
N_THREADS = getenv_int("N_THREADS", os.cpu_count() or 4)
N_GPU_LAYERS = getenv_int("N_GPU_LAYERS", 0)
HOST = getenv_str("HOST", "127.0.0.1")
PORT = getenv_int("PORT", 8000)
TEMPERATURE_DEFAULT = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS_DEFAULT = getenv_int("MAX_TOKENS", 512)
MAX_TOKENS_LIMIT = getenv_int("MAX_TOKENS_LIMIT", 8192)
SYSTEM_PROMPT = getenv_str(
    "SYSTEM_PROMPT",
    "You are a helpful assistant running locally.",
)

# Ops knobs
MAX_CONCURRENT_REQUESTS = getenv_int("MAX_CONCURRENT_REQUESTS", 32)
ACQUIRE_TIMEOUT_MS = getenv_int("ACQUIRE_TIMEOUT_MS", 10000)
STREAMING_TIMEOUT_S = getenv_int("STREAMING_TIMEOUT_S", 120)
