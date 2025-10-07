import os
from typing import Any, Dict, Tuple

try:  # optional in tests/CI
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


_CFG: Dict[str, Any] = {}


def detect_cpu_count() -> int:
    """Return the number of logical CPUs available to the process."""

    try:
        count = os.cpu_count()
    except Exception:  # pragma: no cover - extremely defensive
        count = None
    if not count or count < 1:
        return 1
    return count


def recommend_thread_count(cpu_count: int) -> int:
    """Choose a conservative default thread count for llama.cpp."""

    if cpu_count <= 1:
        return 1
    if cpu_count <= 4:
        return cpu_count
    # Leave a couple of cores for the rest of the system when many are present.
    return max(1, cpu_count - 1)


def _detect_gpu_with_torch() -> Tuple[int, str]:
    try:  # Optional dependency; avoid hard import failures.
        import torch  # type: ignore
    except (ImportError, ModuleNotFoundError):
        return 0, "unavailable"

    try:
        if torch.cuda.is_available():
            count = torch.cuda.device_count() or 0
            if count:
                return count, "torch.cuda"
    except RuntimeError:  # pragma: no cover - torch backend errors
        return 0, "torch-error"
    return 0, "torch"


def _detect_gpu_from_env() -> Tuple[int, str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return 0, "env"
    if visible.strip() in {"", "-1", "none"}:
        return 0, "env"
    devices = [d.strip() for d in visible.split(",") if d.strip()]
    return (len(devices), "env") if devices else (0, "env")


def detect_gpu_count() -> Tuple[int, str]:
    """Detect the number of CUDA-capable GPUs available."""

    count, source = _detect_gpu_with_torch()
    if count:
        return count, source
    env_count, env_source = _detect_gpu_from_env()
    if env_count:
        return env_count, env_source
    return 0, source


def recommend_gpu_layers(gpu_count: int) -> int:
    """Select a default number of GPU layers based on detected hardware."""

    if gpu_count <= 0:
        return 0
    # Offload roughly 32 layers per detected GPU, capped for safety.
    return max(0, min(80, 32 * gpu_count))


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

_CPU_COUNT = detect_cpu_count()
_GPU_COUNT, _GPU_SOURCE = detect_gpu_count()

DEFAULT_THREADS = recommend_thread_count(_CPU_COUNT)
DEFAULT_GPU_LAYERS = recommend_gpu_layers(_GPU_COUNT)


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
N_THREADS = getenv_int("N_THREADS", DEFAULT_THREADS)
N_GPU_LAYERS = getenv_int("N_GPU_LAYERS", DEFAULT_GPU_LAYERS)
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

GPU_AVAILABLE = _GPU_COUNT > 0

HARDWARE = {
    "cpu_count": _CPU_COUNT,
    "recommended_threads": DEFAULT_THREADS,
    "configured_threads": N_THREADS,
    "gpu_available": GPU_AVAILABLE,
    "gpu_count": _GPU_COUNT,
    "gpu_detection_source": _GPU_SOURCE,
    "recommended_gpu_layers": DEFAULT_GPU_LAYERS,
    "configured_gpu_layers": N_GPU_LAYERS,
}
