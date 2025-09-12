import os


def getenv_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def getenv_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
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
SYSTEM_PROMPT = getenv_str(
    "SYSTEM_PROMPT",
    "You are a helpful assistant running locally.",
)
