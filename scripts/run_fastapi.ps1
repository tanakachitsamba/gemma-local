param(
  [string]$ModelPath = "./models/gemma-2-2b-it.Q4_K_M.gguf",
  [int]$NCtx = 4096,
  [int]$NThreads = [Environment]::ProcessorCount,
  [int]$NGpuLayers = 0,
  [string]$Host = "127.0.0.1",
  [int]$Port = 8000
)

Set-Location -LiteralPath (Join-Path $PSScriptRoot "..")

if (!(Test-Path ".venv")) {
  python -m venv .venv
}
& .\.venv\Scripts\python -m pip install --upgrade pip | Out-Null
& .\.venv\Scripts\python -m pip install -r model_server/requirements.txt | Out-Null

$env:MODEL_PATH = $ModelPath
$env:N_CTX = "$NCtx"
$env:N_THREADS = "$NThreads"
$env:N_GPU_LAYERS = "$NGpuLayers"
$env:HOST = $Host
$env:PORT = "$Port"

& .\.venv\Scripts\python -m uvicorn model_server.server:app --host $Host --port $Port

