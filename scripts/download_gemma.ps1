<#
.SYNOPSIS
  Helper notes for obtaining Gemma GGUF models.

.DESCRIPTION
  This script does not auto-download by default to avoid license friction.
  Steps:
    1) Visit a GGUF repository you’re allowed to use, e.g.:
       - https://huggingface.co/bartowski/gemma-2-2b-it-GGUF
       - https://huggingface.co/TheBloke/gemma-2b-it-GGUF
    2) Choose a quant like Q4_K_M and download the .gguf file.
    3) Place it into ../models/ and set MODEL_PATH accordingly.

  If you have `huggingface-cli` configured and license accepted, you may uncomment
  an example line below and run it.
#>

param()

$modelsDir = Join-Path $PSScriptRoot "..\models"
if (!(Test-Path $modelsDir)) {
  New-Item -ItemType Directory -Path $modelsDir | Out-Null
}

Write-Host "Models directory is: $modelsDir"
Write-Host "Please manually download a Gemma GGUF (e.g., gemma-2-2b-it.Q4_K_M.gguf) into this folder."

# Example (requires huggingface-cli logged in and access):
# huggingface-cli download bartowski/gemma-2-2b-it-GGUF --include "*Q4_K_M*.gguf" --local-dir "$modelsDir"

