$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Accelerate = Join-Path $ProjectRoot ".venv\Scripts\accelerate.exe"
$EngineRoot = Join-Path $ProjectRoot "trainer-engine"
$DatasetConfig = Join-Path $ProjectRoot "dataset_config.toml"
$ModelDir = Join-Path $ProjectRoot "models"
$DitPath = Join-Path $ModelDir "wan2.1_t2v_1.3B_bf16.safetensors"
$VaePath = Join-Path $ModelDir "wan_2.1_vae.safetensors"
$T5Path = Join-Path $ModelDir "models_t5_umt5-xxl-enc-bf16.pth"

function Require-File([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required file is missing: $Path"
    }
}

Require-File $Python
Require-File $DatasetConfig
