$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvironmentDir = Join-Path $ProjectRoot ".venv"
$EngineDir = Join-Path $ProjectRoot "trainer-engine"
$EngineRevision = "8934cfbbb4b9bcfa8071ce209129f0c5eb5df2e6"

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python Launcher was not found. Install 64-bit Python 3.10, then run setup.ps1 again."
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git was not found. Install Git for Windows, then run setup.ps1 again."
}

if (-not (Test-Path (Join-Path $EnvironmentDir "Scripts\python.exe"))) {
    Write-Host "Creating the Python 3.10 environment..."
    & py -3.10 -m venv $EnvironmentDir
    if ($LASTEXITCODE -ne 0) { throw "Python 3.10 is required to create the environment." }
}

$Python = Join-Path $EnvironmentDir "Scripts\python.exe"
if (-not (Test-Path (Join-Path $EngineDir ".git"))) {
    Write-Host "Downloading the tested Musubi Tuner source..."
    & git clone https://github.com/kohya-ss/musubi-tuner.git $EngineDir
    if ($LASTEXITCODE -ne 0) { throw "Musubi Tuner could not be downloaded." }
}
& git -C $EngineDir fetch origin $EngineRevision --depth 1
if ($LASTEXITCODE -ne 0) { throw "The tested Musubi Tuner revision could not be downloaded." }
& git -C $EngineDir checkout --detach $EngineRevision
if ($LASTEXITCODE -ne 0) { throw "The tested Musubi Tuner revision could not be selected." }

Write-Host "Installing the CUDA 12.4 training environment..."
& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip could not be updated." }
& $Python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) { throw "PyTorch could not be installed." }
& $Python -m pip install --editable $EngineDir
if ($LASTEXITCODE -ne 0) { throw "Musubi Tuner dependencies could not be installed." }
& $Python -m pip install psutil tensorboard
if ($LASTEXITCODE -ne 0) { throw "Desktop app dependencies could not be installed." }

& $Python (Join-Path $ProjectRoot "app.py") --initialize
if ($LASTEXITCODE -ne 0) { throw "The local workspace could not be initialized." }
& $Python (Join-Path $ProjectRoot "tools\verify_setup.py")
if ($LASTEXITCODE -ne 0) { throw "The setup check did not pass." }
Write-Host "Setup is complete. Download the model files next, then launch the app."
Write-Host "  powershell -ExecutionPolicy Bypass -File .\scripts\download_models.ps1"
