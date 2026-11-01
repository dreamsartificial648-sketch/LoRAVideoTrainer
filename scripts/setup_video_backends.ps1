param([ValidateSet('all', 'modelscope_video', 'ltx_video', 'ltx25')][string]$Backend = 'all', [string]$Ltx25Store = '')
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Backends = if ($Backend -eq 'all') { @('modelscope_video', 'ltx_video') } else { @($Backend) }
foreach ($SelectedBackend in $Backends) {
    if ($SelectedBackend -eq 'ltx25') {
        $SetupArgs = @((Join-Path $ProjectRoot 'tools\ltx25_runtime\setup_runtime.py'))
        if ($Ltx25Store) { $SetupArgs += @('--store', $Ltx25Store) }
        & py -3.10 @SetupArgs
        if ($LASTEXITCODE -ne 0) { throw 'LTX-2.5 setup failed. Review the hardware/dependency error above.' }
        continue
    }
    $EnvironmentDir = Join-Path $ProjectRoot "runtimes\$SelectedBackend\.venv"
    $BackendPython = Join-Path $EnvironmentDir 'Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $BackendPython)) {
        & py -3.10 -m venv $EnvironmentDir
        if ($LASTEXITCODE -ne 0) { throw 'Install 64-bit Python 3.10 with the Python Launcher.' }
    }
    & $BackendPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw 'pip update failed.' }
    & $BackendPython -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
    if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
    & $BackendPython -m pip install -r (Join-Path $ProjectRoot 'requirements-video.txt') av==14.2.0
    if ($LASTEXITCODE -ne 0) { throw 'Video backend dependency installation failed.' }
    & $BackendPython (Join-Path $ProjectRoot 'tools\video_backend.py') --backend $SelectedBackend --mode check
    if ($LASTEXITCODE -ne 0) { throw "$SelectedBackend setup check failed." }
}
Write-Host 'Video backends installed. Select the model in the app, then use Download Model Files.'
