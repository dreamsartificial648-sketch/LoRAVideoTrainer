. "$PSScriptRoot\common.ps1"

New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null

Write-Host "Downloading Wan 2.1 T2V 1.3B DiT and VAE..."
& $Python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Comfy-Org/Wan_2.1_ComfyUI_repackaged', 'split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors', local_dir=r'$ModelDir'); hf_hub_download('Comfy-Org/Wan_2.1_ComfyUI_repackaged', 'split_files/vae/wan_2.1_vae.safetensors', local_dir=r'$ModelDir')"

Write-Host "Downloading the Wan text encoder..."
& $Python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Wan-AI/Wan2.1-I2V-14B-720P', 'models_t5_umt5-xxl-enc-bf16.pth', local_dir=r'$ModelDir')"

$DownloadedDit = Join-Path $ModelDir "split_files\diffusion_models\wan2.1_t2v_1.3B_bf16.safetensors"
$DownloadedVae = Join-Path $ModelDir "split_files\vae\wan_2.1_vae.safetensors"
Move-Item -LiteralPath $DownloadedDit -Destination $DitPath -Force
Move-Item -LiteralPath $DownloadedVae -Destination $VaePath -Force

Require-File $DitPath
Require-File $VaePath
Require-File $T5Path
Write-Host "Model files are ready in $ModelDir"
