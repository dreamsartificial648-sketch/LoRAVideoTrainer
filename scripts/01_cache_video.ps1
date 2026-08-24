. "$PSScriptRoot\common.ps1"
Require-File $VaePath

& $Python "$EngineRoot\src\musubi_tuner\wan_cache_latents.py" `
    --dataset_config $DatasetConfig `
    --vae $VaePath `
    --vae_cache_cpu

if ($LASTEXITCODE -ne 0) { throw "Video latent caching failed with exit code $LASTEXITCODE" }
