. "$PSScriptRoot\common.ps1"
Require-File $DitPath
Require-File $VaePath
Require-File $T5Path

$LoraPath = Join-Path $ProjectRoot "output\my_video_lora.safetensors"
if (-not (Test-Path -LiteralPath $LoraPath -PathType Leaf)) {
    $LatestCheckpoint = Get-ChildItem -LiteralPath (Join-Path $ProjectRoot "output") `
        -Filter "my_video_lora*.safetensors" -File |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $LatestCheckpoint) {
        throw "No trained LoRA checkpoint was found in the output folder."
    }
    $LoraPath = $LatestCheckpoint.FullName
}
$SampleDir = Join-Path $ProjectRoot "samples"
New-Item -ItemType Directory -Force -Path $SampleDir | Out-Null

& $Python "$EngineRoot\src\musubi_tuner\wan_generate_video.py" `
    --fp8 `
    --task t2v-1.3B `
    --from_file (Join-Path $ProjectRoot "validation_prompts.txt") `
    --save_path $SampleDir `
    --output_type video `
    --dit $DitPath `
    --vae $VaePath `
    --t5 $T5Path `
    --attn_mode torch `
    --blocks_to_swap 29 `
    --lora_weight $LoraPath `
    --lora_multiplier 0.8

if ($LASTEXITCODE -ne 0) { throw "Sample generation failed with exit code $LASTEXITCODE" }
