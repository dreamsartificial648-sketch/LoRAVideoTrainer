. "$PSScriptRoot\common.ps1"
Require-File $T5Path

& $Python "$EngineRoot\src\musubi_tuner\wan_cache_text_encoder_outputs.py" `
    --dataset_config $DatasetConfig `
    --t5 $T5Path `
    --batch_size 1 `
    --fp8_t5

if ($LASTEXITCODE -ne 0) { throw "Text caching failed with exit code $LASTEXITCODE" }
