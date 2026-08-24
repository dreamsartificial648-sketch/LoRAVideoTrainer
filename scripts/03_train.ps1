. "$PSScriptRoot\common.ps1"
Require-File $DitPath

$OutputDir = Join-Path $ProjectRoot "output"
$LoggingDir = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Force -Path $OutputDir, $LoggingDir | Out-Null

& $Accelerate launch `
    --num_cpu_threads_per_process 1 `
    --mixed_precision bf16 `
    "$EngineRoot\src\musubi_tuner\wan_train_network.py" `
    --task t2v-1.3B `
    --dit $DitPath `
    --dataset_config $DatasetConfig `
    --sdpa `
    --mixed_precision bf16 `
    --save_precision bf16 `
    --fp8_base `
    --optimizer_type adamw8bit `
    --learning_rate 0.0001 `
    --gradient_checkpointing `
    --blocks_to_swap 20 `
    --block_swap_h2d_only `
    --block_swap_ring_size 1 `
    --max_data_loader_n_workers 1 `
    --network_module networks.lora_wan `
    --network_dim 16 `
    --network_alpha 16 `
    --timestep_sampling shift `
    --discrete_flow_shift 3.0 `
    --max_train_epochs 12 `
    --save_every_n_epochs 2 `
    --seed 42 `
    --output_dir $OutputDir `
    --output_name my_video_lora `
    --log_with tensorboard `
    --logging_dir $LoggingDir

if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE" }
