# LoRA Video Trainer

A local Windows desktop app for preparing video datasets, training Wan 2.1 T2V
1.3B LoRAs with Musubi Tuner, and generating test clips. The interface is built
with Tkinter and runs entirely on your computer—there is no browser server and
your clips, captions, checkpoints, and generations are not uploaded by the app.

## What it includes

- import, preview, caption, validate, and batch-split training clips
- cache latents and text embeddings, train LoRAs, and follow live progress
- configurable epochs, learning rate, rank/alpha, and block swapping
- resume from a LoRA checkpoint and generate comparable test videos
- organize generated takes in an ordered storyboard
- automatically restore local settings between sessions

## Requirements

- Windows 10 or 11
- an NVIDIA GPU with approximately 12 GB VRAM (the tested preset uses an RTX 3060)
- 32 GB system RAM recommended
- 64-bit Python 3.10 and Git for Windows
- enough disk space for the Python environment, model weights, caches, and outputs

The setup uses CUDA 12.4 PyTorch wheels. A current NVIDIA driver compatible with
that runtime is required.

## Install

Open PowerShell in the project folder and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\download_models.ps1
```

The second command downloads the Wan model files from Hugging Face and can take
a while. Model weights are never stored in this repository.

Then double-click `Launch LoRA Video Trainer.bat`.

## First training run

1. In **Settings**, choose a unique trigger such as `my_subject_token`.
2. In **Dataset**, import short authorized source videos or split longer videos.
3. Add a concise visual caption to every clip and include the exact trigger.
4. Run **Validate Dataset**, then **Run Full Pipeline**.
5. Choose a unique run name when prompted. Checkpoints appear in `output/`.

A useful starting dataset is 25–40 clips, each one continuous 2–4 second shot,
with varied framing, expressions, clothing, and backgrounds. Use only footage you
have the right and consent to train on.

## Local data and privacy

The following paths are deliberately excluded from Git and start empty on every
fresh clone:

- `dataset/` — source clips and captions
- `models/` — downloaded base model weights
- `cache/` — cached training tensors
- `output/` — trained LoRA checkpoints
- `samples/` — generated videos
- `logs/` — training logs
- `app_settings.json`, `storyboard.json`, and `dataset_config.toml` — local state
- `.venv/` and `trainer-engine/` — downloaded dependencies

Before publishing a fork, review the staged file list with `git status` and never
force-add files from these paths.

## Model and trainer sources

The setup pins [Musubi Tuner](https://github.com/kohya-ss/musubi-tuner) to the
revision tested with this app. Model downloads come from:

- `Comfy-Org/Wan_2.1_ComfyUI_repackaged` (DiT and VAE)
- `Wan-AI/Wan2.1-I2V-14B-720P` (UMT5 text encoder)

Those projects and model files have their own licenses and terms. They are not
redistributed by this repository.

## Manual commands

The app is the normal workflow. For troubleshooting, these commands run the same
individual stages:

```powershell
.\.venv\Scripts\python.exe .\tools\verify_setup.py
.\.venv\Scripts\python.exe .\tools\validate_dataset.py --trigger my_subject_token
.\scripts\01_cache_video.ps1
.\scripts\02_cache_text.ps1
.\scripts\03_train.ps1
.\scripts\04_generate_sample.ps1
```

Wan 2.1 generation uses the supported 832×480 landscape or 480×832 portrait
buckets. Longer clips cost substantially more time and VRAM, so begin with the
short fast-preview preset.

## Project status

This is an early Windows-focused release. Expect long setup, training, and
generation times, and keep backups of any dataset or checkpoint you value.
