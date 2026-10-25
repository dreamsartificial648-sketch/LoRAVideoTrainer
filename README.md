# LoRA Video Trainer

A local Windows desktop app for preparing video datasets, training video
LoRAs, and generating test clips. Supported models are Wan 2.1 T2V 1.3B,
ModelScope T2V 1.7B, and LTX-Video 2B 0.9.6 dev. The interface is built
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

The Wan setup uses CUDA 12.4 PyTorch wheels; the additional backends use CUDA 12.6. A current NVIDIA driver compatible with
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

## ModelScope and LTX-Video 2B

Use the model selector at the top of the app. **Install Model Backend** creates
an isolated Python environment for the selected model; **Download Model Files**
downloads its pinned weights into `models/<backend>/`. These downloads can be
large. Wan's existing environment and model files are retained.

You can also install the two additional environments from PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_video_backends.ps1
```

Pass `-Backend modelscope_video` or `-Backend ltx_video` to install just one.
The desktop app itself still uses the `.venv` created by `setup.ps1`.

| Model | Initial training preset | Generation frames |
| --- | --- | --- |
| Wan 2.1 T2V 1.3B | Existing 448×256, 25/49 frames, rank 16 | 4N+1 |
| ModelScope T2V 1.7B | 256×256, 8 frames, rank/alpha 4, FP16 | Seconds × FPS, rounded |
| LTX-Video 2B 0.9.6 dev | 256×256, 17 frames, rank/alpha 4, BF16 | 8N+1, capped at 257 |

For the additional models, choose the training resolution and frame count on the
Training tab. Both cache buttons prepare latents and captions together. Training
also prepares or reuses the cache automatically; changed captions, videos,
resolution, frame count, or seed invalidate it. Start with the small presets;
higher resolutions and longer clips require more memory. LTX requires BF16 GPU
support. These two backends are experimental and have not been validated with a
full-size training run in this integration.

New checkpoints are stored in `output/<backend>/<run name>/`, with per-epoch
checkpoints and `training_metrics.jsonl`. The checkpoint picker shows the selected
model's outputs. Continuation accepts this trainer's compatible attention LoRAs
(also compatible ADAM LoRAs), restores their rank/alpha, and starts a fresh
optimizer. Other LoRA formats are rejected. Block swapping and TensorBoard
logging are Wan features; the additional backends use gradient checkpointing and
JSONL metrics. Cache data lives in `cache/<backend>/` and samples remain in
`samples/` for preview and storyboard use.

LTX here means **LTX-Video 2B 0.9.6 dev**, distinct from LTX-2.

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
- `runtimes/` — isolated additional-model Python environments

Before publishing a fork, review the staged file list with `git status` and never
force-add files from these paths.

## Model and trainer sources

The setup pins [Musubi Tuner](https://github.com/kohya-ss/musubi-tuner) to the
revision tested with this app. Model downloads come from:

- `Comfy-Org/Wan_2.1_ComfyUI_repackaged` (DiT and VAE)
- `Wan-AI/Wan2.1-I2V-14B-720P` (UMT5 text encoder)
- [ali-vilab/text-to-video-ms-1.7b](https://huggingface.co/ali-vilab/text-to-video-ms-1.7b) (ModelScope)
- [Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video) (2B 0.9.6 dev checkpoint and T5)
- `Lightricks/LTX-Video-0.9.5` (LTX VAE and transformer configuration)
- `Lightricks/LTX-Video-0.9.7-dev` (LTX scheduler)

Those projects and model files have their own licenses and terms. They are not
redistributed by this repository.
The additional workers use pinned Diffusers/PEFT dependencies and model revisions
recorded in `tools/backends/`. See `THIRD_PARTY_NOTICES.md` for ADAM source attribution.

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
