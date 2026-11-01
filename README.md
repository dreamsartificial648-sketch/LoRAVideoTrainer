# LoRA Video Trainer

A local Windows desktop app for preparing video datasets, training video
LoRAs, and generating test clips. Supported models are Wan 2.1 T2V 1.3B,
ModelScope T2V 1.7B, LTX-Video 2B 0.9.6 dev, and experimental **LTX-2.5 22B audio + video**. The interface is built
with Tkinter and runs entirely on your computer—there is no browser server and
your clips, captions, checkpoints, and generations are not uploaded by the app.

## What it includes

- import, preview, caption, validate, and batch-split training clips
- cache latents and text embeddings, train LoRAs, and follow live progress
- configurable epochs, learning rate, rank/alpha, and block swapping
- resume from a LoRA checkpoint and generate comparable test videos
- organize generated takes in an ordered storyboard
- automatically restore local settings between sessions
- LTX-2.5 joint video/audio or visual-only LoRA training, staged text encoding,
  INT8 base weights, CPU offloading, and synchronized MP4/WAV generation

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

The older **LTX-Video 2B 0.9.6 dev** and **LTX-2.5 22B** are separate backends.
Their model weights and adapters are incompatible.

## LTX-2.5 22B (experimental)

Select **LTX-2.5 22B audio + video (experimental)**, then **Install Model Backend**.
It installs a separate Python 3.10 / CUDA 12.6 environment and a pinned AI Toolkit
checkout. Wan, ModelScope and LTX-Video keep their own environments. LTX-2.5 is an
explicit opt-in and is not installed by the older `-Backend all` command.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_video_backends.ps1 -Backend ltx25
```

For an external SSD, choose **Choose Model Storage** before downloading, or add
`-Ltx25Store 'E:\LTX25'` to that command. The default weight store is
`models/ltx25/`; the optional override is saved in ignored `runtimes/ltx25/paths.json`.
`LORAVIDEO_LTX25_STORE` can also override the store for a launch. Existing ADAM
users may select ADAM's LTX-2.5 **store root** (the directory containing `models/`)
to reuse the same base weights. This does not copy datasets or trained LoRAs.

1. On **Training**, click **Model Access / Login**. Read and accept the applicable terms on the
   [official LTX-2.5 model page](https://huggingface.co/Lightricks/LTX-2.5), and enter
   your own Hugging Face read token in the local masked dialog. Credentials go to
   Hugging Face's local credential store, never app settings, command arguments,
   requests, or Git. You can also log in locally using the backend's `hf auth login`.
2. Click **Download Model Files**. Both dev and distilled profiles plus the text
   encoder and VAEs total approximately **56.1 GiB**. Downloads are resumable and
   new downloads are SHA256-verified. Leave at least 20 GiB extra on the weight
   drive, and additional space for the environment, paging, prepared clips and
   caches. Larger datasets can require substantially more space.
3. Import captioned clips or split originals with LTX-2.5 selected. This splitter
   preserves source sound as PCM MKV at 24 FPS. Existing silent clips made for the
   older models need re-splitting for audio training. Describe visible motion and
   sound in each caption, for example `subject_token closes a door. Audio: A wooden thud.`
   Turn **Train synchronized audio** off for a visual-only dataset.
4. Start with **256x256, 17 frames, rank/alpha 4, offload 100%, host pinning off,
   step cap 1, save every 1 step**. Cache Video and Cache Captions each prepare
   both streams and text, without optimizer updates; Train also prepares caches
   automatically. Changing sources, captions, resolution, frame count, audio mode
   or seed selects a new cache. Archived originals are excluded from training.
5. Generate with the saved adapter to check loading. A one-step adapter tests the
   pipeline and is not a useful finished LoRA. For a real run, set **Step cap to 0**
   for the epoch-equivalent budget, and increase the save interval if desired.
   Toolkit sampling determines clip order; an epoch-equivalent budget does not
   guarantee each clip is sampled once. Use **65 frames** for about 2.71 seconds
   from a three-second source, after the shorter test succeeds.

Training uses the dev base, batch size 1, BF16 compute, FP32 adapters, AdamW8bit,
gradient checkpointing and cached features. The quantized text encoder runs in a
separate process and exits before the denoiser loads. Continuation loads adapter
weights/rank/alpha with a **fresh optimizer/schedule**, not exact training-state
resume. Checkpoints use atomic saves and are stored under `output/ltx25/<run>/`.
Compatible ADAM LTX-2.5 adapters are accepted; Wan and LTX-Video 2B adapters are rejected.

For generation, **Use Base Model** clears the adapter selection: distilled uses
its fixed eight-step schedule. Selecting an adapter automatically uses dev,
regardless of the base-profile selector. Native output is 24 FPS with synchronized
audio; MP4 and PCM WAV files go to `samples/`, alongside a run directory with
settings and measured memory/timing. Double-click an MP4 to hear its audio in your
system player; the lightweight in-app frame preview is silent. This integration
does not include image-to-video or the refinement/upscaling stage.

| Hardware | LTX-2.5 support in this release |
| --- | --- |
| RTX 3060 12 GB + 32 GB RAM | Reference configuration; experimental, start with the tiny preset |
| Native-BF16 NVIDIA GPU with 8–10 GiB VRAM | Unverified; warning before work, fit is not guaranteed |
| GTX 1060 6 GB / Pascal | Unsupported: no native BF16 and below this backend's VRAM floor |
| RTX 20-series / other pre-Ampere GPUs | Unsupported by this BF16 backend, even with more VRAM |
| CPU-only | Unsupported |

The backend checks compute capability (8.0+), total VRAM (at least 8 GiB), RAM,
and actual CUDA kernels before declaring setup ready. The 8 GiB floor is a release
support policy, **not a measured minimum or a promise that any 8 GiB GPU fits**.
The ADAM reference measured approximately 6.76 GiB PyTorch-allocated VRAM for tiny
training jobs, approximately 22–24 GiB worker RAM, and about 43 seconds for an
optimizer step with prompt features already cached. Driver/display memory, other
programs, setup and encoding add to those figures. CPU offloading consumes RAM
and transfer bandwidth; it cannot make a GTX 1060 acquire BF16 support.

Hobby use and posting generated videos on YouTube still require compliance with
the applicable model, input-data and content terms. This source release does not
redistribute base weights or grant rights to them. The LTX-2.x Community License
has revenue-based commercial requirements, derivative-transfer obligations and
restricted uses, including a competing-product restriction; free/hobby use is
not a blanket exemption. See [the official license](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x)
and [model terms and notices](docs/LTX25_MODEL_TERMS.md). Sharing the trainer's
source and distributing a trained LoRA are separate licensing questions. No
approval or endorsement by Lightricks is implied.

Validate the installation without downloading model weights:

```powershell
.\runtimes\ltx25\.venv\Scripts\python.exe .\tools\video_backend.py --backend ltx25 --mode check
.\runtimes\ltx25\.venv\Scripts\python.exe .\scripts\check_ltx25_runtime.py
```

The opt-in `scripts/check_ltx25_pipeline.py` runs the full model against a synthetic
clip using weights already present locally. It checks caches, a one-step joint
update, adapter generation and 65-frame continuation; it can take several minutes
and does not assess learned output quality. Its logs and media stay out of Git.
See [measured integration validation](docs/LTX25_VALIDATION.md) for the actual
RTX 3060 checks and their limits.

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
- [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) (quantized dev/distilled transformers,
  matching Gemma 4 encoder and audio/video VAEs; pinned revision in `ltx25_contract.py`)

Those projects and model files have their own licenses and terms. They are not
redistributed by this repository. LTX-2.5 uses a separately downloaded,
MIT-licensed [AI Toolkit](https://github.com/ostris/ai-toolkit) checkout and pinned
Diffusers/Transformers dependencies, recorded in `tools/ltx25_runtime/`.
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
