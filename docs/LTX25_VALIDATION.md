# LTX-2.5 integration validation — 2026-10-31

Hardware: NVIDIA RTX 3060 12 GB, approximately 32 GB system RAM, Windows.
An independent `runtimes/ltx25/.venv` was installed with this repository's setup
script, using the pinned AI Toolkit, Diffusers and model revisions. Existing
official ADAM weights were reused through the local store override; weights
were not copied into the repository or downloaded again.

The runtime passed real CUDA INT8 and BF16 matrix multiplication, AdamW8bit,
imports and revision/patch checks. `pip check` reported no broken requirements.
The small-network check passed audio LoRA gradients, adapter save/reload equality,
staged prompt lookup, sliced connector equivalence, PCM decoding and video export.

The full-model fixture was one synthetic 256x256 three-second moving color
pattern with a 48 kHz tone and a combined video/audio caption. It is a pipeline
test, not a quality benchmark. Batch size was 1, rank/alpha 4, 100% offloading,
host pinning off, BF16 compute and FP32 adapter weights.

| Completed operation | Model-stage seconds | Peak allocated VRAM | Peak worker RSS |
| --- | ---: | ---: | ---: |
| Cache video/audio latents and text, without optimizer updates | 23.88 | 2.57 GiB | 4.35 GiB |
| Joint training, 17 frames, one optimizer update | 86.86 | 6.76 GiB | 19.47 GiB |
| Reload that adapter for dev generation, 17 frames, two sampling steps | 107.65 | 7.86 GiB | 24.34 GiB |
| Continue that adapter at 65 frames, one optimizer update | 87.91 | 6.76 GiB | 23.68 GiB |

Model-stage times include model loading and stage output, but exclude the separate
text-encoder process, CLI/import startup and media preparation. The first text
cache stage took 36.50 seconds with 11.89 GiB peak worker RSS; generation's new
prompts took 52.89 seconds with similar RSS. Training reused cached prompt features.
The training optimizer step itself took approximately 65.72 seconds in this run.
Driver/display memory and other processes are additional to allocated VRAM/RSS.
These figures are observations of tiny jobs, not sustained throughput guarantees.

The joint update reported video loss `0.2226901` and audio loss `0.5617921`.
All saved adapter tensors were finite, and all 1,056 inspected audio up-projection
tensors changed from zero initialization. The compatible checkpoint was about
161 MiB. The dev adapter loaded with no missing keys. Its generated MP4 contained
256x256 H.264 video at 24 FPS and AAC sound, both about 0.708 seconds; a separate
24-bit PCM WAV was also exported. Continuation loaded the first adapter with no
missing keys, used a fresh optimizer/schedule and completed a new step numbered 1.
It reported video/audio losses `0.1643576` / `0.1861643`; different frame counts
make these losses unsuitable for a controlled quality comparison. Its optimizer
step took approximately 51.73 seconds.

All ten contract/interface/worker tests passed in the isolated LTX-2.5 runtime,
including the existing ModelScope/LTX-Video adapter regression. In the desktop
environment, nine passed and that worker test was skipped because PEFT is isolated.
They cover
model routing/defaults, GPU policy, early GTX 1060 rejection before installation,
storage portability, PCM splitting, soundtrack validation, archive exclusion,
and caption/cache invalidation. No GTX 1060 was physically tested: its exclusion
is based on the BF16 requirement and this backend's documented support policy.

To reproduce the checks after installing the backend:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\runtimes\ltx25\.venv\Scripts\python.exe .\scripts\check_ltx25_runtime.py
.\runtimes\ltx25\.venv\Scripts\python.exe .\scripts\check_ltx25_pipeline.py
```

Logs, request metadata, synthetic training clips, adapters and generated media
stay in ignored local folders. Full datasets, learned motion/audio quality,
higher resolutions and memory fit on other GPUs have not been validated here.
