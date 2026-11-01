"""LTX-2.5 contracts adapted from ADAM; safe to import in the desktop process."""
from __future__ import annotations
import json
import os
from pathlib import Path
from tools.backends.common import validate_settings

ROOT = Path(__file__).resolve().parents[2]


def runtime_root(root=ROOT):
    return Path(root) / 'runtimes/ltx25'


def store_root(root=ROOT):
    """Optional external SSD; never ship a machine-specific path."""
    override = os.environ.get('LORAVIDEO_LTX25_STORE')
    if override:
        return Path(override).expanduser().resolve()
    config = runtime_root(root) / 'paths.json'
    if config.is_file():
        return Path(json.loads(config.read_text(encoding='utf-8'))['store']).expanduser().resolve()
    return Path(root).resolve() / 'models/ltx25'


def settings_for(mode, values):
    return validate_settings(values, TRAINING_SETTINGS if mode == 'training' else GENERATION_SETTINGS)


def hardware_report(name, capability, vram_bytes, ram_bytes):
    """Release support policy, separate from an assertion that the model will fit."""
    errors, warnings = [], []
    if capability[0] < 8:
        errors.append('This LTX-2.5 backend requires native BF16 (NVIDIA compute capability 8.0+). '
                      'GTX 1060/Pascal and RTX 20-series GPUs are unsupported by this backend; '
                      'extra RAM or offloading cannot add native BF16 support.')
    if vram_bytes < 8 * 1024**3:
        errors.append('This experimental LTX-2.5 release requires at least 8 GiB total VRAM. '
                      '6 GB GPUs are unsupported; the measured RTX 3060 training peak already exceeded 6 GiB.')
    elif vram_bytes < 11 * 1024**3:
        warnings.append('8–10 GiB GPUs are unverified. Start at 256x256, 17 frames, rank 4, '
                        '100% offloading and a one-step cap. Memory fit is not guaranteed.')
    if ram_bytes < 30 * 1024**3:
        warnings.append('32 GB system RAM is recommended. Offloading uses system RAM; '
                        'the reference worker reached approximately 22–24 GiB RSS, excluding other programs.')
    return {'gpu': name, 'compute_capability': list(capability), 'vram_gib': round(vram_bytes / 1024**3, 2),
            'ram_gib': round(ram_bytes / 1024**3, 2), 'errors': errors, 'warnings': warnings}


def require_hardware():
    import torch
    import psutil
    if not torch.cuda.is_available():
        raise ValueError('LTX-2.5 requires an NVIDIA CUDA GPU. Install its isolated backend and a compatible driver.')
    device = torch.cuda.get_device_properties(0)
    report = hardware_report(device.name, (device.major, device.minor), device.total_memory, psutil.virtual_memory().total)
    print(json.dumps(report, indent=2), flush=True)
    if report['errors']:
        raise ValueError('\n'.join(report['errors']))
    return report

MODEL_ID = "Lightricks/LTX-2.5"

REVISION = "49b093a39c19169a8a81721dcc541e07e945232f"

UPSTREAM_REVISION = "126ac4d3b8a09b5403ee2f5ee6f59d88453b478c"

MODEL_TYPE = "adam_ltx25_av_lora_v1"

FILES = {
    "distilled": "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
    "dev": "diffusion_models/ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors",
    "text_encoder": "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    "video_vae": "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
    "audio_vae": "vae/ltx-2.5-audio-vae-bf16.safetensors",
}

TRAINING_SETTINGS = {
    "trigger_word": {"label": "Trigger word", "type": "text", "default": "subject_token", "required": True, "group": "Dataset"},
    "resolution": {"label": "Training resolution", "type": "choice", "options": ["256x256", "384x256", "512x320", "512x512"], "default": "256x256", "group": "Dataset"},
    "target_frames": {"label": "Frames per clip", "type": "choice", "options": ["17", "33", "49", "65", "97", "193"], "default": "17", "group": "Dataset"},
    "train_audio": {"label": "Train synchronized audio too", "type": "bool", "default": True, "group": "Dataset", "description": "Requires recorded audio in every training clip. Uses joint video and audio loss."},
    "audio_loss_multiplier": {"label": "Audio loss weight", "type": "float", "default": 1., "min": .01, "max": 10., "decimals": 2, "group": "Training"},
    "learning_rate": {"label": "Learning rate", "type": "float", "default": .0001, "min": .0000001, "max": .01, "decimals": 7, "group": "Training"},
    "rank": {"label": "LoRA rank", "type": "int", "default": 4, "min": 1, "max": 64, "group": "Training"},
    "alpha": {"label": "LoRA alpha", "type": "int", "default": 4, "min": 1, "max": 128, "group": "Training"},
    "gradient_accumulation": {"label": "Accumulate clips", "type": "int", "default": 1, "min": 1, "max": 32, "group": "Training"},
    "gradient_checkpointing": {"label": "Gradient checkpointing", "type": "bool", "default": True, "group": "Memory"},
    "offload_percent": {"label": "Layers offloaded to RAM (%)", "type": "int", "default": 100, "min": 0, "max": 100, "group": "Memory"},
    "pin_host_memory": {"label": "Pin offloaded weights in RAM", "type": "bool", "default": False, "group": "Memory", "advanced": True, "description": "Can speed transfers with abundant RAM; keep off on a 32 GB system to avoid large pinned-memory copies."},
    "save_every": {"label": "Save every N optimizer steps", "type": "int", "default": 1, "min": 1, "max": 10000, "group": "Checkpoints"},
    "max_steps": {"label": "Optimizer step cap (0 = epochs)", "type": "int", "default": 1, "min": 0, "max": 100000, "group": "Advanced"},
    "seed": {"label": "Training seed", "type": "int", "default": 42, "min": 0, "max": 2147483647, "group": "Advanced"},
}

GENERATION_SETTINGS = {
    "prompt": {"label": "Video and sound description", "type": "multiline_text", "default": "", "required": True, "group": "Prompt"},
    "negative_prompt": {"label": "Negative prompt (dev)", "type": "text", "default": "", "group": "Prompt"},
    "profile": {"label": "Inference model", "type": "choice", "options": ["distilled", "dev"], "default": "distilled", "group": "Generation", "description": "Distilled uses its fixed 8-step schedule. Custom LoRAs use the matching dev base."},
    "format": {"label": "Resolution", "type": "choice", "options": ["256x256", "384x256", "512x320", "512x512", "704x480", "480x704"], "default": "256x256", "group": "Video"},
    "duration": {"label": "Requested seconds", "type": "float", "default": .75, "min": .5, "max": 8., "decimals": 2, "step": .25, "group": "Video"},
    "fps": {"label": "FPS", "type": "int", "default": 24, "min": 24, "max": 24, "group": "Video"},
    "steps": {"label": "Dev sampling steps", "type": "int", "default": 25, "min": 2, "max": 100, "group": "Generation"},
    "guidance": {"label": "Dev guidance", "type": "float", "default": 3., "min": 1., "max": 10., "decimals": 1, "group": "Generation"},
    "lora_strength": {"label": "LoRA strength", "type": "float", "default": 1., "min": 0., "max": 2., "decimals": 2, "group": "Generation"},
    "offload_percent": {"label": "Layers offloaded to RAM (%)", "type": "int", "default": 100, "min": 0, "max": 100, "group": "Memory"},
    "pin_host_memory": {"label": "Pin offloaded weights in RAM", "type": "bool", "default": False, "group": "Memory", "advanced": True, "description": "Keep off with 32 GB RAM. Pageable weights reduce loading copies but can make transfers slower."},
    "seed": {"label": "Seed", "type": "int", "default": 42, "min": 0, "max": 2147483647, "group": "Generation"},
    "randomize_seed": {"label": "Randomize seed", "type": "bool", "default": False, "group": "Generation"},
}

def frames_for_seconds(seconds, fps):
    return max(9, min(193, 8 * round((seconds * fps - 1) / 8) + 1))

def _read_checkpoint_info(path):
    from safetensors import safe_open
    path = Path(path)
    if not path.is_file() or path.suffix.lower() != ".safetensors":
        raise ValueError("Choose a compatible LTX-2.5 dev LoRA safetensors checkpoint.")
    with safe_open(str(path), framework="np") as stream:
        meta = stream.metadata() or {}
        if (meta.get("model_type") != MODEL_TYPE or meta.get("base_model") != MODEL_ID or meta.get("revision") != REVISION
                or meta.get("base_profile") != "dev"):
            raise ValueError("This is not a compatible LoRA Video Trainer / ADAM LTX-2.5 dev LoRA. Old LTX-Video, Wan and audio-only adapters cannot be used.")
        if not any("lora" in key.lower() for key in stream.keys()):
            raise ValueError("Checkpoint contains no adapter tensors.")
        for field, limit in (("rank", 64), ("alpha", 128)):
            try:
                valid = 1 <= int(meta.get(field, "0")) <= limit
            except (TypeError, ValueError):
                valid = False
            if not valid:
                raise ValueError("Invalid adapter rank or alpha.")
        return meta

def checkpoint_info(path):
    from safetensors import SafetensorError
    try:
        return _read_checkpoint_info(path)
    except (SafetensorError, OSError, KeyError, TypeError) as exc:
        raise ValueError("Cannot read a complete LTX-2.5 LoRA checkpoint; the file may be interrupted or incompatible.") from exc
