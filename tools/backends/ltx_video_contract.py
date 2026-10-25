from __future__ import annotations
import json
from pathlib import Path
from tools.backends.common import validate_settings

MODEL_ID = "Lightricks/LTX-Video"

REVISION = "8984fa25007f376c1a299016d0957a37a2f797bb"

CHECKPOINT = "ltxv-2b-0.9.6-dev-04-25.safetensors"

MODEL_TYPE = "ltx_video_2b_096_dev_lora"

ADAPTER_FORMAT = "adam_peft_attention_v1"

TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]

def prepare_cache(cache: Path) -> None:
    """Recreate cleared caches, including the target of a Windows junction."""
    # mkdir through a dangling junction cannot recreate its missing target.
    (cache.resolve() / "hub").mkdir(parents=True, exist_ok=True)

def frames_for_seconds(seconds: float, fps: int) -> int:
    return min(257, max(9, 8 * round((seconds * fps - 1) / 8) + 1))

def checkpoint_info(path: Path) -> dict[str, str]:
    """Accept only this integration's explicitly identified safetensors format."""
    from safetensors import safe_open
    if path.suffix.lower() != ".safetensors" or not path.is_file():
        raise ValueError("Choose an LTX-Video .safetensors LoRA checkpoint.")
    try:
        with safe_open(str(path), framework="np") as stream:
            meta = stream.metadata() or {}
            if (meta.get("model_type") != MODEL_TYPE or meta.get("adapter_format") != ADAPTER_FORMAT
                    or meta.get("base_model") != MODEL_ID or meta.get("revision") != REVISION
                    or meta.get("checkpoint") != CHECKPOINT):
                raise ValueError("This is not an LTX-Video 2B 0.9.6 dev LoRA. Wan, SDXL and legacy LTX-Video LoRA formats cannot be loaded here.")
            rank, alpha = int(meta["rank"]), int(meta["alpha"])
            if not 1 <= rank <= 64 or not 1 <= alpha <= 128:
                raise ValueError("Invalid LoRA rank or alpha.")
            keys = set(stream.keys())
            if not keys or any(not k.endswith((".lora_A.weight", ".lora_B.weight")) for k in keys):
                raise ValueError("Checkpoint contains unsupported adapter tensors.")
            for key in keys:
                shape = stream.get_slice(key).get_shape()
                if len(shape) != 2 or (shape[0] if key.endswith(".lora_A.weight") else shape[1]) != rank:
                    raise ValueError("Adapter tensor dimensions do not match the declared rank.")
                partner = key.replace(".lora_A.weight", ".lora_B.weight") if key.endswith(".lora_A.weight") else key.replace(".lora_B.weight", ".lora_A.weight")
                if partner not in keys:
                    raise ValueError("Checkpoint contains an incomplete LoRA pair.")
            return meta
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Cannot read LTX-Video checkpoint: {path.name}: {exc}") from exc

TRAINING_SETTINGS = {
    "trigger_word": {"label": "Trigger word", "type": "text", "default": "subject_token", "required": True, "group": "Dataset"},
    "resolution": {"label": "Training resolution", "type": "choice", "options": ["256x256", "384x256", "512x320", "512x512"], "default": "256x256", "group": "Dataset"},
    "target_frames": {"label": "Frames per clip", "type": "choice", "options": ["9", "17", "25", "49"], "default": "17", "group": "Dataset", "description": "Samples evenly across each clip. Short, consistent clips work best."},
    "learning_rate": {"label": "Learning rate", "type": "float", "default": 0.0001, "min": 0.0000001, "max": 0.01, "decimals": 7, "group": "Training"},
    "rank": {"label": "LoRA rank", "type": "int", "default": 4, "min": 1, "max": 64, "group": "Training"},
    "alpha": {"label": "LoRA alpha", "type": "int", "default": 4, "min": 1, "max": 128, "group": "Training"},
    "gradient_accumulation": {"label": "Accumulate clips", "type": "int", "default": 1, "min": 1, "max": 32, "group": "Training", "description": "One clip per microbatch; accumulate gradients for a larger effective batch."},
    "gradient_checkpointing": {"label": "Gradient checkpointing", "type": "bool", "default": True, "group": "Memory"},
    "save_every": {"label": "Save every N epochs", "type": "int", "default": 1, "min": 1, "max": 1000, "group": "Checkpoints"},
    "seed": {"label": "Training seed", "type": "int", "default": 42, "min": 0, "max": 2147483647, "group": "Advanced", "advanced": True},
}

GENERATION_SETTINGS = {
    "prompt": {"label": "Prompt", "type": "multiline_text", "default": "", "required": True, "group": "Prompt"},
    "negative_prompt": {"label": "Negative prompt", "type": "multiline_text", "default": "", "group": "Prompt"},
    "format": {"label": "Video resolution", "type": "choice", "options": ["256x256", "384x256", "512x320", "512x512", "704x480", "480x704"], "default": "512x320", "group": "Video"},
    "duration": {"label": "Requested seconds", "type": "float", "default": 2.0, "min": 0.5, "max": 8.0, "decimals": 2, "step": 0.5, "group": "Video"},
    "fps": {"label": "Video FPS", "type": "int", "default": 24, "min": 4, "max": 30, "group": "Video", "description": "Controls temporal conditioning and saved playback speed."},
    "steps": {"label": "Inference steps", "type": "int", "default": 30, "min": 1, "max": 100, "group": "Generation"},
    "guidance": {"label": "Prompt guidance (CFG)", "type": "float", "default": 3.5, "min": 0.0, "max": 30.0, "decimals": 2, "step": 0.5, "group": "Generation"},
    "lora_strength": {"label": "LoRA strength", "type": "float", "default": 1.0, "min": 0.0, "max": 2.0, "decimals": 2, "step": 0.05, "group": "Generation"},
    "seed": {"label": "Seed", "type": "int", "default": 42, "min": 0, "max": 2147483647, "group": "Generation"},
    "randomize_seed": {"label": "Randomize seed", "type": "bool", "default": False, "group": "Generation"},
}

def settings_for(mode, values):
    return validate_settings(values, TRAINING_SETTINGS if mode == "training" else GENERATION_SETTINGS)
