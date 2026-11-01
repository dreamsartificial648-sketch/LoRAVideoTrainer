"""Pure configuration helpers shared by the UI, worker and setup checks."""
from __future__ import annotations

import math
from pathlib import Path

from tools.backends.ltx25_contract import FILES


def model_config(store, settings, profile):
    paths = {key: str(Path(store) / "models" / value) for key, value in FILES.items()}
    offload = settings["offload_percent"] / 100
    return {
        "arch": "ltx2.5", "name_or_path": paths[profile], "dtype": "bf16",
        "te_name_or_path": paths["text_encoder"], "quantize": True, "qtype": "convrot8",
        "quantize_te": True, "qtype_te": "convrot8", "low_vram": True,
        "layer_offloading": offload > 0, "layer_offloading_transformer_percent": offload,
        "layer_offloading_text_encoder_percent": offload,
        "model_kwargs": {"dit_path": paths[profile], "text_encoder_path": paths["text_encoder"],
                         "video_vae_path": paths["video_vae"], "audio_vae_path": paths["audio_vae"]},
    }


def training_config(request):
    s = request["settings"]
    width, height = map(int, s["resolution"].split("x"))
    count = request["clip_count"]
    steps = min(s["max_steps"] or 10**9, request["epochs"] * math.ceil(count / s["gradient_accumulation"]))
    output = Path(request["output_dir"])
    network = {"type": "lora", "linear": s["rank"], "linear_alpha": s["alpha"]}
    if request.get("resume_from"):
        network["pretrained_lora_path"] = request["resume_from"]
    process = {
        "type": "sd_trainer", "training_folder": str(output / "checkpoints"), "device": "cuda:0",
        "network": network, "model": model_config(request["store"], s, "dev"),
        "save": {"dtype": "float32", "save_every": s["save_every"], "max_step_saves_to_keep": 3, "push_to_hub": False},
        "datasets": [{"folder_path": request["prepared_dataset"], "caption_ext": "txt", "caption_dropout_rate": 0,
                      "resolution": [math.ceil(math.sqrt(width * height))], "bucket_tolerance": 32,
                      "num_frames": int(s["target_frames"]), "fps": 24, "do_audio": s["train_audio"],
                      "do_i2v": False, "auto_frame_count": True, "trim_auto_frame_count_tail": True,
                      "cache_latents_to_disk": True, "cache_text_embeddings": True,
                      "num_workers": 0, "batch_load_threads": 1, "cache_latents_num_workers": 1,
                      "random_crop": False, "random_scale": False, "flip_x": False}],
        "train": {"batch_size": 1, "steps": steps, "gradient_accumulation": s["gradient_accumulation"],
                  "train_unet": True, "train_text_encoder": False, "dtype": "bf16",
                  "gradient_checkpointing": s["gradient_checkpointing"], "optimizer": "adamw8bit", "lr": s["learning_rate"],
                  "noise_scheduler": "flowmatch", "timestep_type": "weighted", "seed": s["seed"],
                  "cache_text_embeddings": True, "unload_text_encoder": True,
                  "audio_loss_multiplier": s["audio_loss_multiplier"], "skip_first_sample": True, "disable_sampling": True},
        "sample": {"sampler": "flowmatch", "sample_every": steps + 1, "width": width, "height": height,
                   "num_frames": int(s["target_frames"]), "fps": 24, "prompts": ["A short video with sound"],
                   "guidance_scale": 3., "sample_steps": 25},
        "logging": {"log_every": 1, "use_wandb": False},
    }
    return {"job": "extension", "config": {"name": request["model_name"], "process": [process]}}
