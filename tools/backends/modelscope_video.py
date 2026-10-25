"""Isolated ModelScope inference and spatial/temporal attention LoRA training.

Only this subprocess imports PyTorch, Diffusers and PEFT. Use the isolated ModelScope runtime installed by setup_video_backends.ps1.
"""
from __future__ import annotations

import gc
import json
import math
from pathlib import Path
import random

from tools.backends import modelscope_video_contract as contract


def load_pipeline():
    import torch
    from diffusers import TextToVideoSDPipeline
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the configured ModelScope environment.")
    print(f"Loading {contract.MODEL_ID} on {torch.cuda.get_device_name(0)}", flush=True)
    return TextToVideoSDPipeline.from_pretrained(
        contract.MODEL_ID, revision=contract.REVISION, variant="fp16",
        torch_dtype=torch.float16, use_safetensors=True)


def inject_lora(unet, rank, alpha):
    from peft import LoraConfig, inject_adapter_in_model
    return inject_adapter_in_model(LoraConfig(r=rank, lora_alpha=alpha, target_modules=contract.TARGET_MODULES,
                                              lora_dropout=0.0, bias="none"), unet)


def load_lora(unet, path, strength=1.0):
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from peft.tuners.tuners_utils import BaseTunerLayer
    from safetensors.torch import load_file
    meta = contract.checkpoint_info(Path(path))
    unet = inject_lora(unet, int(meta["rank"]), int(meta["alpha"]))
    weights = load_file(str(path))
    expected = get_peft_model_state_dict(unet)
    if weights.keys() != expected.keys() or any(weights[k].shape != expected[k].shape for k in weights):
        raise ValueError("This LoRA's target layers do not match the ModelScope architecture.")
    outcome = set_peft_model_state_dict(unet, weights)
    if outcome.unexpected_keys:
        raise ValueError(f"Unexpected LoRA keys: {outcome.unexpected_keys[:3]}")
    for module in unet.modules():
        if isinstance(module, BaseTunerLayer):
            module.scale_layer(strength)
    return unet


def generate(request):
    import torch
    from diffusers import DPMSolverMultistepScheduler
    from diffusers.utils import export_to_video
    from tqdm import tqdm
    settings = contract.settings_for("generation", request["settings"])
    pipe = load_pipeline()
    if request.get("model_path") and settings["lora_strength"] != 0:
        pipe.unet = load_lora(pipe.unet, request["model_path"], settings["lora_strength"])
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
    # A named counter lets the desktop app distinguish denoising from loading/download loops.
    pipe.set_progress_bar_config(desc="steps")
    width, height = map(int, settings["format"].split("x"))
    frames = contract.frames_for_seconds(settings["duration"], settings["fps"])
    print(f"Seed: {settings['seed']}; {width}x{height}; {frames} frames", flush=True)
    with torch.inference_mode():
        result = pipe(prompt=settings["prompt"], negative_prompt=settings["negative_prompt"] or None,
                      width=width, height=height, num_frames=frames, num_inference_steps=settings["steps"],
                      guidance_scale=settings["guidance"], generator=torch.Generator(device="cpu").manual_seed(settings["seed"]))
    video = Path(request.get("sample_path", Path(request["output_dir"]) / "video.mp4"))
    export_to_video(result.frames[0], str(video), fps=settings["fps"])
    print(f"Saved: {video}", flush=True)


def fingerprint(folder):
    from tools.backends.common import fingerprint as _fingerprint
    return [list(item) for item in _fingerprint(folder)]


def build_training_cache(pipe, request, settings):
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageOps
    from safetensors.torch import save_file
    import torch
    from tqdm import tqdm
    from tools.backends.common import clips_in

    folder = Path(request["dataset_dir"])
    cache = Path(request["training_cache"])
    cache.mkdir(exist_ok=False)
    width, height = map(int, settings["resolution"].split("x"))
    frames = int(settings["target_frames"])
    paths = []
    clips = clips_in(folder)
    pipe.vae.to(device="cuda", dtype=torch.float32)
    pipe.vae.requires_grad_(False)
    pipe.vae.eval()
    # Keep text encoding separate from VAE encoding so neither competes with UNet training.
    for index, clip in enumerate(tqdm(clips, desc="cache latents")):
        if fingerprint(folder) != request["dataset_files"]:
            raise ValueError("Dataset changed while caching; start a new run.")
        reader = imageio.get_reader(str(clip))
        try:
            count = reader.count_frames()
            if count < frames:
                raise ValueError(f"{clip.name}: requires at least {frames} frames.")
            pixels = [np.asarray(ImageOps.fit(Image.fromarray(reader.get_data(int(i))), (width, height), method=Image.Resampling.LANCZOS))
                      for i in np.linspace(0, count - 1, frames, dtype=int)]
        finally:
            reader.close()
        pixels = torch.from_numpy(np.stack(pixels)).permute(0, 3, 1, 2).float().div(127.5).sub(1)
        chunks = []
        with torch.no_grad():
            for pixel in pixels.split(1):
                chunks.append(pipe.vae.encode(pixel.to("cuda")).latent_dist.sample().cpu())
        latent = torch.cat(chunks).permute(1, 0, 2, 3).unsqueeze(0) * pipe.vae.config.scaling_factor
        path = cache / f"clip_{index:06d}.safetensors"
        save_file({"latents": latent.half().contiguous()}, str(path))
        paths.append((path, clip.with_suffix(".txt").read_text(encoding="utf-8-sig").strip()))
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()
    pipe.text_encoder.to("cuda")
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder.eval()
    from safetensors.torch import load_file
    with torch.no_grad():
        for path, caption in tqdm(paths, desc="cache captions"):
            tokens = pipe.tokenizer(caption, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                                    truncation=True, return_tensors="pt")
            if len(pipe.tokenizer(caption, truncation=False).input_ids) > pipe.tokenizer.model_max_length:
                print(f"Caption truncated to {pipe.tokenizer.model_max_length} tokens: {path.name}", flush=True)
            encoded = pipe.text_encoder(tokens.input_ids.to("cuda"))[0].cpu()
            # Clone out of the memory map before replacing this file on Windows.
            latents = load_file(str(path))["latents"].clone()
            save_file({"latents": latents, "text": encoded.contiguous()}, str(path))
    pipe.text_encoder.to("cpu")
    return [path for path, _ in paths]


def checkpoint_blocks(unet):
    import torch.utils.checkpoint
    # The legacy UNet3D class has no enable_gradient_checkpointing implementation.
    # Checkpoint its blocks without changing state-dict keys or module types.
    for block in [*unet.down_blocks, unet.mid_block, *unet.up_blocks, unet.transformer_in]:
        original = block.forward
        def forward(*args, _original=original, **kwargs):
            return torch.utils.checkpoint.checkpoint(_original, *args, use_reentrant=False, **kwargs)
        block.forward = forward


def save_adapter(unet, path, settings, epoch, step):
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    tensors = {k: v.detach().float().cpu().contiguous() for k, v in get_peft_model_state_dict(unet).items()}
    metadata = {"model_type": contract.MODEL_TYPE, "adapter_format": contract.ADAPTER_FORMAT,
                "base_model": contract.MODEL_ID, "revision": contract.REVISION,
                "rank": str(settings["rank"]), "alpha": str(settings["alpha"]),
                "epoch": str(epoch), "step": str(step), "target_modules": json.dumps(contract.TARGET_MODULES)}
    temporary = path.with_suffix(".tmp")
    save_file(tensors, str(temporary), metadata=metadata)
    temporary.replace(path)


def cache_training_data(pipe, request, settings):
    from tools.backends.common import cached_training_data
    return cached_training_data(request, settings, lambda: build_training_cache(pipe, request, settings))


def train(request):
    import torch
    import torch.nn.functional as F
    from diffusers import DDPMScheduler
    from safetensors.torch import load_file
    from tqdm import tqdm

    settings = contract.settings_for("training", request["settings"])
    seed = settings["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pipe = load_pipeline()
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    cached = cache_training_data(pipe, request, settings)
    unet = pipe.unet
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    unet.requires_grad_(False)
    if request.get("resume_from"):
        meta = contract.checkpoint_info(Path(request["resume_from"]))
        settings.update(rank=int(meta["rank"]), alpha=int(meta["alpha"]))
        unet = load_lora(unet, request["resume_from"])
    else:
        unet = inject_lora(unet, settings["rank"], settings["alpha"])
    trainable = [param for param in unet.parameters() if param.requires_grad]
    if not trainable:
        raise ValueError("No trainable LoRA parameters were injected.")
    # FP32 master adapter weights keep AMP gradient unscaling valid.
    for parameter in trainable:
        parameter.data = parameter.data.float()
    unet.enable_forward_chunking(chunk_size=1, dim=1)
    if settings["gradient_checkpointing"]:
        checkpoint_blocks(unet)
    unet.to("cuda")
    unet.train()
    optimizer = torch.optim.AdamW(trainable, lr=settings["learning_rate"])
    scaler = torch.amp.GradScaler("cuda")
    accumulation = settings["gradient_accumulation"]
    total = math.ceil(len(cached) / accumulation) * request["epochs"]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}; optimizer steps: {total}", flush=True)
    output = Path(request["output_dir"])
    optimizer.zero_grad(set_to_none=True)
    step = 0
    with tqdm(total=total, desc="steps") as progress:
        for epoch in range(1, request["epochs"] + 1):
            if fingerprint(Path(request["dataset_dir"])) != request["dataset_files"]:
                raise ValueError("Dataset changed during training; start a new run.")
            order = list(cached)
            random.shuffle(order)
            for index, path in enumerate(order):
                tensors = load_file(str(path), device="cuda")
                latents, text = tensors["latents"], tensors["text"]
                noise = torch.randn_like(latents)
                timestep = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device="cuda").long()
                noisy = noise_scheduler.add_noise(latents, noise, timestep)
                with torch.autocast("cuda", dtype=torch.float16):
                    prediction = unet(noisy, timestep, encoder_hidden_states=text).sample
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timestep)
                    else:
                        raise ValueError("Unsupported diffusion prediction type.")
                    group_size = min(accumulation, len(order) - (index // accumulation) * accumulation)
                    loss = F.mse_loss(prediction.float(), target.float())
                if not torch.isfinite(loss):
                    raise ValueError("Training loss became non-finite. Reduce learning rate and inspect the dataset.")
                scaler.scale(loss / group_size).backward()
                if (index + 1) % accumulation == 0 or index + 1 == len(order):
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    if not torch.isfinite(norm):
                        raise ValueError("LoRA gradients became non-finite. Reduce learning rate or try a different seed.")
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    progress.set_postfix(loss=f"{loss.item():.5f}", epoch=epoch)
                    progress.update(1)
                    with (output / "training_metrics.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({"step": step, "epoch": epoch, "loss": loss.item(), "gradient_norm": norm.item()}) + "\n")
            if epoch % settings["save_every"] == 0:
                save_adapter(unet, output / f"{request['model_name']}_epoch{epoch:04d}.safetensors", settings, epoch, step)
            info_path = output / "model_info.json"
            info = json.loads(info_path.read_text(encoding="utf-8"))
            info.update(epochs_completed=epoch, optimizer_steps=step)
            temporary = info_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(info, indent=2), encoding="utf-8")
            temporary.replace(info_path)
    final = output / f"{request['model_name']}.safetensors"
    save_adapter(unet, final, settings, request["epochs"], step)
    print(f"Saved adapter: {final}", flush=True)
    print(f"Peak allocated VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f} GiB", flush=True)
