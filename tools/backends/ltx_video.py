"""LTX-Video 2B 0.9.6 dev: staged loading and transformer flow-matching LoRA.

Executed only by the isolated runtime. No GPU packages are imported by the UI.
"""
from __future__ import annotations

import gc
import json
import math
from pathlib import Path
import random

from tools.backends import ltx_video_contract as contract

VAE_REPO = "Lightricks/LTX-Video-0.9.5"
VAE_REVISION = "e58e28c39631af4d1468ee57a853764e11c1d37e"
SCHEDULER_REPO = "Lightricks/LTX-Video-0.9.7-dev"
SCHEDULER_REVISION = "2101082f5eb5540770a2df43747feadb6f69b889"
CHECKPOINT = contract.CHECKPOINT


def release():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def encode_captions(captions):
    """T5 XXL is offloaded by leaf module; it never shares VRAM with the DiT."""
    import torch
    from accelerate import cpu_offload
    from transformers import T5EncoderModel, T5TokenizerFast
    from tqdm import tqdm
    print("Encoding captions with offloaded T5 XXL", flush=True)
    tokenizer = T5TokenizerFast.from_pretrained(contract.MODEL_ID, revision=contract.REVISION, subfolder="tokenizer")
    encoder = T5EncoderModel.from_pretrained(contract.MODEL_ID, revision=contract.REVISION,
                                             subfolder="text_encoder", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    encoder.requires_grad_(False).eval()
    cpu_offload(encoder, execution_device=torch.device("cuda"))
    encoded = []
    with torch.no_grad():
        for caption in tqdm(captions, desc="cache captions"):
            if len(tokenizer(caption, truncation=False).input_ids) > 128:
                print("Caption exceeds 128 tokens and will be truncated.", flush=True)
            tokens = tokenizer(caption, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
            mask = tokens.attention_mask.to("cuda")
            text = encoder(tokens.input_ids.to("cuda"), attention_mask=mask)[0]
            encoded.append({"text": text.cpu().contiguous(), "mask": mask.cpu().contiguous()})
    del encoder, tokenizer
    release()
    return encoded


def load_vae():
    import torch
    from diffusers import AutoencoderKLLTXVideo
    vae = AutoencoderKLLTXVideo.from_pretrained(VAE_REPO, revision=VAE_REVISION, subfolder="vae", torch_dtype=torch.bfloat16)
    vae.enable_tiling()
    return vae


def load_transformer():
    import torch
    from diffusers import LTXVideoTransformer3DModel
    from huggingface_hub import hf_hub_download
    checkpoint = hf_hub_download(contract.MODEL_ID, CHECKPOINT, revision=contract.REVISION)
    return LTXVideoTransformer3DModel.from_single_file(
        checkpoint, config=VAE_REPO, subfolder="transformer", revision=VAE_REVISION, torch_dtype=torch.bfloat16)


def inject_lora(model, rank, alpha):
    from peft import LoraConfig, inject_adapter_in_model
    return inject_adapter_in_model(LoraConfig(r=rank, lora_alpha=alpha, target_modules=contract.TARGET_MODULES,
                                              lora_dropout=0.0, bias="none"), model)


def load_lora(model, path, strength=1.0):
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from peft.tuners.tuners_utils import BaseTunerLayer
    from safetensors.torch import load_file
    meta = contract.checkpoint_info(Path(path))
    model = inject_lora(model, int(meta["rank"]), int(meta["alpha"]))
    weights = load_file(str(path))
    expected = get_peft_model_state_dict(model)
    if weights.keys() != expected.keys() or any(weights[k].shape != expected[k].shape for k in weights):
        raise ValueError("This LoRA's target layers do not match LTX-Video 2B 0.9.6 dev.")
    outcome = set_peft_model_state_dict(model, weights)
    if outcome.unexpected_keys:
        raise ValueError(f"Unexpected LoRA keys: {outcome.unexpected_keys[:3]}")
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            module.scale_layer(strength)
    return model


def generate(request):
    import torch
    from diffusers import LTXPipeline, FlowMatchEulerDiscreteScheduler
    from diffusers.utils import export_to_video
    settings = contract.settings_for("generation", request["settings"])
    positive, negative = encode_captions([settings["prompt"], settings["negative_prompt"]])
    pipe = LTXPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(SCHEDULER_REPO, revision=SCHEDULER_REVISION, subfolder="scheduler"),
        vae=load_vae(), transformer=load_transformer(), tokenizer=None, text_encoder=None)
    if request.get("model_path") and settings["lora_strength"] != 0:
        pipe.transformer = load_lora(pipe.transformer, request["model_path"], settings["lora_strength"])
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(desc="steps")
    # LTXPipeline 0.35 leaves supplied embeddings on their original device.
    # Encoder output was intentionally cached on CPU before releasing T5.
    positive = {key: value.to("cuda") for key, value in positive.items()}
    negative = {key: value.to("cuda") for key, value in negative.items()}
    width, height = map(int, settings["format"].split("x"))
    frames = contract.frames_for_seconds(settings["duration"], settings["fps"])
    print(f"Seed {settings['seed']}; {width}x{height}; {frames} frames; {settings['fps']} FPS", flush=True)
    with torch.inference_mode():
        result = pipe(prompt_embeds=positive["text"], prompt_attention_mask=positive["mask"],
                      negative_prompt_embeds=negative["text"], negative_prompt_attention_mask=negative["mask"],
                      width=width, height=height, num_frames=frames, frame_rate=settings["fps"],
                      num_inference_steps=settings["steps"], guidance_scale=settings["guidance"],
                      decode_timestep=0.05, decode_noise_scale=0.025,
                      generator=torch.Generator(device="cpu").manual_seed(settings["seed"]))
    video = Path(request.get("sample_path", Path(request["output_dir"]) / "video.mp4"))
    export_to_video(result.frames[0], str(video), fps=settings["fps"])
    print(f"Saved: {video}", flush=True)
    print(f"Peak allocated VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f} GiB", flush=True)


def fingerprint(folder):
    from tools.backends.common import fingerprint as _fingerprint
    return [list(item) for item in _fingerprint(folder)]


def build_training_cache(request, settings):
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageOps
    from safetensors.torch import save_file
    import torch
    from diffusers import LTXPipeline
    from tqdm import tqdm
    from tools.backends.common import clips_in
    folder = Path(request["dataset_dir"])
    cache = Path(request["training_cache"])
    cache.mkdir(exist_ok=False)
    width, height = map(int, settings["resolution"].split("x"))
    frames = int(settings["target_frames"])
    clips = clips_in(folder)
    captions = [clip.with_suffix(".txt").read_text(encoding="utf-8-sig").strip() for clip in clips]
    encoded = encode_captions(captions)
    vae = load_vae().to("cuda")
    vae.requires_grad_(False).eval()
    paths = []
    for index, clip in enumerate(tqdm(clips, desc="cache latents")):
        if fingerprint(folder) != request["dataset_files"]:
            raise ValueError("Dataset changed while caching; start a new run.")
        reader = imageio.get_reader(str(clip))
        try:
            count = reader.count_frames()
            fps = reader.get_meta_data()["fps"]
            if count < frames or not math.isfinite(fps) or fps <= 0:
                raise ValueError(f"{clip.name}: requires at least {frames} frames and a valid FPS.")
            pixels = [np.asarray(ImageOps.fit(Image.fromarray(reader.get_data(int(i))), (width, height), method=Image.Resampling.LANCZOS))
                      for i in np.linspace(0, count - 1, frames, dtype=int)]
        finally:
            reader.close()
        pixels = torch.from_numpy(np.stack(pixels)).permute(3, 0, 1, 2).unsqueeze(0).to(device="cuda", dtype=torch.bfloat16).div(127.5).sub(1)
        with torch.no_grad():
            latent = vae.encode(pixels).latent_dist.sample()
            latent = LTXPipeline._normalize_latents(latent, vae.latents_mean, vae.latents_std, vae.config.scaling_factor)
        # The sampler spans the complete source clip; preserve that span in RoPE timing.
        effective_fps = (frames - 1) * fps / (count - 1)
        path = cache / f"clip_{index:06d}.safetensors"
        save_file({"latents": latent.cpu().contiguous(), **encoded[index]}, str(path), metadata={"fps": str(effective_fps)})
        paths.append(path)
        del pixels, latent
    del vae, encoded
    release()
    return paths


def save_adapter(model, path, settings, epoch, step):
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    tensors = {k: v.detach().float().cpu().contiguous() for k, v in get_peft_model_state_dict(model).items()}
    metadata = {"model_type": contract.MODEL_TYPE, "adapter_format": contract.ADAPTER_FORMAT,
                "base_model": contract.MODEL_ID, "revision": contract.REVISION, "checkpoint": CHECKPOINT,
                "rank": str(settings["rank"]), "alpha": str(settings["alpha"]),
                "epoch": str(epoch), "step": str(step), "target_modules": json.dumps(contract.TARGET_MODULES)}
    temporary = path.with_suffix(".tmp")
    save_file(tensors, str(temporary), metadata=metadata)
    temporary.replace(path)


def cache_training_data(request, settings):
    from tools.backends.common import cached_training_data
    return cached_training_data(request, settings, lambda: build_training_cache(request, settings))


def train(request):
    import torch
    import torch.nn.functional as F
    from diffusers import LTXPipeline
    from safetensors import safe_open
    from safetensors.torch import load_file
    from tqdm import tqdm
    settings = contract.settings_for("training", request["settings"])
    random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    torch.cuda.manual_seed_all(settings["seed"])
    cached = cache_training_data(request, settings)
    model = load_transformer()
    model.requires_grad_(False)
    if request.get("resume_from"):
        meta = contract.checkpoint_info(Path(request["resume_from"]))
        settings.update(rank=int(meta["rank"]), alpha=int(meta["alpha"]))
        model = load_lora(model, request["resume_from"])
    else:
        model = inject_lora(model, settings["rank"], settings["alpha"])
    trainable = [param for param in model.parameters() if param.requires_grad]
    if not trainable:
        raise ValueError("No trainable LoRA parameters were injected.")
    for parameter in trainable:
        parameter.data = parameter.data.float()
    if settings["gradient_checkpointing"]:
        model.enable_gradient_checkpointing()
    model.to("cuda").train()
    optimizer = torch.optim.AdamW(trainable, lr=settings["learning_rate"])
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
                with safe_open(str(path), framework="pt") as stream:
                    fps = float(stream.metadata()["fps"])
                raw = tensors["latents"]
                _, _, frames, height, width = raw.shape
                latents = LTXPipeline._pack_latents(raw, model.config.patch_size, model.config.patch_size_t)
                # Uniform flow matching: x_sigma = (1-sigma)x + sigma*noise, target = noise-x.
                # Use FP32 for the interpolation and objective, BF16 for transformer computation.
                latents = latents.float()
                noise = torch.randn_like(latents)
                sigma = torch.rand((1,), device="cuda")
                timestep = (sigma * 1000).round().long()
                noisy = (1 - sigma[:, None, None]) * latents + sigma[:, None, None] * noise
                group_size = min(accumulation, len(order) - (index // accumulation) * accumulation)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = model(hidden_states=noisy.to(torch.bfloat16), timestep=timestep,
                                       encoder_hidden_states=tensors["text"], encoder_attention_mask=tensors["mask"],
                                       num_frames=frames, height=height, width=width,
                                       rope_interpolation_scale=(8 / fps, 32, 32), return_dict=False)[0]
                    loss = F.mse_loss(prediction.float(), noise - latents)
                if not torch.isfinite(loss):
                    raise ValueError("Training loss became non-finite. Inspect clips and reduce learning rate.")
                (loss / group_size).backward()
                if (index + 1) % accumulation == 0 or index + 1 == len(order):
                    norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    if not torch.isfinite(norm):
                        raise ValueError("LoRA gradients became non-finite.")
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    progress.set_postfix(loss=f"{loss.item():.5f}", epoch=epoch)
                    progress.update(1)
                    with (output / "training_metrics.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({"step": step, "epoch": epoch, "loss": loss.item(), "gradient_norm": norm.item()}) + "\n")
            if epoch % settings["save_every"] == 0:
                save_adapter(model, output / f"{request['model_name']}_epoch{epoch:04d}.safetensors", settings, epoch, step)
            info_path = output / "model_info.json"
            info = json.loads(info_path.read_text(encoding="utf-8"))
            info.update(epochs_completed=epoch, optimizer_steps=step)
            temporary = info_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(info, indent=2), encoding="utf-8")
            temporary.replace(info_path)
    final = output / f"{request['model_name']}.safetensors"
    save_adapter(model, final, settings, request["epochs"], step)
    print(f"Saved adapter: {final}", flush=True)
    print(f"Peak allocated VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f} GiB", flush=True)
