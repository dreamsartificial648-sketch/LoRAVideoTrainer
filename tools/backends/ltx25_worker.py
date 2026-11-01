"""Pinned AI Toolkit bridge. Runs only inside runtimes/ltx25/.venv."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import subprocess
import importlib
import time
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class Metrics:
    def __init__(self, output, stage):
        import psutil
        self.psutil, self.output, self.stage = psutil, Path(output), stage
        self.start = time.perf_counter()
        self.peak_rss = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()

    def sample(self):
        process = self.psutil.Process()
        while not self.stop.is_set():
            self.peak_rss = max(self.peak_rss, process.memory_info().rss)
            self.stop.wait(.25)

    def save(self):
        import torch
        self.stop.set()
        self.thread.join(timeout=1)
        info = {"elapsed_seconds": time.perf_counter() - self.start, "peak_rss_bytes": self.peak_rss,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()}
        (self.output / f"metrics_{self.stage}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def configure(request):
    os.environ["MODELS_PATH"] = str(Path(request["store"]) / "models")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("AI_TOOLKIT_OFFLOAD_DEPTH", "2")
    upstream = Path(request["upstream"])
    sys.path.insert(0, str(upstream))
    # Some optional extensions use paths relative to the Toolkit checkout.
    os.chdir(upstream)
    if os.environ.get("LORAVIDEO_LTX25_TRACE") == "1":
        import faulthandler
        faulthandler.dump_traceback_later(45, repeat=True)


def configure_host_memory(settings):
    if settings.get("pin_host_memory", False):
        return
    import torch
    from toolkit.memory_management import manager_modules
    original = manager_modules.OstrisLinearLayerMemoryManager.__init__
    def initialize(self, module, manager):
        # Scope the override to synchronous component attachment in this isolated worker.
        # Keep mmap-backed quantized buffers; pin_memory() would copy the entire model.
        pin = torch.Tensor.pin_memory
        try:
            torch.Tensor.pin_memory = lambda tensor, *args, **kwargs: tensor
            original(self, module, manager)
        finally:
            torch.Tensor.pin_memory = pin
    manager_modules.OstrisLinearLayerMemoryManager.__init__ = initialize
    manager_modules._ensure_cpu_pinned = lambda tensor: tensor.to("cpu") if tensor is not None else None
    print("Memory mode: pageable CPU weights; large pinned copies disabled", flush=True)


def install_pcm_decoder():
    """Decode our normalized PCM training masters without TorchCodec/FFmpeg DLLs."""
    import av
    import numpy as np
    import torch
    import torchaudio
    original = torchaudio.load

    def load(path, *args, **kwargs):
        if Path(path).suffix.lower() != ".mkv":
            return original(path, *args, **kwargs)
        with av.open(str(path)) as container:
            stream = container.streams.audio[0]
            rate = stream.codec_context.sample_rate
            layout = stream.codec_context.layout.name
            resampler = av.AudioResampler(format="fltp", layout=layout, rate=rate)
            parts = []
            for frame in container.decode(stream):
                parts.extend(f.to_ndarray() for f in resampler.resample(frame))
            parts.extend(f.to_ndarray() for f in resampler.resample(None))
            if not parts:
                raise ValueError(f"No audio samples in {Path(path).name}")
            data = np.concatenate(parts, axis=1)
        offset = int(kwargs.get("frame_offset", 0))
        length = int(kwargs.get("num_frames", -1))
        data = data[:, offset:offset + length if length >= 0 else None]
        tensor = torch.from_numpy(data.copy())
        return (tensor if kwargs.get("channels_first", True) else tensor.T), rate

    torchaudio.load = load


def train(request):
    from tools.backends.ltx25_config import training_config
    from toolkit.job import get_job
    process_module = importlib.import_module("jobs.process.BaseSDTrainProcess")
    original_bar = process_module.ToolkitProgressBar
    class StepsBar(original_bar):
        def __init__(self, *args, **kwargs):
            kwargs["desc"] = "steps"
            super().__init__(*args, **kwargs)
    process_module.ToolkitProgressBar = StepsBar
    install_pcm_decoder()
    config = training_config(request)
    path = Path(request["output_dir"]) / "training_config.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    job = get_job(config)
    trainer = job.process[0]
    from tools.backends import ltx25_contract as contract
    trainer.meta.update(model_type=contract.MODEL_TYPE, base_model=contract.MODEL_ID,
                        revision=contract.REVISION, base_profile='dev', rank=str(request['settings']['rank']),
                        alpha=str(request['settings']['alpha']), train_audio=str(request['settings']['train_audio']).lower())
    class CachePrepared(Exception):
        pass
    before_loop = trainer.hook_before_train_loop
    def before_training():
        if request.get('cache_only'):
            raise CachePrepared()
        before_loop()
        save_weights = trainer.network.save_weights
        def atomic_save(file, *args, **kwargs):
            target = Path(file)
            temporary = target.with_name(target.stem + '.tmp.safetensors')
            save_weights(str(temporary), *args, **kwargs)
            contract.checkpoint_info(temporary)
            temporary.replace(target)
        trainer.network.save_weights = atomic_save
    trainer.hook_before_train_loop = before_training
    losses = {}
    calculate_loss = trainer.calculate_loss
    def capture_loss(*args, **kwargs):
        loss = calculate_loss(*args, **kwargs)
        losses.update({key: float(value) for key, value in trainer.additional_logs.items() if key in ("loss/img", "loss/audio")})
        return loss
    trainer.calculate_loss = capture_loss
    try:
        job.run()
        import math
        if 'loss/img' not in losses:
            raise RuntimeError('Training finished without reporting a video loss; refusing to mark it successful.')
        if request["settings"]["train_audio"] and "loss/audio" not in losses:
            raise RuntimeError("Joint training finished without reporting an audio loss; refusing to mark this as an AV training success.")
        if any(not math.isfinite(value) for value in losses.values()):
            raise RuntimeError("Training reported a non-finite loss. Review the run before using its checkpoints.")
        (Path(request["output_dir"]) / "training_result.json").write_text(json.dumps({"last_losses": losses, "step": trainer.step_num}, indent=2), encoding="utf-8")
    except CachePrepared:
        print('Video/audio latents and text embeddings cached. No optimizer steps executed.', flush=True)
    finally:
        job.cleanup()


def attach_lora(model, path, settings):
    """Use the same LoRA network and key conversion as the trainer."""
    import torch
    from tools.backends.ltx25_contract import checkpoint_info
    from toolkit.config_modules import NetworkConfig
    from toolkit.lora_special import LoRASpecialNetwork
    meta = checkpoint_info(path)
    config = NetworkConfig(type="lora", linear=int(meta["rank"]), linear_alpha=int(meta["alpha"]))
    net = LoRASpecialNetwork(text_encoder=model.text_encoder, unet=model.model,
                            lora_dim=config.linear, alpha=config.linear_alpha, multiplier=settings["lora_strength"],
                            train_unet=True, train_text_encoder=False, is_transformer=True, base_model=model,
                            network_config=config, transformer_only=config.transformer_only, target_lin_modules=model.target_lora_modules)
    net.apply_to(model.text_encoder, model.model, False, True)
    extras = net.load_weights(str(path))
    if extras:
        raise ValueError("LoRA has tensors that do not match the LTX-2.5 network.")
    net.force_to(torch.device("cuda:0"), dtype=torch.float32)
    net.can_merge_in = False
    net.eval()
    net.is_active = True
    model.network = net
    net._update_torch_multiplier()
    return net


def sampling_scheduler(config, profile):
    from diffusers import FlowMatchEulerDiscreteScheduler
    overrides = {"use_dynamic_shifting": profile == "dev"}
    if profile == "distilled":
        # Do not inherit the dev scheduler's terminal rescaling: these sigmas are fixed.
        overrides.update(shift=1., shift_terminal=None)
    return FlowMatchEulerDiscreteScheduler.from_config(config, **overrides)


def slice_prompt_connectors(pipeline):
    """Encode guidance's positive/negative prompts separately to bound temporary buffers."""
    import torch
    original = pipeline.connectors.forward
    def forward(text_encoder_hidden_states, attention_mask, **kwargs):
        if text_encoder_hidden_states.shape[0] <= 1:
            return original(text_encoder_hidden_states, attention_mask, **kwargs)
        outputs = [original(text_encoder_hidden_states[i:i + 1], attention_mask[i:i + 1], **kwargs)
                   for i in range(text_encoder_hidden_states.shape[0])]
        return tuple(torch.cat([output[i] for output in outputs], dim=0) for i in range(len(outputs[0])))
    pipeline.connectors.forward = forward


def generate(request):
    import numpy as np
    import soundfile as sf
    import torch
    from tools.backends import ltx25_media as av_media, ltx25_contract as ltx25
    from tools.backends.ltx25_config import model_config
    from toolkit.config_modules import ModelConfig
    from extensions_built_in.diffusion_models.ltx2.ltx2 import LTX25Model
    from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
    from diffusers.utils import export_to_video

    s = request["settings"]
    profile = "dev" if request.get("model_path") else s["profile"]
    torch.set_grad_enabled(False)
    model = LTX25Model(device="cuda:0", model_config=ModelConfig(**model_config(request["store"], s, profile)), dtype="bf16")
    print(f"Loading INT8 LTX-2.5 {profile}, with layer offloading", flush=True)
    model.load_model()
    if os.environ.get("LORAVIDEO_LTX25_TRACE") == "1":
        import faulthandler
        faulthandler.cancel_dump_traceback_later()
    if request.get("model_path"):
        attach_lora(model, request["model_path"], s)
    print("Encoding video/audio prompt", flush=True)
    positive = model.pad_embeds(model.get_prompt_embeds([s["prompt"]]))
    negative = model.pad_embeds(model.get_prompt_embeds([s["negative_prompt"] or ""])) if profile == "dev" else None
    # Build via the Toolkit's memory-managed components. Their .to() honors layer offloading.
    pipeline = model.get_generation_pipeline()
    pipeline.set_progress_bar_config(desc="steps")
    model.text_encoder_to("cpu")
    slice_prompt_connectors(pipeline)
    # Keep the VAEs/vocoder out of VRAM during denoising, and park the large
    # prompt connectors after their single pass. The transformer's custom .to()
    # continues to honor its per-layer offloading manager.
    pipeline.enable_model_cpu_offload(gpu_id=0)
    pipeline.vae.enable_tiling()
    pipeline.vae.tile_sample_min_num_frames = 16
    pipeline.vae.tile_sample_stride_num_frames = 8
    pipeline.vae.use_framewise_decoding = True
    gc.collect()
    torch.cuda.empty_cache()
    pipeline.scheduler = sampling_scheduler(pipeline.scheduler.config, profile)
    width, height = map(int, s["format"].split("x"))
    frames = ltx25.frames_for_seconds(s["duration"], 24)
    kwargs = {"prompt_embeds": positive.text_embeds.to("cuda:0", dtype=torch.bfloat16),
              "prompt_attention_mask": positive.attention_mask.to("cuda:0"),
              "height": height, "width": width, "num_frames": frames, "frame_rate": 24,
              "generator": torch.Generator(device="cuda:0").manual_seed(s["seed"]),
              "output_type": "np", "return_dict": False}
    if profile == "distilled":
        kwargs.update(sigmas=DISTILLED_SIGMA_VALUES, guidance_scale=1., audio_guidance_scale=1.,
                      stg_scale=0., audio_stg_scale=0., modality_scale=1., audio_modality_scale=1., guidance_rescale=0., audio_guidance_rescale=0.)
    else:
        kwargs.update(num_inference_steps=s["steps"], guidance_scale=s["guidance"], audio_guidance_scale=7.,
                      stg_scale=1., audio_stg_scale=1., modality_scale=3., audio_modality_scale=3.,
                      guidance_rescale=.7, audio_guidance_rescale=.7, spatio_temporal_guidance_blocks=[28], use_cross_timestep=True,
                      negative_prompt_embeds=negative.text_embeds.to("cuda:0", dtype=torch.bfloat16),
                      negative_prompt_attention_mask=negative.attention_mask.to("cuda:0"))
    print("Denoising synchronized video and audio", flush=True)
    video, audio = pipeline(**kwargs)
    output = Path(request["output_dir"])
    export_to_video(video[0], str(output / "silent.mp4"), fps=24)
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().float().cpu().numpy()
    audio = np.asarray(audio)
    if audio.ndim == 3:
        audio = audio[0]
    if audio.ndim == 2 and audio.shape[0] <= 8:
        audio = audio.T
    rate = int(pipeline.vocoder.config.output_sampling_rate)
    samples = round(frames / 24 * rate)
    audio = audio[:samples]
    if len(audio) < samples:
        audio = np.pad(audio, ((0, samples - len(audio)), (0, 0)) if audio.ndim == 2 else (0, samples - len(audio)))
    if not np.isfinite(audio).all():
        raise ValueError("Generated waveform contains non-finite samples.")
    sf.write(str(output / "audio.wav"), audio, rate, subtype="PCM_24")
    av_media.mux(output / "silent.mp4", output / "audio.wav", output / "video.mp4", frames / 24)
    (output / "silent.mp4").unlink()
    print("Saved video.mp4 and audio.wav", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--encode-only", action="store_true")
    args = parser.parse_args()
    request_path = args.request.resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    from tools.backends.ltx25_contract import require_hardware
    require_hardware()
    configure(request)
    configure_host_memory(request["settings"])
    from tools.backends.ltx25_prompts import encode, install_cached_encoder
    if args.encode_only:
        metrics = Metrics(request["output_dir"], "encoder")
        try:
            encode(request)
        finally:
            metrics.save()
        return
    # A process boundary releases pageable and pinned host allocations on Windows.
    subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--request", str(request_path), "--encode-only"], check=True)
    install_cached_encoder(request)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("LTX-2.5 needs the isolated CUDA runtime and an NVIDIA GPU.")
    metrics = Metrics(request["output_dir"], "model")
    try:
        if request["mode"] == "train":
            train(request)
        elif request["mode"] == "generate":
            generate(request)
        else:
            raise ValueError("Unsupported LTX-2.5 worker operation.")
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError("LTX-2.5 exceeded available VRAM. Try 256×256, 17 frames, rank 4 and 100% offloading; close other GPU workloads. Joint AV training on 12 GB remains experimental.") from exc
    finally:
        metrics.save()


if __name__ == "__main__":
    main()
