"""Exercise joint LoRA gradients, checkpoint round-trip and staged prompt lookup without base downloads."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runtimes/ltx25/upstream"))

import torch
from safetensors.torch import save_file
from toolkit.config_modules import ModelConfig, NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.models.v2.diffusion_models.ltx2 import LTX2VideoTransformer3DModel
from extensions_built_in.diffusion_models.ltx2.ltx2 import LTX25Model
from tools.backends import ltx25_contract as ltx25
from tools.backends import ltx25_media as av_media
from tools.backends.ltx25_worker import attach_lora, install_pcm_decoder, sampling_scheduler, slice_prompt_connectors
from tools.backends.ltx25_prompts import cache_path, install_cached_encoder


def main():
    folder = ROOT / "logs/ltx25-runtime-check"
    folder.mkdir(parents=True, exist_ok=True)
    model = LTX25Model(device="cuda:0", model_config=ModelConfig(arch="ltx2.5", name_or_path="test"), dtype="bf16")
    from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
    schedule = sampling_scheduler(model.get_train_scheduler().config, "distilled")
    schedule.set_timesteps(sigmas=DISTILLED_SIGMA_VALUES)
    assert len(schedule.timesteps) == 8 and torch.allclose(schedule.sigmas[:-1], torch.tensor(DISTILLED_SIGMA_VALUES))
    torch.manual_seed(42)
    tiny = LTX2VideoTransformer3DModel(in_channels=4, out_channels=4, num_attention_heads=2, attention_head_dim=16,
        cross_attention_dim=32, audio_in_channels=4, audio_out_channels=4, audio_num_attention_heads=2,
        audio_attention_head_dim=16, audio_cross_attention_dim=32, caption_channels=32, num_layers=1)
    model.model = tiny.cuda().to(torch.bfloat16).requires_grad_(False)
    model.text_encoder = None
    config = NetworkConfig(type="lora", linear=4, linear_alpha=4)
    net = LoRASpecialNetwork(text_encoder=None, unet=model.model, lora_dim=4, alpha=4,
        train_unet=True, train_text_encoder=False, is_transformer=True, base_model=model,
        network_config=config, transformer_only=config.transformer_only, target_lin_modules=model.target_lora_modules)
    net.apply_to(None, model.model, False, True)
    net.force_to(torch.device("cuda:0"), dtype=torch.float32)
    net.is_active = True
    net._update_torch_multiplier()
    model.network = net
    block = model.model.transformer_blocks[0]
    video_layer = block.attn1.to_q
    audio_layer = block.audio_attn1.to_q
    x = torch.randn(1, 8, 32, device="cuda", dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = video_layer(x).float().square().mean() + audio_layer(x).float().square().mean()
    loss.backward()
    audio_grads = [p.grad for n, p in net.named_parameters() if "audio" in n and p.grad is not None]
    assert audio_grads and any(g.abs().sum() > 0 for g in audio_grads), "Audio LoRA did not receive gradients"
    optimizer.step()
    torch.set_grad_enabled(False)
    expected = audio_layer(x).float().clone()
    path = folder / "joint-test.safetensors"
    net.save_weights(str(path), dtype=torch.float32, metadata={"model_type": ltx25.MODEL_TYPE,
        "base_model": ltx25.MODEL_ID, "revision": ltx25.REVISION, "base_profile": "dev", "rank": "4", "alpha": "4"})
    ltx25.checkpoint_info(path)
    # Attach the same saved adapter to a fresh network with identical base weights.
    torch.manual_seed(42)
    tiny2 = LTX2VideoTransformer3DModel.from_config(tiny.config).cuda().to(torch.bfloat16).requires_grad_(False)
    model.model = tiny2
    model.network = None
    attach_lora(model, path, {"lora_strength": 1.})
    actual = tiny2.transformer_blocks[0].audio_attn1.to_q(x).float()
    assert torch.allclose(expected, actual, atol=.01, rtol=.01), "LoRA checkpoint did not round-trip"
    request = {"output_dir": str(folder)}
    for prompt, length in (("one", 2), ("two", 4)):
        dest = cache_path(request, prompt)
        dest.parent.mkdir(parents=True, exist_ok=True)
        save_file({"text": torch.ones(1, length, 8), "mask": torch.ones(1, length, dtype=torch.long)}, str(dest))
    install_cached_encoder(request)
    embeds = model.get_prompt_embeds(["one", "two"])
    assert embeds.text_embeds.shape == (2, 4, 8)
    assert embeds.attention_mask[0].tolist() == [0, 0, 1, 1]
    from types import SimpleNamespace
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
    connectors = LTX2TextConnectors(caption_channels=32, text_proj_in_factor=2, video_connector_num_attention_heads=2,
        video_connector_attention_head_dim=16, video_connector_num_layers=1, video_connector_num_learnable_registers=4,
        audio_connector_num_attention_heads=2, audio_connector_attention_head_dim=16, audio_connector_num_layers=1,
        audio_connector_num_learnable_registers=4, per_modality_projections=True, video_hidden_dim=32, audio_hidden_dim=32).eval()
    states = torch.randn(2, 16, 64)
    mask = torch.ones(2, 16, dtype=torch.long)
    reference = connectors(states, mask)
    slice_prompt_connectors(SimpleNamespace(connectors=connectors))
    sliced = connectors(states, mask)
    assert all(torch.allclose(a, b, atol=1e-5, rtol=1e-5) for a, b in zip(reference, sliced))
    media = folder / "pcm-test.mkv"
    av_media.command([av_media.ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=s=256x256:r=24:d=1",
                      "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1", "-c:v", "libx264", "-c:a", "pcm_f32le", str(media)])
    install_pcm_decoder()
    import torchaudio
    waveform, rate = torchaudio.load(str(media))
    assert rate == 48000 and waveform.shape == (1, 48000) and waveform.abs().max() > .1
    from diffusers.utils import export_to_video
    import numpy as np
    export_to_video(np.zeros((9, 256, 256, 3), dtype=np.float32), str(folder / "export-test.mp4"), fps=24)
    assert av_media.probe(folder / "export-test.mp4")["fps"] == 24
    result = {"audio_gradients": True, "lora_round_trip": True, "cached_prompt_lookup": True,
              "pcm_decoder": True, "video_export": True, "loss": float(loss), "adapter_bytes": path.stat().st_size, "gpu": torch.cuda.get_device_name()}
    (folder / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
