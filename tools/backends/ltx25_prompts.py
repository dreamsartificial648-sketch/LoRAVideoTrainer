"""Staged Gemma prompt caching: never retain Gemma beside the 22B denoiser."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace


def cache_path(request, prompt):
    # Per-run directory; model revision is fixed by the worker's contract.
    digest = hashlib.sha256(prompt.strip().encode()).hexdigest()
    return Path(request.get("cache_root", Path(request["output_dir"]) / 'cache')) / "prompts" / f"{digest}.safetensors"


def encode(request):
    import torch
    from safetensors.torch import load_file, save_file
    from toolkit.config_modules import ModelConfig
    from extensions_built_in.diffusion_models.ltx2.ltx2 import LTX25Model
    from tools.backends.ltx25_config import model_config

    s = request["settings"]
    prompts = {"", "A short video with sound"}
    if request["mode"] == "train":
        prompts.update(p.read_text(encoding="utf-8").strip() for p in Path(request["prepared_dataset"]).glob("*.txt"))
    else:
        prompts.update([s["prompt"].strip(), s["negative_prompt"].strip()])
    missing = [p for p in sorted(prompts) if not cache_path(request, p).is_file()]
    if not missing:
        return
    profile = "dev" if request["mode"] == "train" or request.get("model_path") else s["profile"]
    model = LTX25Model(device="cuda:0", model_config=ModelConfig(**model_config(request["store"], s, profile)), dtype="bf16")
    path = model._resolve_te_path()
    print("Prompt stage: loading only the quantized Gemma encoder", flush=True)
    state = load_file(path)
    tokenizer = model._load_gemma4_tokenizer(path, state)
    encoder, quantized = model._load_gemma4_text_encoder(path, state)
    del state
    encoder.aitk_is_quantized = bool(quantized)
    encoder.aitk_post_load(**model.component_load_kwargs("te"))
    encoder.to(model.device_torch).eval().requires_grad_(False)
    model.text_encoder = [encoder]
    model.tokenizer = [tokenizer]
    model.pipeline = SimpleNamespace(text_encoder=encoder)
    with torch.inference_mode():
        for index, prompt in enumerate(missing, 1):
            embeds = model.get_prompt_embeds([prompt])
            dest = cache_path(request, prompt)
            dest.parent.mkdir(parents=True, exist_ok=True)
            temporary = dest.with_suffix(".tmp")
            save_file({"text": embeds.text_embeds.detach().cpu().contiguous(), "mask": embeds.attention_mask.detach().cpu().contiguous()}, str(temporary))
            temporary.replace(dest)
            print(f"Prompt cache: {index}/{len(missing)}", flush=True)
    # This entire process exits before the parent loads the transformer.


def install_cached_encoder(request):
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    from toolkit.prompt_utils import PromptEmbeds
    from extensions_built_in.diffusion_models.ltx2.ltx2 import LTX25Model

    class CachedEncoder(torch.nn.Module):
        """Placement placeholder; actual features must come from the first stage."""
        def __init__(self):
            super().__init__()
            # Accelerate's component offload hook inspects the first parameter.
            self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16), requires_grad=False)
            self.config = SimpleNamespace(use_cache=False)

        @property
        def device(self):
            return self.anchor.device

        @property
        def dtype(self):
            return self.anchor.dtype

        def aitk_post_load(self, **kwargs):
            return self

        def forward(self, *args, **kwargs):
            raise RuntimeError("Prompt cache was bypassed; refusing to load an extra text encoder.")

    def skip_encoder(self, path, state):
        return CachedEncoder(), 0

    def prompt_embeds(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        tensors = []
        for prompt in prompts:
            path = cache_path(request, prompt)
            if not path.is_file():
                raise ValueError(f"Caption changed after prompt preparation; no cached embedding for {prompt[:80]!r}. Start a fresh run.")
            tensors.append(load_file(str(path)))
        length = max(t["text"].shape[1] for t in tensors)
        result = PromptEmbeds([torch.cat([F.pad(t["text"], (0, 0, length - t["text"].shape[1], 0)) for t in tensors]), None])
        result.attention_mask = torch.cat([F.pad(t["mask"], (length - t["mask"].shape[1], 0)) for t in tensors])
        return result

    LTX25Model._load_gemma4_text_encoder = skip_encoder
    LTX25Model.get_prompt_embeds = prompt_embeds
