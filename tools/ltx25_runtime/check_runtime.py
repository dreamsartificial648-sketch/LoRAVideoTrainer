"""Verify actual CUDA/INT8/BF16 and optimizer operations before marking setup ready."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.backends.ltx25_contract import UPSTREAM_REVISION, require_hardware, runtime_root


def main():
    hardware = require_hardware()
    runtime = runtime_root()
    upstream = runtime / 'upstream'
    revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise RuntimeError('Unexpected AI Toolkit revision; rerun the backend installer.')
    for file, marker in [('toolkit/memory_management/manager.py', 'ADAM_IDENTITY_MEMBERSHIP_V1'),
                         ('toolkit/lora_special.py', 'ADAM_SKIP_UNUSED_FULL_WEIGHT_PROBE_V1')]:
        if marker not in (upstream / file).read_text(encoding='utf-8'):
            raise RuntimeError('Required runtime patch is missing; rerun the backend installer.')
    sys.path.insert(0, str(upstream))
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
    import torch
    import torchaudio
    import av
    import bitsandbytes as bnb
    import diffusers
    import transformers
    from extensions_built_in.diffusion_models.ltx2.ltx2 import LTX25Model
    from toolkit.job import get_job
    from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
    from diffusers import LTX2Pipeline
    from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES
    left = torch.ones((32, 32), device='cuda', dtype=torch.int8)
    if torch._int_mm(left, left).shape != (32, 32):
        raise RuntimeError('CUDA INT8 matrix multiplication failed.')
    bf = torch.ones((32, 32), device='cuda', dtype=torch.bfloat16)
    if not torch.isfinite(bf @ bf).all():
        raise RuntimeError('CUDA BF16 matrix multiplication failed.')
    p = torch.nn.Parameter(torch.ones(4096, device='cuda'))
    opt = bnb.optim.AdamW8bit([p], lr=.0001)
    p.square().mean().backward()
    opt.step()
    info = {'status': 'imports_and_cuda_passed', 'upstream_revision': revision, 'hardware': hardware,
            'torch': torch.__version__, 'diffusers': diffusers.__version__, 'transformers': transformers.__version__}
    (runtime / 'runtime_ready.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps(info, indent=2), flush=True)


if __name__ == '__main__':
    main()
