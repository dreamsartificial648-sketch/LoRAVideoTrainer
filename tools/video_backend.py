"""Standalone CLI used by the desktop app's isolated video environments."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ('modelscope_video', 'ltx_video')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=BACKENDS, required=True)
    parser.add_argument('--mode', choices=('validate', 'cache', 'train', 'generate', 'download', 'check'), required=True)
    parser.add_argument('--settings', default='{}', help='JSON settings for the selected operation')
    parser.add_argument('--dataset', type=Path, default=ROOT / 'dataset/videos')
    parser.add_argument('--name', default='my_video_lora')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--resume', default='')
    parser.add_argument('--checkpoint', default='')
    args = parser.parse_args()
    contract = importlib.import_module(f'tools.backends.{args.backend}_contract')
    os.environ['HF_HOME'] = str(ROOT / 'models' / args.backend)
    os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
    if args.mode == 'check':
        import torch, diffusers, peft, transformers
        print(f'{args.backend}: torch {torch.__version__}, diffusers {diffusers.__version__}, PEFT {peft.__version__}, transformers {transformers.__version__}')
        require_cuda(args.backend)
        print(f'GPU: {torch.cuda.get_device_name(0)}; setup check passed.')
        return
    worker = importlib.import_module(f'tools.backends.{args.backend}')
    if args.mode == 'download':
        download(args.backend, contract, worker)
        return
    settings = contract.settings_for('generation' if args.mode == 'generate' else 'training', json.loads(args.settings))
    request = {'backend': args.backend, 'base_model': contract.MODEL_ID, 'revision': contract.REVISION,
               'settings': settings, 'cache_dir': str(ROOT / 'cache'),
               'training_cache': str(ROOT / 'cache' / args.backend), 'dataset_dir': str(args.dataset.resolve())}
    if args.mode == 'generate':
        if args.checkpoint:
            contract.checkpoint_info(Path(args.checkpoint))
        require_cuda(args.backend)
        # A unique filename preserves the sample browser and storyboard's flat MP4 layout.
        sample = ROOT / 'samples' / f'{args.backend}_{datetime.now():%Y%m%d_%H%M%S_%f}.mp4'
        sample.parent.mkdir(exist_ok=True)
        request.update(model_path=args.checkpoint, output_dir=str(sample.parent), sample_path=str(sample))
        worker.generate(request)
        return
    from tools.backends.common import validate_dataset
    request['dataset_files'] = validate_dataset(args.dataset, settings['trigger_word'], int(settings['target_frames']))
    if args.mode == 'validate':
        print('Dataset validation passed.')
        return
    require_cuda(args.backend)
    if args.mode == 'cache':
        import torch
        torch.manual_seed(settings['seed'])
        torch.cuda.manual_seed_all(settings['seed'])
        if args.backend == 'modelscope_video':
            worker.cache_training_data(worker.load_pipeline(), request, settings)
        else:
            worker.cache_training_data(request, settings)
        return
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.name) or args.name.endswith('.'):
        raise ValueError('Use a run name containing letters, numbers, dots, underscores or hyphens.')
    if args.name.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}:
        raise ValueError('Choose a run name that is not a Windows reserved filename.')
    if not 1 <= args.epochs <= 100000:
        raise ValueError('Epochs must be between 1 and 100000.')
    if args.resume:
        contract.checkpoint_info(Path(args.resume))
    output = ROOT / 'output' / args.backend / args.name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    request.update(output_dir=str(output), model_name=args.name, epochs=args.epochs, resume_from=args.resume)
    (output / 'model_info.json').write_text(json.dumps(request, indent=2), encoding='utf-8')
    worker.train(request)


def require_cuda(backend):
    import torch
    if not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable. Install the isolated backend with setup_video_backends.ps1.')
    if backend == 'ltx_video' and not torch.cuda.is_bf16_supported():
        raise ValueError('LTX-Video requires a CUDA GPU supporting BF16.')


def download(backend, contract, worker):
    from huggingface_hub import hf_hub_download, snapshot_download
    if backend == 'modelscope_video':
        snapshot_download(contract.MODEL_ID, revision=contract.REVISION,
                          allow_patterns=['model_index.json', 'scheduler/*', 'tokenizer/*', 'text_encoder/config.json',
                                          'text_encoder/*.fp16.safetensors', 'unet/config.json', 'unet/*.fp16.safetensors',
                                          'vae/config.json', 'vae/*.fp16.safetensors'])
    else:
        hf_hub_download(contract.MODEL_ID, contract.CHECKPOINT, revision=contract.REVISION)
        snapshot_download(contract.MODEL_ID, revision=contract.REVISION,
                          allow_patterns=['tokenizer/*', 'text_encoder/*'])
        snapshot_download(worker.VAE_REPO, revision=worker.VAE_REVISION, allow_patterns=['vae/*', 'transformer/config.json'])
        snapshot_download(worker.SCHEDULER_REPO, revision=worker.SCHEDULER_REVISION, allow_patterns=['scheduler/*'])
    print('Model files downloaded to the local models folder.')


if __name__ == '__main__':
    main()
