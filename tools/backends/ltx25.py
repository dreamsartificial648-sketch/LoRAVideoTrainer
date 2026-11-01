"""Standalone orchestration for staged, quantized LTX-2.5 training and generation."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from tools.backends import ltx25_contract as contract, ltx25_media as media
from tools.backends.common import clips_in

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temporary.replace(path)


def validate_dataset(folder, settings):
    rows = []
    for clip in clips_in(folder):
        # Archived originals must never silently re-enter training.
        if '_source_archive' in clip.relative_to(folder).parts:
            continue
        caption_file = clip.with_suffix('.txt')
        if not caption_file.is_file():
            raise ValueError(f'{clip.name}: add a .txt caption including the trigger word.')
        caption = caption_file.read_text(encoding='utf-8-sig').strip()
        trigger = settings['trigger_word']
        if not caption or not re.search(rf'(?<!\w){re.escape(trigger)}(?!\w)', caption):
            raise ValueError(f'{clip.name}: caption must include the exact trigger {trigger!r}.')
        info = media.probe(clip)
        duration = int(settings['target_frames']) / 24
        if not info['video'] or info['duration'] + .01 < duration:
            raise ValueError(f'{clip.name}: needs at least {duration:.3f} seconds of video at the selected 24 FPS timeline.')
        if settings['train_audio']:
            if not info['audio']:
                raise ValueError(f'{clip.name}: joint audio training requires a soundtrack. '
                                 'Re-split the original with LTX-2.5 selected, or turn off Train synchronized audio.')
            audio_duration = float(info['audio'].get('duration', info['duration']))
            if audio_duration + .05 < duration:
                raise ValueError(f'{clip.name}: soundtrack is shorter than the selected training interval.')
        rows.append({'video': str(clip.resolve()), 'caption': caption})
    if not rows:
        raise ValueError('The dataset has no captioned video clips.')
    return rows


def segment(source, target, start, duration, width, height, audio=True):
    args = [media.ffmpeg(), '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
            '-ss', str(start), '-i', str(source), '-t', str(duration), '-map', '0:v:0']
    if audio:
        args += ['-map', '0:a:0', '-af', 'aresample=async=1:first_pts=0,apad', '-c:a', 'pcm_f32le']
    else:
        args += ['-an']
    args += ['-vf', f'setpts=PTS-STARTPTS,fps=24,scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}',
             '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', str(target)]
    media.command(args)


def prepare_dataset(folder, settings, rows):
    sources = [{**row, 'sha256': media.fingerprint(row['video'])} for row in rows]
    identity = {'sources': sources, 'resolution': settings['resolution'], 'frames': settings['target_frames'],
                'audio': settings['train_audio'], 'seed': settings['seed'], 'revision': contract.REVISION, 'version': 1}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = ROOT / 'cache/ltx25' / digest
    manifest = cache / 'prepared.json'
    prepared = cache / 'prepared'
    expected = [prepared / f'clip_{index:05d}.mkv' for index in range(len(rows))]
    if manifest.is_file() and all(p.is_file() and p.with_suffix('.txt').is_file() for p in expected):
        if json.loads(manifest.read_text(encoding='utf-8')) == identity:
            print(f'Reusing {len(rows)} synchronized clips and their Toolkit caches.', flush=True)
            return cache, prepared
    prepared.mkdir(parents=True, exist_ok=True)
    width, height = map(int, settings['resolution'].split('x'))
    for index, (row, target) in enumerate(zip(rows, expected), 1):
        print(f'Prepare synchronized clip {index}/{len(rows)}', flush=True)
        segment(row['video'], target, 0, int(settings['target_frames']) / 24, width, height, settings['train_audio'])
        if media.fingerprint(row['video']) != sources[index - 1]['sha256']:
            raise ValueError('A source video changed during preparation. Retry with a stable dataset.')
        if Path(row['video']).with_suffix('.txt').read_text(encoding='utf-8-sig').strip() != row['caption']:
            raise ValueError('A caption changed during preparation. Retry with a stable dataset.')
        target.with_suffix('.txt').write_text(row['caption'] + '\n', encoding='utf-8')
    write_json(manifest, identity)
    return cache, prepared


def require_runtime():
    runtime = contract.runtime_root()
    ready = runtime / 'runtime_ready.json'
    if not ready.is_file():
        raise ValueError('Use Install Model Backend for LTX-2.5, then Check Setup.')
    info = json.loads(ready.read_text(encoding='utf-8'))
    if info.get('status') != 'imports_and_cuda_passed' or info.get('upstream_revision') != contract.UPSTREAM_REVISION:
        raise ValueError('LTX-2.5 setup is stale. Reinstall this backend.')
    return runtime


def require_weights(profile):
    store = contract.store_root()
    missing = [contract.FILES[key] for key in (profile, 'text_encoder', 'video_vae', 'audio_vae')
               if not (store / 'models' / contract.FILES[key]).is_file()]
    if missing:
        raise ValueError('LTX-2.5 weights are missing. Use Model Access / Login, accept the official model terms, '
                         'then Download Model Files. Missing: ' + ', '.join(missing))


def run_worker(request, output):
    request.update(output_dir=str(output), store=str(contract.store_root()),
                   upstream=str(contract.runtime_root() / 'upstream'))
    write_json(output / 'request.json', request)
    subprocess.run([sys.executable, '-u', str(Path(__file__).with_name('ltx25_worker.py')),
                    '--request', str(output / 'request.json')], check=True)


def login():
    import tkinter as tk
    import webbrowser
    from tkinter import simpledialog
    from huggingface_hub import login as hf_login
    webbrowser.open('https://huggingface.co/Lightricks/LTX-2.5')
    ui = tk.Tk()
    ui.withdraw()
    try:
        token = simpledialog.askstring('LTX-2.5 model access',
            'Accept access and the terms on the official model page.\n'
            'Paste a Hugging Face read token. It is saved only in the local Hugging Face credential store.',
            show='*', parent=ui)
        if not token:
            print('Login cancelled; no credentials changed.', flush=True)
            return
        # Do not expose authentication exception text, which may contain request details.
        try:
            hf_login(token=token.strip(), add_to_git_credential=False)
        except Exception:
            raise ValueError('Hugging Face login failed. Check the token and model access on the official page.') from None
    finally:
        ui.destroy()
    print('Local Hugging Face login complete. Credentials are never added to app settings or requests.', flush=True)


def run(args):
    if args.mode == 'login':
        login()
        return
    if args.mode == 'check':
        from tools.ltx25_runtime.check_runtime import main
        main()
        return
    # Validation is a CPU operation. Every model operation checks hardware before model loading/download.
    if args.mode != 'validate':
        contract.require_hardware()
        require_runtime()
    if args.mode == 'download':
        from tools.ltx25_runtime.download_models import main
        if main(['--profile', 'all']):
            raise ValueError('Official model access/download failed; see the instructions above.')
        return
    settings = contract.settings_for('generation' if args.mode == 'generate' else 'training', json.loads(args.settings))
    if args.mode == 'generate':
        generate(args, settings)
        return
    folder = args.dataset.resolve()
    rows = validate_dataset(folder, settings)
    if args.mode == 'validate':
        print(f'Validated {len(rows)} clips at 24 FPS; synchronized audio: {settings["train_audio"]}.', flush=True)
        return
    if args.resume:
        meta = contract.checkpoint_info(Path(args.resume))
        settings.update(rank=int(meta['rank']), alpha=int(meta['alpha']))
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.name) or args.name.endswith('.'):
        raise ValueError('Use a run name containing letters, numbers, dots, underscores or hyphens.')
    if args.name.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}:
        raise ValueError('Choose a run name that is not a Windows reserved filename.')
    if not 1 <= args.epochs <= 100000:
        raise ValueError('Epochs must be between 1 and 100000.')
    require_weights('dev')
    output = ROOT / 'output/ltx25' / args.name
    if args.mode == 'train' and output.exists():
        raise ValueError('That run folder already exists; choose a new run name.')
    cache, prepared = prepare_dataset(folder, settings, rows)
    if args.mode == 'cache':
        output = cache / 'cache_job'
        output.mkdir(exist_ok=True)
    else:
        output.mkdir(parents=True, exist_ok=False)
    request = {'mode': 'train', 'cache_only': args.mode == 'cache', 'cache_root': str(cache), 'settings': settings,
               'model_name': args.name, 'epochs': args.epochs, 'resume_from': args.resume,
               'clip_count': len(rows), 'prepared_dataset': str(prepared)}
    metadata = {'status': 'running', 'base_model': contract.MODEL_ID, 'revision': contract.REVISION,
                'settings': settings, 'dataset_dir': str(folder), 'clip_count': len(rows),
                'resume_from': args.resume, 'continuation_mode': 'weights_only' if args.resume else 'fresh'}
    write_json(output / 'model_info.json', metadata)
    try:
        run_worker(request, output)
        if args.mode == 'train':
            checkpoints = sorted((output / 'checkpoints').rglob('*.safetensors'), key=lambda p: p.stat().st_mtime)
            if not checkpoints:
                raise RuntimeError('Training finished without saving an adapter.')
            for checkpoint in checkpoints:
                contract.checkpoint_info(checkpoint)
            metadata.update(json.loads((output / 'training_result.json').read_text(encoding='utf-8')))
            metadata['checkpoint'] = str(checkpoints[-1])
        metadata['status'] = 'finished'
    except BaseException as exc:
        metadata.update(status='failed', error=str(exc))
        raise
    finally:
        write_json(output / 'model_info.json', metadata)
    print('LTX-2.5 caches prepared without optimizer updates.' if args.mode == 'cache' else f'LTX-2.5 LoRA saved to {output}', flush=True)


def generate(args, settings):
    import random
    if args.checkpoint:
        contract.checkpoint_info(Path(args.checkpoint))
        settings['profile'] = 'dev'
    if settings['randomize_seed']:
        settings['seed'] = random.SystemRandom().randrange(2147483648)
    require_weights(settings['profile'])
    name = f'ltx25_{datetime.now():%Y%m%d_%H%M%S_%f}'
    samples = ROOT / 'samples'
    output = samples / name
    output.mkdir(parents=True, exist_ok=False)
    metadata = {'status': 'running', 'base_model': contract.MODEL_ID, 'revision': contract.REVISION,
                'settings': settings, 'model_path': args.checkpoint}
    write_json(output / 'generation.json', metadata)
    try:
        run_worker({'mode': 'generate', 'settings': settings, 'model_path': args.checkpoint}, output)
        video, audio = output / 'video.mp4', output / 'audio.wav'
        info, sound = media.probe(video), media.probe(audio)
        duration = contract.frames_for_seconds(settings['duration'], 24) / 24
        if (not info['audio'] or abs(info['duration'] - duration) > .1 or abs(sound['duration'] - duration) > .1
                or f'{info["width"]}x{info["height"]}' != settings['format'] or abs(info['fps'] - 24) > .1):
            raise RuntimeError('Generated audio/video did not match the requested timeline, audio or dimensions.')
        # The existing sample browser/storyboard discovers flat MP4 files.
        video.replace(samples / f'{name}.mp4')
        audio.replace(samples / f'{name}.wav')
        metadata.update(status='finished', video=str(samples / f'{name}.mp4'), audio=str(samples / f'{name}.wav'))
    except BaseException as exc:
        metadata.update(status='failed', error=str(exc))
        raise
    finally:
        write_json(output / 'generation.json', metadata)
    print(f'Saved synchronized MP4 and WAV: {samples / name}', flush=True)
