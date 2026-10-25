"""CPU-only validation and cache bookkeeping shared by video workers."""
from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

VIDEO_EXTENSIONS = {'.mp4', '.mov', '.mkv', '.webm', '.m4v', '.avi'}


def clips_in(folder):
    return sorted(p for p in Path(folder).rglob('*') if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


def fingerprint(folder):
    return [[str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns]
            for clip in clips_in(folder) for p in (clip, clip.with_suffix('.txt'))]


def validate_settings(values, schema):
    result = {key: spec['default'] for key, spec in schema.items()}
    if set(values) - set(schema):
        raise ValueError('Unsupported settings: ' + ', '.join(sorted(set(values) - set(schema))))
    result.update(values)
    for key, spec in schema.items():
        value, kind = result[key], spec['type']
        if kind in {'int', 'float'}:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{spec['label']} must be a finite number.")
            if kind == 'int' and int(value) != value:
                raise ValueError(f"{spec['label']} must be a whole number.")
            if not spec['min'] <= value <= spec['max']:
                raise ValueError(f"{spec['label']} must be between {spec['min']} and {spec['max']}.")
            result[key] = int(value) if kind == 'int' else float(value)
        elif kind == 'bool' and not isinstance(value, bool):
            raise ValueError(f"{spec['label']} must be true or false.")
        elif kind == 'choice' and value not in spec['options']:
            raise ValueError(f"Choose a supported {spec['label']}.")
        elif kind in {'text', 'multiline_text'}:
            if not isinstance(value, str) or len(value) > 2000 or '\0' in value or (spec.get('required') and not value.strip()):
                raise ValueError(f"{spec['label']} requires text up to 2000 characters.")
            result[key] = value.strip()
    return result


def validate_dataset(folder, trigger, frames):
    import av
    clips = clips_in(folder)
    if not clips:
        raise ValueError('The dataset has no video clips.')
    for clip in clips:
        caption = clip.with_suffix('.txt').read_text(encoding='utf-8-sig').strip()
        if not caption or not re.search(rf'(?<!\w){re.escape(trigger)}(?!\w)', caption):
            raise ValueError(f'{clip.name}: caption must include the exact trigger {trigger!r}.')
        with av.open(str(clip)) as container:
            # Decode to count: container frame-count metadata is often missing or inaccurate.
            count = sum(1 for _ in container.decode(video=0))
        if count < frames:
            raise ValueError(f'{clip.name}: needs at least {frames} frames, found {count}.')
    return fingerprint(folder)


def cached_training_data(request, settings, build):
    cache = Path(request['training_cache'])
    manifest = cache / 'manifest.json'
    allowed = Path(request['cache_dir']).resolve() / request['backend']
    if cache.resolve() != allowed:
        raise ValueError('Training cache is outside the selected model cache directory.')
    expected = {key: request[key] for key in ('backend', 'dataset_files')}
    expected.update(version=1, revision=request.get('revision'), base_model=request.get('base_model'),
                    resolution=settings['resolution'], frames=settings['target_frames'], seed=settings['seed'])
    try:
        saved = json.loads(manifest.read_text(encoding='utf-8'))
        paths = [cache / name for name in saved['files']]
        if saved['identity'] == expected and paths and all(p.is_file() for p in paths):
            print(f'Reusing {len(paths)} cached clips and captions.', flush=True)
            return paths
    except (OSError, ValueError, KeyError):
        pass
    # Only the designated model's disposable cache may be replaced.
    if cache.exists():
        shutil.rmtree(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    paths = build()
    if fingerprint(Path(request['dataset_dir'])) != request['dataset_files']:
        raise ValueError('Dataset changed while caching; rebuild the cache.')
    manifest.write_text(json.dumps({'identity': expected, 'files': [p.name for p in paths]}, indent=2), encoding='utf-8')
    return paths
