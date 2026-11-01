"""Explicit full-model smoke test. Uses installed official weights; never downloads them.

Run with the LTX-2.5 Python. Creates a synthetic video with a tone, caches it,
trains one step, loads that adapter for two-step dev generation, and continues it
for one step at 65 frames. This tests the pipeline, not learned model quality.
"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.backends import ltx25_contract as contract, ltx25_media as media


def main():
    contract.require_hardware()
    name = f'ltx25_smoke_{datetime.now():%Y%m%d_%H%M%S}'
    folder = ROOT / 'logs' / name
    dataset = folder / 'dataset'
    dataset.mkdir(parents=True)
    media.command([media.ffmpeg(), '-hide_banner', '-loglevel', 'error', '-y',
                   '-f', 'lavfi', '-i', 'testsrc2=s=256x256:r=24:d=3',
                   '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000:duration=3',
                   '-c:v', 'libx264', '-c:a', 'pcm_f32le', str(dataset / 'clip.mkv')])
    (dataset / 'clip.txt').write_text('subject_token shows a moving color test pattern. Audio: A steady tone.\n', encoding='utf-8')
    base = [sys.executable, '-u', str(ROOT / 'tools/video_backend.py'), '--backend', 'ltx25']
    def run(label, mode, settings, *extra):
        print(f'{label}: running full-model check; log: {folder / (label + ".log")}', flush=True)
        with (folder / f'{label}.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(base + ['--mode', mode, '--settings', json.dumps(settings), *extra],
                                    stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        if result.returncode:
            print((folder / f'{label}.log').read_text(encoding='utf-8', errors='replace')[-6000:], flush=True)
            raise RuntimeError(f'{label} failed; inspect its log.')
        print(f'{label}: passed', flush=True)
    settings = {'trigger_word': 'subject_token', 'max_steps': 1, 'save_every': 1}
    run('cache', 'cache', settings, '--dataset', str(dataset))
    run('train', 'train', settings, '--dataset', str(dataset), '--name', name, '--epochs', '1')
    output = ROOT / 'output/ltx25' / name
    checkpoint = Path(json.loads((output / 'model_info.json').read_text(encoding='utf-8'))['checkpoint'])
    contract.checkpoint_info(checkpoint)
    run('generate', 'generate', {'prompt': 'subject_token shows moving colors. Audio: A steady tone.',
                                'format': '256x256', 'duration': .75, 'steps': 2}, '--checkpoint', str(checkpoint))
    run('continue', 'train', {**settings, 'target_frames': '65'}, '--dataset', str(dataset),
        '--name', name + '_continued', '--epochs', '1', '--resume', str(checkpoint))
    result = {'status': 'passed', 'cache': True, 'one_step_joint_training': True,
              'adapter_generation': True, '65_frame_continuation': True, 'checkpoint': str(checkpoint),
              'synthetic_dataset': True, 'quality_benchmark': False}
    (folder / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
