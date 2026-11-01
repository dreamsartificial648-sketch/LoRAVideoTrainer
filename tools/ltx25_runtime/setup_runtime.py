"""Install the pinned LTX-2.5 runtime without touching other model environments."""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.backends.ltx25_contract import UPSTREAM_REVISION, hardware_report, runtime_root, store_root


def run(*args):
    subprocess.run([str(arg) for arg in args], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', type=Path, help='Optional SSD directory for base weights')
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError('Install 64-bit Python 3.10 and use py -3.10.')
    # Standard-library-only probe: reject a known unsupported GPU before the multi-GB Torch install.
    # Custom device mappings are checked using Torch after installation instead.
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        try:
            probe = subprocess.run(['nvidia-smi', '--id=0', '--query-gpu=name,compute_cap,memory.total',
                                    '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            if probe.returncode == 0:
                name, capability, memory = next(csv.reader(probe.stdout.splitlines()))
                report = hardware_report(name.strip(), tuple(map(int, capability.strip().split('.'))),
                                         int(memory.strip()) * 1024**2, 32 * 1024**3)
                if report['errors']:
                    raise RuntimeError('\n'.join(report['errors']))
        except (OSError, ValueError, StopIteration, subprocess.TimeoutExpired):
            pass  # Unknown/old nvidia-smi: the mandatory Torch probe remains authoritative.
    runtime = runtime_root()
    runtime.mkdir(parents=True, exist_ok=True)
    if args.store:
        store = args.store.expanduser().resolve()
        store.mkdir(parents=True, exist_ok=True)
        (runtime / 'paths.json').write_text(json.dumps({'store': str(store)}, indent=2), encoding='utf-8')
    python = runtime / '.venv/Scripts/python.exe'
    if not python.is_file():
        run(sys.executable, '-m', 'venv', runtime / '.venv')
    run(python, '-m', 'pip', 'install', '--upgrade', 'pip')
    run(python, '-m', 'pip', 'install', 'torch==2.7.1', 'torchvision==0.22.1', 'torchaudio==2.7.1',
        '--index-url', 'https://download.pytorch.org/whl/cu126')
    run(python, '-m', 'pip', 'install', 'psutil==7.2.2')
    # Fail before installing the remaining stack or downloading dozens of GB of weights.
    run(python, '-c', 'from tools.backends.ltx25_contract import require_hardware; require_hardware()')
    upstream = runtime / 'upstream'
    if not upstream.exists():
        run('git', 'clone', '--no-checkout', '--depth', '1', 'https://github.com/ostris/ai-toolkit.git', upstream)
    current = subprocess.run(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    if current != UPSTREAM_REVISION:
        dirty = subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True).strip()
        # A just-created --no-checkout clone has an empty index; checkout the pinned source directly.
        if dirty and (upstream / 'toolkit').exists():
            raise RuntimeError('The Toolkit checkout has local changes. Preserve them before changing its revision.')
        run('git', '-C', upstream, 'fetch', 'origin', UPSTREAM_REVISION, '--depth', '1')
        run('git', '-C', upstream, 'checkout', '--detach', UPSTREAM_REVISION)
    elif not (upstream / 'toolkit').exists():
        run('git', '-C', upstream, 'checkout', '--detach', UPSTREAM_REVISION)
    here = Path(__file__).resolve().parent
    run(python, '-m', 'pip', 'install', '-r', here / 'requirements.lock.txt', '-c', here / 'constraints.txt')
    run(python, here / 'patch_runtime.py')
    run(python, here / 'check_runtime.py')
    print(f'LTX-2.5 installed. Weight store: {store_root()}', flush=True)


if __name__ == '__main__':
    # Relative package imports in child commands must also work from another launch directory.
    os.chdir(ROOT)
    main()
