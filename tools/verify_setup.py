from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import av
import torch


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENGINE_REVISION = "8934cfbbb4b9bcfa8071ce209129f0c5eb5df2e6"


def main() -> int:
    problems: list[str] = []
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyAV: {av.__version__}")

    if not torch.cuda.is_available():
        problems.append("PyTorch cannot access CUDA.")
    else:
        props = torch.cuda.get_device_properties(0)
        vram = props.total_memory / 1024**3
        print(f"GPU: {props.name} ({vram:.1f} GB VRAM)")
        if vram < 11.0:
            problems.append("The starter preset expects approximately 12 GB VRAM.")

    engine = ROOT / "trainer-engine"
    if not engine.exists():
        problems.append("trainer-engine is missing.")
    else:
        try:
            revision = subprocess.check_output(
                ["git", "-C", str(engine), "rev-parse", "HEAD"], text=True
            ).strip()
            print(f"Trainer revision: {revision}")
            if revision != EXPECTED_ENGINE_REVISION:
                print("NOTE: trainer revision differs from the tested workspace revision.")
        except (OSError, subprocess.CalledProcessError):
            problems.append("Could not read the trainer revision.")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        print(f"System FFmpeg: {ffmpeg}")
        print("NOTE: dataset inspection uses PyAV, so the old system FFmpeg is not required.")
    else:
        print("System FFmpeg: not found (PyAV is sufficient for validation/training reads).")

    for folder in ("models", "dataset/videos", "cache", "output", "samples", "logs"):
        path = ROOT / folder
        if not path.exists():
            problems.append(f"Missing folder: {path}")

    if problems:
        print("\nSetup check failed:")
        for problem in problems:
            print(f"- {problem}")
        return 1

    print("\nSetup check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
