from __future__ import annotations

import argparse
from pathlib import Path


def initialize_workspace(root: Path) -> None:
    """Create the empty local folders and a portable default dataset config."""
    root = root.resolve()
    for relative in (
        "dataset/videos",
        "cache",
        "models",
        "output",
        "samples",
        "logs",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)

    config = root / "dataset_config.toml"
    if not config.exists():
        video_dir = (root / "dataset" / "videos").as_posix()
        cache_dir = (root / "cache").as_posix()
        config.write_text(
            "\n".join(
                [
                    "[general]",
                    "resolution = [448, 256]",
                    'caption_extension = ".txt"',
                    "batch_size = 1",
                    "enable_bucket = true",
                    "bucket_no_upscale = true",
                    "",
                    "[[datasets]]",
                    f'video_directory = "{video_dir}"',
                    f'cache_directory = "{cache_dir}"',
                    "target_frames = [25, 49]",
                    'frame_extraction = "head"',
                    "num_repeats = 1",
                    "",
                ]
            ),
            encoding="utf-8",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    initialize_workspace(args.root)
    print(f"Empty runtime workspace initialized at {args.root.resolve()}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
