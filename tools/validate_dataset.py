from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import av


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}


@dataclass
class Result:
    path: Path
    duration: float
    fps: float
    width: int
    height: int
    frames: int


def inspect_video(path: Path) -> Result:
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("no video stream")
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        frames = int(stream.frames or 0)
        duration = 0.0
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = float(container.duration / av.time_base)
        if frames == 0 and fps > 0 and duration > 0:
            frames = round(fps * duration)
        return Result(path, duration, fps, stream.width, stream.height, frames)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate paired video/caption data for the starter LoRA experiment.")
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parents[1] / "dataset" / "videos")
    parser.add_argument("--trigger", default="subject_token")
    args = parser.parse_args()

    videos = sorted(p for p in args.directory.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
    orphan_captions = sorted(
        p for p in args.directory.glob("*.txt")
        if not any(p.with_suffix(ext).exists() for ext in VIDEO_EXTENSIONS)
    )
    errors: list[str] = []
    warnings: list[str] = []

    if not videos:
        errors.append(f"No videos found in {args.directory}")

    print(f"Found {len(videos)} video(s).")
    total_seconds = 0.0
    for video in videos:
        caption_path = video.with_suffix(".txt")
        if not caption_path.exists():
            errors.append(f"{video.name}: missing {caption_path.name}")
            continue
        caption = caption_path.read_text(encoding="utf-8-sig").strip()
        if not caption:
            errors.append(f"{caption_path.name}: caption is empty")
        trigger_pattern = rf"(?<![\w]){re.escape(args.trigger)}(?![\w])"
        if not re.search(trigger_pattern, caption):
            errors.append(f"{caption_path.name}: missing exact trigger token '{args.trigger}'")
        if len(caption) > 500:
            warnings.append(f"{caption_path.name}: caption is unusually long; do not paste a transcript")

        try:
            result = inspect_video(video)
            total_seconds += result.duration
            print(
                f"- {video.name}: {result.width}x{result.height}, "
                f"{result.duration:.2f}s, {result.fps:.2f} fps, ~{result.frames} frames"
            )
            if result.duration and not 1.5 <= result.duration <= 5.0:
                warnings.append(f"{video.name}: prefer a single continuous 2-4 second shot")
            if result.frames and result.frames < 25:
                errors.append(f"{video.name}: fewer than 25 frames")
            if min(result.width, result.height) < 256:
                warnings.append(f"{video.name}: source resolution is below the 256-pixel short edge")
        except Exception as exc:
            errors.append(f"{video.name}: cannot read video ({exc})")

    for caption in orphan_captions:
        warnings.append(f"{caption.name}: caption has no matching video")

    print(f"Total footage: {total_seconds:.1f} seconds")
    if len(videos) < 20:
        warnings.append("The baseline target is at least 25 clean clips; early pipeline tests can use fewer.")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")
    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("\nDataset validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
