from __future__ import annotations

import argparse
import math
import shutil
from fractions import Fraction
from pathlib import Path

import av


OUTPUT_FPS = 16
OUTPUT_FRAMES = 49
CLIP_SECONDS = OUTPUT_FRAMES / OUTPUT_FPS


def video_duration(path: Path) -> float:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.duration is not None and stream.time_base is not None:
            return float(stream.duration * stream.time_base)
        if container.duration is not None:
            return float(container.duration / av.time_base)
    raise ValueError("The source video duration could not be determined.")


def evenly_spaced_starts(duration: float, count: int) -> list[float]:
    latest = duration - CLIP_SECONDS
    if latest < 0:
        raise ValueError(f"The source must be at least {CLIP_SECONDS:.2f} seconds long.")
    if count == 1:
        return [latest / 2]
    return [latest * index / (count - 1) for index in range(count)]


def unique_prefix(output_dir: Path, source_stem: str) -> str:
    prefix = f"{source_stem}_clip"
    if not any(output_dir.glob(f"{prefix}_*.mp4")):
        return prefix
    run = 2
    while any(output_dir.glob(f"{prefix}{run}_*.mp4")):
        run += 1
    return f"{prefix}{run}"


def encode_clip(source: Path, destination: Path, start: float) -> int:
    with av.open(str(source)) as input_container:
        input_stream = input_container.streams.video[0]
        width = input_stream.width - (input_stream.width % 2)
        height = input_stream.height - (input_stream.height % 2)
        seek_timestamp = max(0, int(start / float(input_stream.time_base)))
        input_container.seek(seek_timestamp, stream=input_stream, backward=True, any_frame=False)

        with av.open(str(destination), mode="w") as output_container:
            output_stream = output_container.add_stream("libx264", rate=OUTPUT_FPS)
            output_stream.width = width
            output_stream.height = height
            output_stream.pix_fmt = "yuv420p"
            output_stream.options = {"preset": "fast", "crf": "18"}

            written = 0
            next_time = start
            last_frame = None
            for frame in input_container.decode(input_stream):
                if frame.pts is None:
                    continue
                frame_time = float(frame.pts * frame.time_base)
                if frame_time + 1e-6 < next_time:
                    continue
                last_frame = frame
                while frame_time + 1e-6 >= next_time and written < OUTPUT_FRAMES:
                    converted = last_frame.reformat(width=width, height=height, format="yuv420p")
                    converted.pts = written
                    converted.time_base = Fraction(1, OUTPUT_FPS)
                    for packet in output_stream.encode(converted):
                        output_container.mux(packet)
                    written += 1
                    next_time = start + written / OUTPUT_FPS
                if written >= OUTPUT_FRAMES:
                    break
            for packet in output_stream.encode():
                output_container.mux(packet)
    return written


def move_with_unique_name(source: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    counter = 2
    while destination.exists():
        destination = destination_dir / f"{source.stem}_{counter}{source.suffix}"
        counter += 1
    shutil.move(str(source), str(destination))
    return destination


def archive_source_and_cache(source: Path, output_dir: Path, cache_dir: Path | None) -> None:
    if source.parent.resolve() != output_dir.resolve():
        print("Source is outside the active dataset, so it was left in place.", flush=True)
        return
    archive_dir = output_dir / "_source_archive"
    caption = source.with_suffix(".txt")
    archived_source = move_with_unique_name(source, archive_dir)
    if caption.exists():
        move_with_unique_name(caption, archive_dir)
    print(f"Archived long source to: {archived_source}", flush=True)

    if cache_dir and cache_dir.exists():
        stale = [path for path in cache_dir.glob(f"{source.stem}*.safetensors") if path.is_file()]
        if stale:
            cache_archive = cache_dir / "_source_archive"
            for path in stale:
                move_with_unique_name(path, cache_archive)
            print(f"Archived {len(stale)} stale cache file(s) so they cannot enter the new run.", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create evenly spaced, training-ready clips from one long video.")
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--caption-file", type=Path)
    parser.add_argument("--trigger", default="subject_token")
    parser.add_argument("--archive-source", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source video not found: {source}")
    if source.suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}:
        raise ValueError("Unsupported source video type.")
    if not 1 <= args.count <= 100:
        raise ValueError("Clip count must be between 1 and 100.")
    output_dir.mkdir(parents=True, exist_ok=True)

    caption = f"{args.trigger}, speaking directly to the camera"
    if args.caption_file and args.caption_file.exists():
        existing = args.caption_file.read_text(encoding="utf-8-sig", errors="replace").strip()
        if existing:
            caption = existing

    duration = video_duration(source)
    starts = evenly_spaced_starts(duration, args.count)
    prefix = unique_prefix(output_dir, source.stem)
    print(
        f"Source duration: {duration:.1f}s. Creating {len(starts)} evenly spaced clips, "
        f"{OUTPUT_FRAMES} frames each at {OUTPUT_FPS} fps ({CLIP_SECONDS:.2f}s).",
        flush=True,
    )

    created: list[Path] = []
    for index, start in enumerate(starts, start=1):
        destination = output_dir / f"{prefix}_{index:03d}.mp4"
        print(f"[{index}/{len(starts)}] {destination.name} from {start:.2f}s", flush=True)
        frames = encode_clip(source, destination, start)
        if frames != OUTPUT_FRAMES:
            if destination.exists():
                destination.unlink()
            raise RuntimeError(f"Only encoded {frames}/{OUTPUT_FRAMES} frames for clip {index}.")
        destination.with_suffix(".txt").write_text(caption + "\n", encoding="utf-8")
        created.append(destination)

    print(f"Created {len(created)} training clips in {output_dir}.", flush=True)
    if args.archive_source:
        archive_source_and_cache(source, output_dir, args.cache_dir.resolve() if args.cache_dir else None)
    print("Review the clips and captions in the Dataset tab before caching or training.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
