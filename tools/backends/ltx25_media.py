"""Audio/video timeline helpers adapted from ADAM."""
from __future__ import annotations

import hashlib

import math

import os

import subprocess

import re

from functools import lru_cache

from pathlib import Path

def command(args: list[str]) -> None:
    result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace",
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if result.returncode:
        raise ValueError(f"Media conversion failed: {result.stderr[-3000:]}")

@lru_cache(maxsize=2)
def binary(name: str) -> str:
    """Skip legacy tools earlier on PATH without modifying the user's PATH."""
    suffix = ".exe" if os.name == "nt" else ""
    candidates = [Path(part.strip('"')) / (name + suffix) for part in os.environ.get("PATH", "").split(os.pathsep) if part]
    if name == 'ffmpeg':
        try:
            import imageio_ffmpeg
            candidates.insert(0, Path(imageio_ffmpeg.get_ffmpeg_exe()))
        except ImportError:
            pass
    for candidate in dict.fromkeys(candidates):
        if not candidate.is_file():
            continue
        result = subprocess.run([str(candidate), "-version"], capture_output=True, text=True, timeout=10,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        match = re.search(r"version\s+(\d+)\.", result.stdout)
        if result.returncode == 0 and match and int(match[1]) >= 5:
            return str(candidate)
    raise ValueError(f"A current {name} (version 5+) is required. Rerun setup.ps1 to install bundled FFmpeg.")

def probe(path: Path) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError("Choose an existing media file.")
    import av
    try:
        with av.open(str(path)) as container:
            video_stream = next(iter(container.streams.video), None)
            audio_stream = next(iter(container.streams.audio), None)
            fallback = float(container.duration or 0) / av.time_base
            def describe(stream):
                if stream is None:
                    return None
                length = float(stream.duration * stream.time_base) if stream.duration is not None and stream.time_base else fallback
                return {'index': stream.index, 'duration': length, 'codec_name': stream.codec_context.name}
            video, audio = describe(video_stream), describe(audio_stream)
            duration = (video or audio or {}).get('duration', fallback)
            width, height = (video_stream.width, video_stream.height) if video_stream else (0, 0)
            fps = float(video_stream.average_rate or 0) if video_stream else 0.
    except av.error.FFmpegError as exc:
        raise ValueError(f'Cannot inspect {path.name}: {exc}') from exc
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Media has no valid duration.")
    return {"path": str(path), "duration": duration, "video": video, "audio": audio,
            "width": width, "height": height, 'fps': fps}

def fingerprint(path: Path, checkpoint=None) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            if checkpoint:
                checkpoint()
            digest.update(block)
    return digest.hexdigest()

def ffmpeg() -> str:
    return binary("ffmpeg")

def mux(video: Path, audio: Path, output: Path, duration: float, *, run=command) -> None:
    run([ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(video), "-i", str(audio),
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-t", str(duration), "-movflags", "+faststart", str(output)])
