"""Download gated official weights once to the shared store, without cache duplicates."""
import argparse
import json
import os
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
import shutil
import sys
from pathlib import Path

# The ordinary HTTP path supports resumable files reliably on this Windows setup.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from tools.backends.ltx25_contract import FILES, MODEL_ID, REVISION, store_root
from huggingface_hub import hf_hub_url, get_hf_file_metadata, get_token


def download(file, info, destination, token):
    """Bound each HTTP request and verify the full official SHA256 before activation."""
    import requests
    target = destination / file
    target.parent.mkdir(parents=True, exist_ok=True)
    size, digest = info["size"], info["etag"].strip('"')
    if target.is_file() and target.stat().st_size == size:
        return
    partial = target.with_suffix(target.suffix + ".partial")
    if not partial.exists():
        # Reuse a previous Hub HTTP attempt, from this exact filename and etag only.
        cache = destination / ".cache/huggingface/download" / Path(file).parent
        candidates = [p for p in cache.glob(f"*.{digest}.incomplete") if p.stat().st_size <= size]
        if candidates:
            max(candidates, key=lambda p: p.stat().st_size).replace(partial)
    if partial.exists() and partial.stat().st_size > size:
        raise RuntimeError(f"Unexpected partial file size for {target.name}; preserve it and investigate.")
    # Reuse the authorized CDN redirect rather than resolving every interval at the Hub.
    # The signed URL stays only in memory and is never logged or written to metadata.
    url = get_hf_file_metadata(hf_hub_url(MODEL_ID, file, revision=REVISION), token=token).location
    headers = {"Authorization": "Bearer " + token} if token and url.startswith("https://huggingface.co/") else {}
    failures = 0
    def fetch(offset, end):
        with requests.get(url, headers={**headers, "Range": f"bytes={offset}-{end}"}, stream=True, timeout=(15, 30)) as response:
            expected = f"bytes {offset}-{end}/{size}"
            if response.status_code != 206 or response.headers.get("content-range") != expected:
                raise RuntimeError(f"Unexpected range response for {target.name}: HTTP {response.status_code}")
            data = bytearray()
            for block in response.iter_content(1024**2):
                data.extend(block)
                if len(data) > end - offset + 1:
                    raise RuntimeError("Download response exceeded its interval.")
            if len(data) != end - offset + 1:
                raise RuntimeError("Download response ended early.")
            return data
    while not partial.exists() or partial.stat().st_size < size:
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            ranges = [(start, min(size - 1, start + 8 * 1024**2 - 1)) for start in range(offset, min(size, offset + 256 * 1024**2), 8 * 1024**2)]
            # Bounded RAM and ordered commits preserve a resumable prefix after interruption.
            with ThreadPoolExecutor(max_workers=32) as pool:
                pending = [pool.submit(fetch, start, end) for start, end in ranges]
                with partial.open("ab") as stream:
                    for future in pending:
                        stream.write(future.result())
                        stream.flush()
            failures = 0
            current = partial.stat().st_size
            print(f"{target.name}: {current / 1024**3:.2f}/{size / 1024**3:.2f} GiB", flush=True)
        except (requests.RequestException, RuntimeError) as exc:
            failures += 1
            print(f"Transfer retry {failures}/5 ({type(exc).__name__}); existing bytes retained.", flush=True)
            if failures >= 5:
                raise RuntimeError(f"Transfer interrupted for {target.name}. Rerun to resume.") from None
            time.sleep(2)
    print(f"Verifying {target.name}", flush=True)
    sha = hashlib.sha256()
    with partial.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            sha.update(block)
    if len(digest) != 64 or sha.hexdigest() != digest:
        raise RuntimeError(f"SHA256 verification failed for {target.name}. Partial file retained for inspection.")
    partial.replace(target)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["distilled", "dev", "all"], default="all")
    args = parser.parse_args(argv)
    store = store_root(HERE.parents[1])
    store.mkdir(parents=True, exist_ok=True)
    profiles = ["distilled", "dev"] if args.profile == "all" else [args.profile]
    keys = (["distilled"] if "distilled" in profiles else []) + ["text_encoder", "video_vae", "audio_vae"] + (["dev"] if "dev" in profiles else [])
    destination = store / "models"
    token = get_token()
    print("Checking official model access and file sizes...", flush=True)
    manifests = {}
    try:
        for key in keys:
            info = get_hf_file_metadata(hf_hub_url(MODEL_ID, FILES[key], revision=REVISION), token=token)
            manifests[key] = {"file": FILES[key], "size": info.size, "etag": info.etag}
    except Exception as exc:
        # No credential material in logs or request files.
        print(f"Model access check failed ({type(exc).__name__}). Accept access at https://huggingface.co/{MODEL_ID} and run the app's Model Access / Login button locally.")
        return 1
    needed = sum(m["size"] for m in manifests.values() if not (destination / m["file"]).is_file())
    free = shutil.disk_usage(store).free
    print(f"Store: {store}\nNew weights: {needed / 1024**3:.1f} GiB; free: {free / 1024**3:.1f} GiB", flush=True)
    if free < needed + 20 * 1024**3:
        raise RuntimeError("Insufficient storage: leave at least 20 GiB after the model download for caches and system use.")
    for key in keys:
        print(f"Downloading {FILES[key]}", flush=True)
        download(FILES[key], manifests[key], destination, token)
    (store / "weights.json").write_text(json.dumps({"repo": MODEL_ID, "revision": REVISION, "files": manifests}, indent=2), encoding="utf-8")
    print("Official LTX-2.5 weights ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
