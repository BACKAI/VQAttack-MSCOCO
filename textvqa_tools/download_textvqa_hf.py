#!/usr/bin/env python3
"""Download lmms-lab/textvqa from Hugging Face into a local dataset folder.

The dataset repository stores the actual data as parquet shards under `data/`.
This script uses only the Python standard library so it can run without
installing `huggingface_hub` or `datasets`.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


REPO_ID = "lmms-lab/textvqa"
BASE_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main"

FILES = [
    ".gitattributes",
    "README.md",
    "data/test-00000-of-00004.parquet",
    "data/test-00001-of-00004.parquet",
    "data/test-00002-of-00004.parquet",
    "data/test-00003-of-00004.parquet",
    "data/train-00000-of-00020.parquet",
    "data/train-00001-of-00020.parquet",
    "data/train-00002-of-00020.parquet",
    "data/train-00003-of-00020.parquet",
    "data/train-00004-of-00020.parquet",
    "data/train-00005-of-00020.parquet",
    "data/train-00006-of-00020.parquet",
    "data/train-00007-of-00020.parquet",
    "data/train-00008-of-00020.parquet",
    "data/train-00009-of-00020.parquet",
    "data/train-00010-of-00020.parquet",
    "data/train-00011-of-00020.parquet",
    "data/train-00012-of-00020.parquet",
    "data/train-00013-of-00020.parquet",
    "data/train-00014-of-00020.parquet",
    "data/train-00015-of-00020.parquet",
    "data/train-00016-of-00020.parquet",
    "data/train-00017-of-00020.parquet",
    "data/train-00018-of-00020.parquet",
    "data/train-00019-of-00020.parquet",
    "data/validation-00000-of-00003.parquet",
    "data/validation-00001-of-00003.parquet",
    "data/validation-00002-of-00003.parquet",
]


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def remote_size(url: str, timeout: int) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            length = response.headers.get("Content-Length")
    except urllib.error.HTTPError as exc:
        if exc.code == 405:
            return None
        raise
    return int(length) if length is not None else None


def download_one(url: str, destination: Path, timeout: int, retries: int) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = remote_size(url, timeout)

    if expected_size is not None and destination.exists():
        local_size = destination.stat().st_size
        if local_size == expected_size:
            print(f"skip complete: {destination} ({format_bytes(local_size)})")
            return False
        if local_size > expected_size:
            destination.unlink()

    for attempt in range(1, retries + 1):
        local_size = destination.stat().st_size if destination.exists() else 0
        headers = {"User-Agent": "vlm-textvqa-downloader/1.0"}
        mode = "wb"
        if local_size > 0:
            headers["Range"] = f"bytes={local_size}-"
            mode = "ab"

        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if local_size > 0 and response.status != 206:
                    mode = "wb"
                    local_size = 0
                total = expected_size or local_size + int(response.headers.get("Content-Length", 0))
                print(
                    f"download: {destination} "
                    f"from {format_bytes(local_size)} / {format_bytes(total)}"
                )
                downloaded = local_size
                last_report = time.monotonic()
                with destination.open(mode + "") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        now = time.monotonic()
                        if now - last_report >= 10:
                            print(
                                f"  progress: {format_bytes(downloaded)} / "
                                f"{format_bytes(total)}"
                            )
                            last_report = now
            if expected_size is None or destination.stat().st_size == expected_size:
                return True
            print(
                f"size mismatch after attempt {attempt}: "
                f"{format_bytes(destination.stat().st_size)} != {format_bytes(expected_size)}"
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"attempt {attempt} failed for {destination}: {exc}", file=sys.stderr)
            if attempt == retries:
                raise
            time.sleep(min(30, 2 * attempt))
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download lmms-lab/textvqa parquet files.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("dataset/textvqa"),
        help="Output directory.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Network timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Retries per file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print URLs without downloading.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    downloaded_count = 0

    for relative_path in FILES:
        url = f"{BASE_URL}/{relative_path}?download=true"
        destination = args.out_dir / relative_path
        if args.dry_run:
            print(f"{url} -> {destination}")
            continue
        if download_one(url, destination, timeout=args.timeout, retries=args.retries):
            downloaded_count += 1

    if args.dry_run:
        print(f"dry_run_files: {len(FILES)}")
        return 0

    parquet_files = list((args.out_dir / "data").glob("*.parquet"))
    total_bytes = sum(path.stat().st_size for path in args.out_dir.rglob("*") if path.is_file())
    print("Summary")
    print(f"  repo_id: {REPO_ID}")
    print(f"  output_dir: {args.out_dir}")
    print(f"  manifest_files: {len(FILES)}")
    print(f"  downloaded_or_updated: {downloaded_count}")
    print(f"  parquet_files: {len(parquet_files)}")
    print(f"  total_size: {format_bytes(total_bytes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
