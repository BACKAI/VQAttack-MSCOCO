#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MSCOCO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOWNLOAD_DIR=""
SKIP_IMAGES=0
SKIP_QUESTIONS=0
SKIP_ANNOTATIONS=0
PYTHON_BIN="${PYTHON:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mscoco-root) MSCOCO_ROOT="$2"; shift 2 ;;
    --download-dir) DOWNLOAD_DIR="$2"; shift 2 ;;
    --skip-images) SKIP_IMAGES=1; shift ;;
    --skip-questions) SKIP_QUESTIONS=1; shift ;;
    --skip-annotations) SKIP_ANNOTATIONS=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$DOWNLOAD_DIR" ]]; then
  DOWNLOAD_DIR="${MSCOCO_ROOT}/downloads"
fi

mkdir -p "$MSCOCO_ROOT" "$DOWNLOAD_DIR"

zip_is_valid() {
  local zip_path="$1"
  if command -v unzip >/dev/null 2>&1; then
    unzip -tq "$zip_path" >/dev/null 2>&1
    return $?
  fi
  "$PYTHON_BIN" - "$zip_path" <<'PY'
import sys
import zipfile

zip_path = sys.argv[1]
try:
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
except zipfile.BadZipFile:
    raise SystemExit(1)
raise SystemExit(0 if bad is None else 1)
PY
}

download_file() {
  local url="$1"
  local target="$2"
  if [[ -s "$target" ]]; then
    if zip_is_valid "$target"; then
      echo "Already downloaded and zip-verified: ${target}"
      return
    fi
    echo "Existing zip is incomplete or invalid, resuming download: ${target}"
  fi

  echo "Downloading ${url}"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -x 8 -s 8 -c -o "$(basename "$target")" -d "$(dirname "$target")" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -c -O "$target" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -L --fail --continue-at - -o "$target" "$url"
  else
    echo "Need one downloader: aria2c, wget, or curl." >&2
    exit 1
  fi
}

extract_zip() {
  local zip_path="$1"
  local target_dir="$2"
  echo "Extracting ${zip_path} into ${target_dir}"
  if command -v unzip >/dev/null 2>&1; then
    unzip -n "$zip_path" -d "$target_dir"
  else
    "$PYTHON_BIN" - "$zip_path" "$target_dir" <<'PY'
import sys
import zipfile
from pathlib import Path

zip_path = Path(sys.argv[1])
target_dir = Path(sys.argv[2])
with zipfile.ZipFile(zip_path) as zf:
    for member in zf.infolist():
        target = target_dir / member.filename
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member) as src, target.open("wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
PY
  fi
}

if [[ "$SKIP_IMAGES" -eq 0 ]]; then
  download_file "http://images.cocodataset.org/zips/train2014.zip" "${DOWNLOAD_DIR}/train2014.zip"
  download_file "http://images.cocodataset.org/zips/val2014.zip" "${DOWNLOAD_DIR}/val2014.zip"
  extract_zip "${DOWNLOAD_DIR}/train2014.zip" "$MSCOCO_ROOT"
  extract_zip "${DOWNLOAD_DIR}/val2014.zip" "$MSCOCO_ROOT"
fi

if [[ "$SKIP_QUESTIONS" -eq 0 ]]; then
  download_file "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip" "${DOWNLOAD_DIR}/v2_Questions_Train_mscoco.zip"
  download_file "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip" "${DOWNLOAD_DIR}/v2_Questions_Val_mscoco.zip"
  extract_zip "${DOWNLOAD_DIR}/v2_Questions_Train_mscoco.zip" "$MSCOCO_ROOT"
  extract_zip "${DOWNLOAD_DIR}/v2_Questions_Val_mscoco.zip" "$MSCOCO_ROOT"
fi

if [[ "$SKIP_ANNOTATIONS" -eq 0 ]]; then
  download_file "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip" "${DOWNLOAD_DIR}/v2_Annotations_Train_mscoco.zip"
  download_file "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip" "${DOWNLOAD_DIR}/v2_Annotations_Val_mscoco.zip"
  extract_zip "${DOWNLOAD_DIR}/v2_Annotations_Train_mscoco.zip" "$MSCOCO_ROOT"
  extract_zip "${DOWNLOAD_DIR}/v2_Annotations_Val_mscoco.zip" "$MSCOCO_ROOT"
fi

"$PYTHON_BIN" "${SCRIPT_DIR}/verify_mscoco_vqa_assets.py" \
  --mscoco-root "$MSCOCO_ROOT"
