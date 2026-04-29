#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MSCOCO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOWNLOAD_DIR=""
SKIP_IMAGES=0
SKIP_QUESTIONS=0
SKIP_ANNOTATIONS=0

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

download_file() {
  local url="$1"
  local target="$2"
  if [[ -s "$target" ]]; then
    echo "Already downloaded: ${target}"
    return
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
  if ! command -v unzip >/dev/null 2>&1; then
    echo "Missing unzip. Install it outside this script, then rerun." >&2
    exit 1
  fi
  echo "Extracting ${zip_path} into ${target_dir}"
  unzip -n "$zip_path" -d "$target_dir"
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

"${PYTHON:-python}" "${SCRIPT_DIR}/verify_mscoco_vqa_assets.py" \
  --mscoco-root "$MSCOCO_ROOT"
