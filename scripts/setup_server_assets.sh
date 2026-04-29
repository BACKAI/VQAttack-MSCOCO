#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_BASE="$(cd "${SCRIPT_DIR}/.." && pwd)"
VLM_ROOT="$(cd "${REPO_BASE}/../.." && pwd)"

TEXTVQA_ROOT="${TEXTVQA_ROOT:-${VLM_ROOT}/dataset/textvqa}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_BASE}/checkpoints/albef}"
SKIP_WEIGHTS=0
SKIP_DATASET=0
FORCE_EXPORT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --textvqa-root) TEXTVQA_ROOT="$2"; shift 2 ;;
    --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
    --skip-weights) SKIP_WEIGHTS=1; shift ;;
    --skip-dataset) SKIP_DATASET=1; shift ;;
    --force-export) FORCE_EXPORT=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

PYTHON_BIN="${PYTHON:-python}"

if [[ "$SKIP_WEIGHTS" -eq 0 ]]; then
  CHECKPOINT_DIR="$CHECKPOINT_DIR" bash "${SCRIPT_DIR}/download_weights.sh"
fi

if [[ "$SKIP_DATASET" -eq 0 ]]; then
  mkdir -p "$TEXTVQA_ROOT"
  "$PYTHON_BIN" "${REPO_BASE}/textvqa_tools/download_textvqa_hf.py" \
    --out-dir "$TEXTVQA_ROOT"

  export_args=(
    --dataset-dir "$TEXTVQA_ROOT"
    --out-dir "${TEXTVQA_ROOT}/original_format"
  )
  if [[ "$FORCE_EXPORT" -eq 1 ]]; then
    export_args+=(--force)
  fi

  "$PYTHON_BIN" "${REPO_BASE}/textvqa_tools/export_textvqa_original_format.py" \
    "${export_args[@]}"
fi

cat <<EOF
Server assets are ready.

TextVQA root:
  ${TEXTVQA_ROOT}

TextVQA original_format:
  ${TEXTVQA_ROOT}/original_format

Checkpoint dir:
  ${CHECKPOINT_DIR}
EOF
