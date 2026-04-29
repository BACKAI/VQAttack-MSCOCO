#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_BASE="$(cd "${SCRIPT_DIR}/.." && pwd)"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_BASE}/checkpoints/albef}"
ALBEF_URL="${ALBEF_URL:-}"
VQA_URL="${VQA_URL:-}"

mkdir -p "$CHECKPOINT_DIR"

if [[ -z "$ALBEF_URL" || -z "$VQA_URL" ]]; then
  cat >&2 <<'EOF'
Set ALBEF_URL and VQA_URL before running this script.

Example:
  ALBEF_URL="https://github.com/<user>/<repo>/releases/download/weights/ALBEF.pth" \
  VQA_URL="https://github.com/<user>/<repo>/releases/download/weights/vqa.pth" \
  bash scripts/download_weights.sh
EOF
  exit 2
fi

download() {
  local url="$1"
  local output="$2"
  if command -v wget >/dev/null 2>&1; then
    wget -O "$output" "$url"
  else
    curl -L -o "$output" "$url"
  fi
}

download "$ALBEF_URL" "${CHECKPOINT_DIR}/ALBEF.pth"
download "$VQA_URL" "${CHECKPOINT_DIR}/vqa.pth"

echo "Downloaded weights to ${CHECKPOINT_DIR}"
