#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLM_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

DEFAULT_MSCOCO_ROOT="${VLM_ROOT}/dataset/MSCOCO"
if [[ -d "$DEFAULT_MSCOCO_ROOT" ]]; then
  MSCOCO_ROOT="$DEFAULT_MSCOCO_ROOT"
else
  MSCOCO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

DEFAULT_REPO_ROOT="${VLM_ROOT}/Attack/VQAttack-MSCOCO/ALBEF_VQAttack/ALBEF_attack"
if [[ ! -d "$DEFAULT_REPO_ROOT" ]]; then
  DEFAULT_REPO_ROOT="${VLM_ROOT}/Attack/VQAttack/ALBEF_VQAttack/ALBEF_attack"
fi

REPO_ROOT="$DEFAULT_REPO_ROOT"
CHECKPOINT_ROOT="${VLM_ROOT}/Attack/VQAttack/checkpoints/albef"
WORK_ROOT="${VLM_ROOT}/outputs/mscoco_vqattack_eps_sweep"
GPU_IDS="4,5,6,7"
TRAIN_IMAGES=300
VAL_IMAGES=100
EPS_START=1
EPS_END=32
EPS_DENOM=255
EPS_ITER_DIVISOR=12.5
CONVERT_IMAGES=1
USE_SIMILARITY=0
LIVE_LOGS=1
LOG_INTERVAL=60
PRINT_FREQ="${VQATTACK_PRINT_FREQ:-50}"
SKIP_COMPLETED=1

usage() {
  cat <<'USAGE'
Usage:
  bash run_mscoco_eps_sweep_4gpu.sh \
    --mscoco-root /path/to/MSCOCO \
    --checkpoint-root /path/to/checkpoints/albef \
    --repo-root /path/to/ALBEF_attack \
    --work-root /path/to/output \
    --gpu-ids 4,5,6,7 \
    --train-images 300 \
    --val-images 100 \
    --eps-start 1 \
    --eps-end 32

Notes:
  Epsilon is interpreted in original pixel scale as k/255.
  ALBEF normalizes images to [-1, 1], so the runner exports:
    VQATTACK_PGD_EPS = 2 * k / 255
    VQATTACK_PGD_EPS_ITER = VQATTACK_PGD_EPS / 12.5
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) REPO_ROOT="$2"; shift 2 ;;
    --mscoco-root) MSCOCO_ROOT="$2"; shift 2 ;;
    --checkpoint-root) CHECKPOINT_ROOT="$2"; shift 2 ;;
    --work-root) WORK_ROOT="$2"; shift 2 ;;
    --gpu-ids) GPU_IDS="$2"; shift 2 ;;
    --train-images) TRAIN_IMAGES="$2"; shift 2 ;;
    --val-images) VAL_IMAGES="$2"; shift 2 ;;
    --eps-start) EPS_START="$2"; shift 2 ;;
    --eps-end) EPS_END="$2"; shift 2 ;;
    --eps-denom) EPS_DENOM="$2"; shift 2 ;;
    --eps-iter-divisor) EPS_ITER_DIVISOR="$2"; shift 2 ;;
    --log-interval) LOG_INTERVAL="$2"; shift 2 ;;
    --print-freq) PRINT_FREQ="$2"; shift 2 ;;
    --no-convert-images) CONVERT_IMAGES=0; shift ;;
    --enable-use-similarity) USE_SIMILARITY=1; shift ;;
    --no-live-logs) LIVE_LOGS=0; shift ;;
    --rerun-completed) SKIP_COMPLETED=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

PYTHON_BIN="${PYTHON:-python}"
RUNNER="${SCRIPT_DIR}/run_mscoco_vqattack_4gpu.sh"

if [[ ! -x "$RUNNER" && ! -f "$RUNNER" ]]; then
  echo "Missing runner: ${RUNNER}" >&2
  exit 1
fi

if [[ "$EPS_START" -lt 0 || "$EPS_END" -lt "$EPS_START" ]]; then
  echo "Invalid epsilon range: ${EPS_START}..${EPS_END}" >&2
  exit 2
fi

for required_path in \
  "$MSCOCO_ROOT" \
  "$REPO_ROOT" \
  "${REPO_ROOT}/VQA.py" \
  "${REPO_ROOT}/adv_attack.py" \
  "${CHECKPOINT_ROOT}/ALBEF.pth" \
  "${CHECKPOINT_ROOT}/vqa.pth"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Missing required path: ${required_path}" >&2
    exit 1
  fi
done

mkdir -p "$WORK_ROOT"

echo "MSCOCO root: ${MSCOCO_ROOT}"
echo "ALBEF VQAttack repo root: ${REPO_ROOT}"
echo "Checkpoint root: ${CHECKPOINT_ROOT}"
echo "Sweep work root: ${WORK_ROOT}"
echo "GPUs: ${GPU_IDS}"
echo "Subset images: train=${TRAIN_IMAGES}, val=${VAL_IMAGES}"
echo "Original-pixel epsilon sweep: ${EPS_START}/${EPS_DENOM} .. ${EPS_END}/${EPS_DENOM}"
echo "Normalized epsilon conversion: eps_norm = 2 * eps_pixel"
echo "Normalized step size: eps_iter_norm = eps_norm / ${EPS_ITER_DIVISOR}"

for eps_num in $(seq "$EPS_START" "$EPS_END"); do
  eps_label="$(printf "eps_%03d_%03d" "$eps_num" "$EPS_DENOM")"
  eps_root="${WORK_ROOT}/${eps_label}"
  final_manifest="${eps_root}/merged/original_format/attack_original_format_manifest.json"

  if [[ "$SKIP_COMPLETED" -eq 1 && -f "$final_manifest" ]]; then
    echo
    echo "===== Skipping ${eps_label}; final manifest already exists ====="
    continue
  fi

  read -r EPS_NORM EPS_ITER_NORM < <(
    "$PYTHON_BIN" -c 'import sys
num = int(sys.argv[1])
denom = float(sys.argv[2])
divisor = float(sys.argv[3])
eps = 2.0 * num / denom
print(f"{eps:.12f} {eps / divisor:.12f}")' \
      "$eps_num" "$EPS_DENOM" "$EPS_ITER_DIVISOR"
  )

  mkdir -p "$eps_root"
  cat > "${eps_root}/epsilon_config.json" <<EOF
{
  "epsilon_label": "${eps_label}",
  "pixel_epsilon": "${eps_num}/${EPS_DENOM}",
  "normalized_epsilon": ${EPS_NORM},
  "normalized_epsilon_iter": ${EPS_ITER_NORM},
  "epsilon_iter_divisor": ${EPS_ITER_DIVISOR},
  "train_images": ${TRAIN_IMAGES},
  "val_images": ${VAL_IMAGES},
  "gpu_ids": "${GPU_IDS}"
}
EOF

  export VQATTACK_PGD_EPS="$EPS_NORM"
  export VQATTACK_PGD_EPS_ITER="$EPS_ITER_NORM"
  export VQATTACK_PRINT_FREQ="$PRINT_FREQ"

  echo
  echo "===== Starting ${eps_label}: pixel_eps=${eps_num}/${EPS_DENOM}, normalized_eps=${EPS_NORM}, normalized_eps_iter=${EPS_ITER_NORM} ====="

  runner_args=(
    --repo-root "$REPO_ROOT"
    --mscoco-root "$MSCOCO_ROOT"
    --checkpoint-root "$CHECKPOINT_ROOT"
    --work-root "$eps_root"
    --gpu-ids "$GPU_IDS"
    --splits "train,val"
    --max-images-per-split "train=${TRAIN_IMAGES},val=${VAL_IMAGES}"
    --log-interval "$LOG_INTERVAL"
    --print-freq "$PRINT_FREQ"
  )

  if [[ "$CONVERT_IMAGES" -eq 1 ]]; then
    runner_args+=(--convert-images)
  fi
  if [[ "$USE_SIMILARITY" -eq 1 ]]; then
    runner_args+=(--enable-use-similarity)
  fi
  if [[ "$LIVE_LOGS" -eq 0 ]]; then
    runner_args+=(--no-live-logs)
  fi

  bash "$RUNNER" "${runner_args[@]}"
  echo "===== Finished ${eps_label} ====="
done

echo "MSCOCO epsilon sweep completed: ${WORK_ROOT}"
