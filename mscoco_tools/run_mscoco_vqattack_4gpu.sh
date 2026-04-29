#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MSCOCO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VLM_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
REPO_BASE="${VLM_ROOT}/Attack/VQAttack"
REPO_ROOT="${REPO_BASE}/ALBEF_VQAttack/ALBEF_attack"
CHECKPOINT_ROOT="${REPO_BASE}/checkpoints/albef"
WORK_ROOT="${VLM_ROOT}/outputs/mscoco_vqattack_4gpu"
OUTPUT_ROOT="${WORK_ROOT}/merged"
FINAL_ORIGINAL_FORMAT_DIR="${OUTPUT_ROOT}/original_format"
FINAL_ORIGINAL_FORMAT_DIR_SET=0
GPU_IDS="4,5,6,7"
NUM_SHARDS=4
LIMIT=0
SPLITS="train,val"
MAX_IMAGES_PER_SPLIT=""
CONVERT_IMAGES=0
SKIP_PREPARE=0
USE_SIMILARITY=0
LIVE_LOGS=1
LOG_INTERVAL=60
PRINT_FREQ="${VQATTACK_PRINT_FREQ:-50}"
VALIDATE_FINAL=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) REPO_ROOT="$2"; shift 2 ;;
    --mscoco-root) MSCOCO_ROOT="$2"; shift 2 ;;
    --checkpoint-root) CHECKPOINT_ROOT="$2"; shift 2 ;;
    --work-root) WORK_ROOT="$2"; OUTPUT_ROOT="${WORK_ROOT}/merged"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --final-original-format-dir) FINAL_ORIGINAL_FORMAT_DIR="$2"; FINAL_ORIGINAL_FORMAT_DIR_SET=1; shift 2 ;;
    --gpu-ids) GPU_IDS="$2"; shift 2 ;;
    --num-shards) NUM_SHARDS="$2"; shift 2 ;;
    --splits) SPLITS="$2"; shift 2 ;;
    --max-images-per-split) MAX_IMAGES_PER_SPLIT="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --convert-images) CONVERT_IMAGES=1; shift ;;
    --skip-prepare) SKIP_PREPARE=1; shift ;;
    --enable-use-similarity) USE_SIMILARITY=1; shift ;;
    --log-interval) LOG_INTERVAL="$2"; shift 2 ;;
    --no-live-logs) LIVE_LOGS=0; shift ;;
    --print-freq) PRINT_FREQ="$2"; shift 2 ;;
    --no-validate-final) VALIDATE_FINAL=0; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$FINAL_ORIGINAL_FORMAT_DIR_SET" -eq 0 ]]; then
  FINAL_ORIGINAL_FORMAT_DIR="${OUTPUT_ROOT}/original_format"
fi

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if [[ "${#GPUS[@]}" -ne "$NUM_SHARDS" ]]; then
  echo "--gpu-ids must contain exactly ${NUM_SHARDS} comma-separated GPU ids, got: ${GPU_IDS}" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON:-python}"
SHARD_ROOT="${WORK_ROOT}/shards"
SPLIT_TAG="${SPLITS//,/_}"
SPLIT_TAG="${SPLIT_TAG// /}"
mkdir -p "$SHARD_ROOT" "$OUTPUT_ROOT"

echo "MSCOCO root: ${MSCOCO_ROOT}"
echo "ALBEF VQAttack repo root: ${REPO_ROOT}"
echo "Checkpoint root: ${CHECKPOINT_ROOT}"
echo "Work root: ${WORK_ROOT}"
echo "Splits: ${SPLITS}"
echo "Max images per split: ${MAX_IMAGES_PER_SPLIT:-none}"
echo "GPUs: ${GPU_IDS}"
echo "Limit per shard: ${LIMIT}"

for required_path in \
  "${MSCOCO_ROOT}/train2014" \
  "${MSCOCO_ROOT}/val2014" \
  "${MSCOCO_ROOT}/v2_OpenEnded_mscoco_train2014_questions.json" \
  "${MSCOCO_ROOT}/v2_OpenEnded_mscoco_val2014_questions.json" \
  "${MSCOCO_ROOT}/v2_mscoco_train2014_annotations.json" \
  "${MSCOCO_ROOT}/v2_mscoco_val2014_annotations.json" \
  "${CHECKPOINT_ROOT}/ALBEF.pth" \
  "${CHECKPOINT_ROOT}/vqa.pth"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Missing required path: ${required_path}" >&2
    echo "Download only the MSCOCO dataset with download_mscoco_vqa_assets.sh; reuse existing ALBEF checkpoints from ${CHECKPOINT_ROOT}." >&2
    exit 1
  fi
done

for shard_index in $(seq 0 $((NUM_SHARDS - 1))); do
  shard_name="$(printf "shard_%02d" "$shard_index")"
  shard_tag="$(printf "full_shard%02d" "$shard_index")"
  shard_dir="${SHARD_ROOT}/${shard_name}"
  eval_json="${shard_dir}/data/mscoco_${SPLIT_TAG}_${shard_tag}.json"
  answer_list="${shard_dir}/data/answer_list.json"
  config_path="${shard_dir}/configs/MSCOCO_${shard_tag}.yaml"

  if [[ "$SKIP_PREPARE" -eq 0 ]]; then
    "$PYTHON_BIN" "${SCRIPT_DIR}/prepare_mscoco_vqattack_albef.py" \
      --mscoco-root "$MSCOCO_ROOT" \
      --output-dir "$shard_dir" \
      --splits "$SPLITS" \
      --max-images-per-split "$MAX_IMAGES_PER_SPLIT" \
      --tag "$shard_tag" \
      --limit "$LIMIT" \
      --num-shards "$NUM_SHARDS" \
      --shard-index "$shard_index"

    "$PYTHON_BIN" "${SCRIPT_DIR}/write_mscoco_vqattack_config.py" \
      --output "$config_path" \
      --eval-json "$eval_json" \
      --answer-list "$answer_list" \
      --image-root "$MSCOCO_ROOT"
  fi
done

declare -a PIDS=()

stop_children() {
  echo "Stopping shard processes..." >&2
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" >/dev/null 2>&1 || true
  done
  exit 130
}
trap stop_children INT TERM

for shard_index in $(seq 0 $((NUM_SHARDS - 1))); do
  shard_name="$(printf "shard_%02d" "$shard_index")"
  shard_tag="$(printf "full_shard%02d" "$shard_index")"
  shard_dir="${SHARD_ROOT}/${shard_name}"
  config_path="${shard_dir}/configs/MSCOCO_${shard_tag}.yaml"
  gpu_id="${GPUS[$shard_index]}"

  (
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    export VQATTACK_DISABLE_USE="$([[ "$USE_SIMILARITY" -eq 1 ]] && echo 0 || echo 1)"
    export VQATTACK_LIMIT="0"
    export VQATTACK_PRINT_FREQ="$PRINT_FREQ"
    export VQATTACK_ALBEF_PRETRAIN_CKPT="${CHECKPOINT_ROOT}/ALBEF.pth"
    export VQATTACK_ALBEF_VQA_CKPT="${CHECKPOINT_ROOT}/vqa.pth"
    export VQATTACK_RIGHT_PART_FILES="${shard_dir}/attack_assets/right_part_mscoco_${shard_tag}.txt"
    export VQATTACK_VILT_ANS_FILES="${shard_dir}/attack_assets/vilt_ans_table_mscoco_${shard_tag}.json"
    export VQATTACK_TARGET_ANS_FILES="${shard_dir}/attack_assets/albef_ans_table_mscoco_${shard_tag}.json"
    export VQATTACK_CHATGPT_FILES="${shard_dir}/attack_assets/chatgpt_identity_mscoco_${shard_tag}.json"
    export VQATTACK_ALL_CORRECT_FILES="${shard_dir}/attack_assets/all_correct_ans_mscoco_${shard_tag}.json"
    export VQATTACK_ADV_IMG_DIR="${shard_dir}/attack_outputs/attack_dir"
    export VQATTACK_ADV_TXT_PATH="${shard_dir}/attack_outputs/adv_txt.json"
    cd "$REPO_ROOT"
    {
      echo "Started ${shard_name} on GPU ${gpu_id} at $(date -Is)"
      "$PYTHON_BIN" VQA.py \
        --config "$config_path" \
        --config_pre "${REPO_ROOT}/configs/Pretrain.yaml" \
        --output_dir "${shard_dir}/run_output"
    } > "${shard_dir}/run.log" 2>&1
  ) &
  PIDS+=("$!")
  echo "Started ${shard_name} on GPU ${gpu_id}, pid ${PIDS[-1]}"
done

print_live_logs() {
  echo
  echo "===== MSCOCO VQAttack live logs $(date -Is) ====="
  for shard_index in $(seq 0 $((NUM_SHARDS - 1))); do
    shard_name="$(printf "shard_%02d" "$shard_index")"
    log_path="${SHARD_ROOT}/${shard_name}/run.log"
    echo "----- ${shard_name}: ${log_path} -----"
    if [[ -f "$log_path" ]]; then
      tail -n 20 "$log_path"
    else
      echo "Log not created yet."
    fi
  done
  echo "===== end live logs ====="
  echo
}

if [[ "$LIVE_LOGS" -eq 1 ]]; then
  while true; do
    alive=0
    for pid in "${PIDS[@]}"; do
      if ps -p "$pid" >/dev/null 2>&1; then
        alive=1
        break
      fi
    done
    if [[ "$alive" -eq 0 ]]; then
      break
    fi
    sleep "$LOG_INTERVAL"
    print_live_logs
  done
fi

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
trap - INT TERM

if [[ "$status" -ne 0 ]]; then
  echo "At least one shard failed. Showing final log tails." >&2
  print_live_logs >&2
  exit "$status"
fi

"$PYTHON_BIN" "${SCRIPT_DIR}/merge_mscoco_vqattack_shards.py" \
  --shard-root "$SHARD_ROOT" \
  --output-dir "$OUTPUT_ROOT" \
  --num-shards "$NUM_SHARDS" \
  --overwrite

if [[ "$CONVERT_IMAGES" -eq 1 ]]; then
  if [[ "$LIMIT" -gt 0 ]]; then
    echo "Skipping final original-format finalization because --limit ${LIMIT} creates partial output."
  else
    "$PYTHON_BIN" "${SCRIPT_DIR}/write_attacked_mscoco_original_format.py" \
      --mscoco-root "$MSCOCO_ROOT" \
      --metadata-json "${OUTPUT_ROOT}/data/mscoco_train_val_full.json" \
      --adv-text-json "${OUTPUT_ROOT}/attack_outputs/adv_txt_full.json" \
      --tensor-dir "${OUTPUT_ROOT}/attack_outputs/attack_dir_full" \
      --output-dir "$FINAL_ORIGINAL_FORMAT_DIR" \
      --splits "$SPLITS" \
      --image-selection "first" \
      --subset-to-metadata \
      --overwrite

    if [[ "$VALIDATE_FINAL" -eq 1 ]]; then
      "$PYTHON_BIN" "${SCRIPT_DIR}/validate_mscoco_vqattack_output.py" \
        --mscoco-root "$MSCOCO_ROOT" \
        --output-dir "$FINAL_ORIGINAL_FORMAT_DIR" \
        --metadata-json "${OUTPUT_ROOT}/data/mscoco_train_val_full.json" \
        --splits "$SPLITS"
    fi
  fi
fi
