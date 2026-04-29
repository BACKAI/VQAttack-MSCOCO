#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_BASE="$(cd "${SCRIPT_DIR}/.." && pwd)"
VLM_ROOT="$(cd "${REPO_BASE}/../.." && pwd)"

REPO_ROOT="${REPO_BASE}/ALBEF_VQAttack/ALBEF_attack"
TEXTVQA_ROOT="${VLM_ROOT}/dataset/textvqa"
ORIGINAL_FORMAT_DIR="${TEXTVQA_ROOT}/original_format"
CHECKPOINT_ROOT="${REPO_BASE}/checkpoints/albef"
WORK_ROOT="${TEXTVQA_ROOT}/vqattack_albef_4gpu"
OUTPUT_ROOT="${WORK_ROOT}/merged"
FINAL_ORIGINAL_FORMAT_DIR="${OUTPUT_ROOT}/original_format"
FINAL_ORIGINAL_FORMAT_DIR_SET=0
GPU_IDS="0,1,2,3"
NUM_SHARDS=4
LIMIT=0
CONVERT_IMAGES=0
SKIP_PREPARE=0
USE_SIMILARITY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) REPO_ROOT="$2"; shift 2 ;;
    --textvqa-root) TEXTVQA_ROOT="$2"; ORIGINAL_FORMAT_DIR="${TEXTVQA_ROOT}/original_format"; shift 2 ;;
    --original-format-dir) ORIGINAL_FORMAT_DIR="$2"; shift 2 ;;
    --checkpoint-root) CHECKPOINT_ROOT="$2"; shift 2 ;;
    --work-root) WORK_ROOT="$2"; OUTPUT_ROOT="${WORK_ROOT}/merged"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --final-original-format-dir) FINAL_ORIGINAL_FORMAT_DIR="$2"; FINAL_ORIGINAL_FORMAT_DIR_SET=1; shift 2 ;;
    --gpu-ids) GPU_IDS="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --format) shift 2 ;;
    --convert-images) CONVERT_IMAGES=1; shift ;;
    --skip-prepare) SKIP_PREPARE=1; shift ;;
    --enable-use-similarity) USE_SIMILARITY=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$FINAL_ORIGINAL_FORMAT_DIR_SET" -eq 0 ]]; then
  FINAL_ORIGINAL_FORMAT_DIR="${OUTPUT_ROOT}/original_format"
fi

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if [[ "${#GPUS[@]}" -ne "${NUM_SHARDS}" ]]; then
  echo "--gpu-ids must contain exactly ${NUM_SHARDS} comma-separated GPU ids, got: ${GPU_IDS}" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON:-python}"
SHARD_ROOT="${WORK_ROOT}/shards"
mkdir -p "$SHARD_ROOT" "$OUTPUT_ROOT"

for shard_index in $(seq 0 $((NUM_SHARDS - 1))); do
  shard_name="$(printf "shard_%02d" "$shard_index")"
  shard_tag="$(printf "full_shard%02d" "$shard_index")"
  shard_dir="${SHARD_ROOT}/${shard_name}"
  eval_json="${shard_dir}/data/textvqa_train_val_${shard_tag}.json"
  answer_list="${shard_dir}/data/answer_list.json"
  config_path="${shard_dir}/configs/TextVQA_${shard_tag}.yaml"

  if [[ "$SKIP_PREPARE" -eq 0 ]]; then
    "$PYTHON_BIN" "${SCRIPT_DIR}/prepare_textvqa_vqattack_albef.py" \
      --original-format-dir "$ORIGINAL_FORMAT_DIR" \
      --output-dir "$shard_dir" \
      --splits "train,val" \
      --tag "$shard_tag" \
      --limit "$LIMIT" \
      --num-shards "$NUM_SHARDS" \
      --shard-index "$shard_index"

    "$PYTHON_BIN" "${SCRIPT_DIR}/write_textvqa_vqattack_config.py" \
      --output "$config_path" \
      --eval-json "$eval_json" \
      --answer-list "$answer_list" \
      --image-root "$ORIGINAL_FORMAT_DIR"
  fi
done

declare -a PIDS=()
for shard_index in $(seq 0 $((NUM_SHARDS - 1))); do
  shard_name="$(printf "shard_%02d" "$shard_index")"
  shard_tag="$(printf "full_shard%02d" "$shard_index")"
  shard_dir="${SHARD_ROOT}/${shard_name}"
  config_path="${shard_dir}/configs/TextVQA_${shard_tag}.yaml"
  gpu_id="${GPUS[$shard_index]}"

  (
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    export VQATTACK_DISABLE_USE="$([[ "$USE_SIMILARITY" -eq 1 ]] && echo 0 || echo 1)"
    export VQATTACK_LIMIT="0"
    export VQATTACK_ALBEF_PRETRAIN_CKPT="${CHECKPOINT_ROOT}/ALBEF.pth"
    export VQATTACK_ALBEF_VQA_CKPT="${CHECKPOINT_ROOT}/vqa.pth"
    export VQATTACK_RIGHT_PART_FILES="${shard_dir}/attack_assets/right_part_textvqa_${shard_tag}.txt"
    export VQATTACK_VILT_ANS_FILES="${shard_dir}/attack_assets/vilt_ans_table_textvqa_${shard_tag}.json"
    export VQATTACK_TARGET_ANS_FILES="${shard_dir}/attack_assets/albef_ans_table_textvqa_${shard_tag}.json"
    export VQATTACK_CHATGPT_FILES="${shard_dir}/attack_assets/chatgpt_identity_textvqa_${shard_tag}.json"
    export VQATTACK_ALL_CORRECT_FILES="${shard_dir}/attack_assets/all_correct_ans_textvqa_${shard_tag}.json"
    export VQATTACK_ADV_IMG_DIR="${shard_dir}/attack_outputs/attack_dir"
    export VQATTACK_ADV_TXT_PATH="${shard_dir}/attack_outputs/adv_txt.json"
    cd "$REPO_ROOT"
    "$PYTHON_BIN" VQA.py \
      --config "$config_path" \
      --config_pre "${REPO_ROOT}/configs/Pretrain.yaml" \
      --output_dir "${shard_dir}/run_output" \
      > "${shard_dir}/run.log" 2>&1
  ) &
  PIDS+=("$!")
  echo "Started ${shard_name} on GPU ${gpu_id}, pid ${PIDS[-1]}"
done

for pid in "${PIDS[@]}"; do
  wait "$pid"
done

"$PYTHON_BIN" "${SCRIPT_DIR}/merge_vqattack_shards.py" \
  --shard-root "$SHARD_ROOT" \
  --output-dir "$OUTPUT_ROOT" \
  --num-shards "$NUM_SHARDS" \
  --overwrite

if [[ "$CONVERT_IMAGES" -eq 1 ]]; then
  "$PYTHON_BIN" "${SCRIPT_DIR}/write_attacked_textvqa_original_format.py" \
    --original-format-dir "$ORIGINAL_FORMAT_DIR" \
    --metadata-json "${OUTPUT_ROOT}/data/textvqa_train_val_full.json" \
    --adv-text-json "${OUTPUT_ROOT}/attack_outputs/adv_txt_full.json" \
    --tensor-dir "${OUTPUT_ROOT}/attack_outputs/attack_dir_full" \
    --output-dir "$FINAL_ORIGINAL_FORMAT_DIR" \
    --splits "train,val" \
    --image-selection "first" \
    --overwrite
fi
