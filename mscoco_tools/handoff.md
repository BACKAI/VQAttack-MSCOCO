# MSCOCO VQAttack 핸드오프

## 변경 내용

- MSCOCO VQA v2 train/validation용 VQAttack 실행 파이프라인을 `D:\VLM\dataset\MSCOCO\code` 내부에 작성했다.
- GitHub에는 코드만 올릴 수 있도록 dataset/checkpoint를 생성하거나 포함하지 않는다.
- 서버에서는 MSCOCO dataset만 직접 다운로드하고, ALBEF checkpoint는 기존 TextVQA 실행에서 쓰는 경로를 재사용한다.
- 기본 GPU를 `4,5,6,7`로 설정했다.
- train/val의 모든 question을 개별 공격 record로 변환하므로 image 하나에 question이 여러 개 있어도 전부 공격 대상이 된다.

## 작성 파일

- `download_mscoco_vqa_assets.sh`
- `verify_mscoco_vqa_assets.py`
- `mscoco_vqa_common.py`
- `prepare_mscoco_vqattack_albef.py`
- `write_mscoco_vqattack_config.py`
- `merge_mscoco_vqattack_shards.py`
- `write_attacked_mscoco_original_format.py`
- `validate_mscoco_vqattack_output.py`
- `run_mscoco_vqattack_4gpu.sh`
- `plan.md`
- `handoff.md`

## 서버 dataset 다운로드

가중치는 다운로드하지 않는다. 아래 명령은 MSCOCO image/question/annotation만 `/var/tmp/jnuadmin_vlm/VLM/dataset/MSCOCO`에 받는다.

```bash
cd /var/tmp/jnuadmin_vlm/VLM/dataset/MSCOCO/code

bash download_mscoco_vqa_assets.sh \
  --mscoco-root /var/tmp/jnuadmin_vlm/VLM/dataset/MSCOCO
```

## Smoke 실행

```bash
cd /var/tmp/jnuadmin_vlm/VLM/dataset/MSCOCO/code

bash run_mscoco_vqattack_4gpu.sh \
  --mscoco-root /var/tmp/jnuadmin_vlm/VLM/dataset/MSCOCO \
  --checkpoint-root /var/tmp/jnuadmin_vlm/VLM/Attack/VQAttack/checkpoints/albef \
  --work-root /var/tmp/jnuadmin_vlm/VLM/outputs/mscoco_vqattack_smoke \
  --gpu-ids 4,5,6,7 \
  --limit 2 \
  --convert-images \
  --log-interval 20
```

`--limit`이 0보다 크면 partial smoke 결과이므로 최종 original-format 변환은 자동으로 건너뛴다.

## Full 실행

```bash
cd /var/tmp/jnuadmin_vlm/VLM/dataset/MSCOCO/code

bash run_mscoco_vqattack_4gpu.sh \
  --mscoco-root /var/tmp/jnuadmin_vlm/VLM/dataset/MSCOCO \
  --checkpoint-root /var/tmp/jnuadmin_vlm/VLM/Attack/VQAttack/checkpoints/albef \
  --work-root /var/tmp/jnuadmin_vlm/VLM/outputs/mscoco_vqattack_4gpu \
  --gpu-ids 4,5,6,7 \
  --convert-images \
  --log-interval 60
```

## 예상 최종 출력

```text
/var/tmp/jnuadmin_vlm/VLM/outputs/mscoco_vqattack_4gpu/merged/original_format/
  train2014/
  val2014/
  v2_OpenEnded_mscoco_train2014_questions.json
  v2_OpenEnded_mscoco_val2014_questions.json
  v2_mscoco_train2014_annotations.json
  v2_mscoco_val2014_annotations.json
  attack_original_format_manifest.json
```

## 진행 확인

```bash
nvidia-smi

tail -n 80 /var/tmp/jnuadmin_vlm/VLM/outputs/mscoco_vqattack_4gpu/shards/shard_00/run.log
tail -n 80 /var/tmp/jnuadmin_vlm/VLM/outputs/mscoco_vqattack_4gpu/shards/shard_01/run.log
tail -n 80 /var/tmp/jnuadmin_vlm/VLM/outputs/mscoco_vqattack_4gpu/shards/shard_02/run.log
tail -n 80 /var/tmp/jnuadmin_vlm/VLM/outputs/mscoco_vqattack_4gpu/shards/shard_03/run.log
```

## 검증 상태

- 실제 서버 다운로드와 GPU 공격 실행은 아직 하지 않았다.
- 로컬 Anaconda Python `3.12.7`로 Python 파일 AST 문법 검사를 통과했다.
- Git Bash로 `run_mscoco_vqattack_4gpu.sh`의 `bash -n` 검사를 통과했다.

## 주의 사항

- `sudo`를 사용하지 않는다.
- `/home`에 큰 output을 쓰지 않는다.
- checkpoint는 새로 받지 않는다.
- large dataset/checkpoint/output은 GitHub에 올리지 않는다.
