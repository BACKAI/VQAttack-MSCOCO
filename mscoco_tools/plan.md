# MSCOCO VQAttack 실행 계획

## 목표

TextVQA와 같은 ALBEF VQAttack 흐름을 MSCOCO VQA v2에 적용한다. 기본 full run은 train/val question 전체를 공격하고, 추가로 validation image 3,166장 subset을 선택해 그 이미지에 연결된 모든 question을 공격할 수 있어야 한다.

## 범위

- 서버 실행 폴더: `/var/tmp/jnuadmin_vlm/VLM/dataset/MSCOCO/code`
- 서버 dataset root: `/var/tmp/jnuadmin_vlm/VLM/dataset/MSCOCO`
- 서버 checkpoint root: `/var/tmp/jnuadmin_vlm/VLM/Attack/VQAttack/checkpoints/albef`
- MSCOCO full output root: `/var/tmp/jnuadmin_vlm/VLM/outputs/mscoco_vqattack_4gpu`
- MSCOCO validation subset output root 권장값: `/var/tmp/jnuadmin_vlm/VLM/outputs/mscoco_vqattack_val3166_4gpu`

## 주요 가정

- large dataset/checkpoint/output은 GitHub에 올리지 않는다.
- ALBEF checkpoint는 TextVQA 서버 실행에서 이미 받은 것을 재사용한다.
- train/val 공격 단위는 image가 아니라 question이다.
- image 기준 subset을 고를 때도 선택된 image에 연결된 모든 question을 유지한다.

## 구현 방향

- `prepare_mscoco_vqattack_albef.py`
  - `--max-images-per-split val=3166` 옵션을 추가한다.
  - split별로 먼저 image subset을 고른 뒤, 해당 image의 모든 question record를 유지한다.
- `run_mscoco_vqattack_4gpu.sh`
  - `--max-images-per-split` 옵션을 prepare 단계로 전달한다.
  - finalization은 metadata 기준 subset 출력을 지원한다.
- `write_attacked_mscoco_original_format.py`
  - `--subset-to-metadata` 옵션으로 metadata에 있는 question/annotation/image만 출력한다.
- `validate_mscoco_vqattack_output.py`
  - `--metadata-json` 옵션으로 full source가 아니라 metadata subset 기준 검증을 수행한다.

## 실행 예시

MSCOCO validation image 3,166장과 그에 연결된 모든 question만 공격:

```bash
cd /var/tmp/jnuadmin_vlm/VLM/dataset/MSCOCO/code

bash run_mscoco_vqattack_4gpu.sh \
  --mscoco-root /var/tmp/jnuadmin_vlm/VLM/dataset/MSCOCO \
  --checkpoint-root /var/tmp/jnuadmin_vlm/VLM/Attack/VQAttack/checkpoints/albef \
  --work-root /var/tmp/jnuadmin_vlm/VLM/outputs/mscoco_vqattack_val3166_4gpu \
  --gpu-ids 4,5,6,7 \
  --splits val \
  --max-images-per-split val=3166 \
  --convert-images \
  --log-interval 60
```

## 완료 기준

- prepare manifest에서 `image_counts.val == 3166`이어야 한다.
- 같은 manifest에서 `split_counts.val`은 선택된 3,166장 image에 연결된 question 수여야 한다.
- local 확인 기준 현재 MSCOCO val 첫 3,166장 image는 `16,981`개 question을 가진다.
- 최종 output validation JSON에는 공격된 question과 `original_question`이 있어야 한다.
- 최종 output annotation JSON과 image 파일은 metadata subset 기준으로 검증되어야 한다.

## 검증

- Python 문법 검증:
  - `prepare_mscoco_vqattack_albef.py`
  - `write_attacked_mscoco_original_format.py`
  - `validate_mscoco_vqattack_output.py`
  - `check_mscoco_vqattack_progress.py`
- Bash 문법 검증:
  - `run_mscoco_vqattack_4gpu.sh`
- Local prepare 확인:
  - `--max-images-per-split val=3` 결과: image 3장, question 10개
  - `--max-images-per-split val=3166` 결과: image 3,166장, question 16,981개

## 진행률 확인

현재 MSCOCO 공격 진행률은 다음 스크립트로 확인한다.

```bash
cd /var/tmp/jnuadmin_vlm/VLM/dataset/MSCOCO/code
conda activate vqattack-textvqa

python check_mscoco_vqattack_progress.py \
  --work-root /var/tmp/jnuadmin_vlm/VLM/outputs/mscoco_vqattack_4gpu
```

validation image 3,166장 subset run은 `--work-root`만 subset output으로 바꾼다.

## VQAv2/MSCOCO epsilon sweep 실행 계획

TextVQA epsilon sweep와 같은 조건으로 VQAv2/MSCOCO subset을 공격한다.

- 대상: VQAv2 train에서 앞쪽 unique image 300장, val에서 앞쪽 unique image 100장
- question 범위: 선택된 image에 연결된 모든 question
- epsilon 범위: 원본 pixel 기준 `1/255`부터 `32/255`
- ALBEF 입력 정규화: image가 `[-1, 1]` 범위이므로 실행 시 `VQATTACK_PGD_EPS=2*k/255`로 변환
- step size: `VQATTACK_PGD_EPS_ITER=(2*k/255)/12.5`
- 출력: `eps_001_255`부터 `eps_032_255`까지 epsilon별 폴더를 만들고, 최종 사용 데이터는 각 폴더의 `merged/original_format`에 저장

새 실행 래퍼는 `run_mscoco_eps_sweep_4gpu.sh`이다. 내부적으로 기존 `run_mscoco_vqattack_4gpu.sh`를 epsilon마다 호출하여 prepare, attack, merge, original format export, validation을 end-to-end로 수행한다.
