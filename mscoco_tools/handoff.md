# MSCOCO VQAttack 인계 메모

## 변경 사항

- MSCOCO validation image subset 공격 옵션을 추가했다.
- 새 옵션:
  - `--max-images-per-split val=3166`
- 이 옵션은 question 3,166개를 자르는 것이 아니라, validation image 3,166장을 선택한 뒤 그 image에 연결된 모든 question을 공격 대상으로 유지한다.

## 수정된 파일

- `prepare_mscoco_vqattack_albef.py`
  - split별 image limit parser 추가
  - image subset 선택 후 해당 image의 모든 question 유지
  - manifest에 `max_images_per_split` 기록
- `run_mscoco_vqattack_4gpu.sh`
  - `--max-images-per-split` 인자 추가
  - prepare 단계로 전달
  - finalization/validation을 metadata subset 기준으로 수행
- `write_attacked_mscoco_original_format.py`
  - `--subset-to-metadata` 추가
  - metadata에 포함된 question/annotation/image만 출력 가능
- `validate_mscoco_vqattack_output.py`
  - `--metadata-json` 추가
  - full source 대신 subset metadata 기준 검증 가능
- `check_mscoco_vqattack_progress.py`
  - shard metadata, `adv_txt.json`, tensor output, `run.log`를 읽어 split별 진행률과 ETA를 출력

## 검증 결과

- Python 문법 검증 통과:
  - `prepare_mscoco_vqattack_albef.py`
  - `write_attacked_mscoco_original_format.py`
  - `validate_mscoco_vqattack_output.py`
  - `check_mscoco_vqattack_progress.py`
- Bash 문법 검증 통과:
  - `run_mscoco_vqattack_4gpu.sh`
- Local prepare 테스트:
  - `val=3` 결과: image 3장, question 10개
  - `val=3166` 결과: image 3,166장, question 16,981개

## 서버 적용 절차

GitHub repo에 반영한 뒤 서버에서:

```bash
cd /var/tmp/jnuadmin_vlm/VLM/Attack/VQAttack-MSCOCO
git pull

cp -a /var/tmp/jnuadmin_vlm/VLM/Attack/VQAttack-MSCOCO/mscoco_tools/. \
  /var/tmp/jnuadmin_vlm/VLM/dataset/MSCOCO/code/
```

MSCOCO validation image 3,166장 subset 공격:

```bash
cd /var/tmp/jnuadmin_vlm/VLM/dataset/MSCOCO/code
conda activate vqattack-textvqa

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

## 남은 주의점

- 현재 서버에서 full MSCOCO train attack이 돌고 있다면, validation subset run은 사용자가 직접 멈춘 뒤 실행해야 한다.
- 기존 full train/val output과 섞이지 않도록 `--work-root`는 반드시 새 경로를 사용한다.
- `--limit`은 question 기준 smoke용이다. 이번 validation 3,166장 subset에는 `--limit`을 쓰지 않는다.
- 진행률에서 `images any complete question`은 image에 question 하나 이상이 저장된 기준이고, `images all questions complete`은 해당 image의 모든 question이 저장된 엄격한 기준이다.

## VQAv2/MSCOCO epsilon sweep 추가

- `ALBEF_VQAttack/ALBEF_attack/adv_attack.py`
  - PGD epsilon과 step size를 `VQATTACK_PGD_EPS`, `VQATTACK_PGD_EPS_ITER` 환경변수로 주입할 수 있게 변경했다.
  - 기본값은 기존 동작과 같게 `0.125`, `0.01`로 유지했다.
  - 실행 로그에 실제 PGD epsilon 값을 출력한다.
- `mscoco_tools/run_mscoco_eps_sweep_4gpu.sh`
  - `1/255`부터 `32/255`까지 epsilon별로 기존 4GPU runner를 반복 실행한다.
  - 기본 subset은 train image 300장, val image 100장이며 선택 image의 모든 question을 포함한다.
  - 각 epsilon 결과는 `eps_XXX_255/merged/original_format`에 image와 VQAv2 question/annotation JSON으로 저장된다.

남은 확인:
- 서버에서 smoke run으로 `eps_001_255` 1개를 먼저 돌려 `VQAttack PGD eps=...` 로그와 manifest count를 확인한다.
- smoke가 통과하면 tmux에서 full sweep을 실행한다.

로컬 검증:
- `python -m py_compile ALBEF_VQAttack/ALBEF_attack/adv_attack.py` 통과
- `bash -n mscoco_tools/run_mscoco_eps_sweep_4gpu.sh` 통과
