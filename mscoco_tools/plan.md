# MSCOCO VQAttack 실행 계획

## 목표

TextVQA에서 쓰던 것과 같은 ALBEF VQAttack 공격 코드를 MSCOCO VQA v2 train/validation 데이터에 적용한다. GitHub에는 코드만 올리고, 서버에서는 MSCOCO dataset만 추가 다운로드한다. ALBEF 가중치는 기존 TextVQA 실행에서 사용 중인 checkpoint 경로를 그대로 재사용한다.

## 범위

- 실행 코드 위치:
  - `D:\VLM\dataset\MSCOCO\code`
- 서버 기준 코드 위치:
  - `/var/tmp/jnuadmin_vlm/VLM/dataset/MSCOCO/code`
- 서버 기준 dataset 위치:
  - `/var/tmp/jnuadmin_vlm/VLM/dataset/MSCOCO`
- 서버 기준 VQAttack repo:
  - `/var/tmp/jnuadmin_vlm/VLM/Attack/VQAttack`
- 서버 기준 checkpoint:
  - `/var/tmp/jnuadmin_vlm/VLM/Attack/VQAttack/checkpoints/albef/ALBEF.pth`
  - `/var/tmp/jnuadmin_vlm/VLM/Attack/VQAttack/checkpoints/albef/vqa.pth`
- 서버 기준 output:
  - `/var/tmp/jnuadmin_vlm/VLM/outputs/mscoco_vqattack_4gpu`

## 공격 대상

- `train2014` image + `v2_OpenEnded_mscoco_train2014_questions.json`
- `val2014` image + `v2_OpenEnded_mscoco_val2014_questions.json`
- 대응 annotation JSON은 VQAttack answer asset 생성을 위해 사용하고, 최종 출력에는 원본 그대로 복사한다.
- test split은 공격하지 않는다.
- image 하나에 question이 여러 개 있으면 모든 question을 개별 record로 만들어 전부 공격한다.
- 최종 이미지 파일은 원본 image 이름과 개수를 유지하고, image당 대표 attacked tensor 하나를 선택해 저장한다.

## GPU 배치

- 기본 실행 GPU는 `4,5,6,7`이다.
- shard 수는 기본 4개다.
- shard별 프로세스는 각각 GPU 하나만 보도록 `CUDA_VISIBLE_DEVICES`를 다시 지정한다.

## 서버 준비 흐름

1. GitHub에서 코드만 clone/pull한다.
2. 서버에서 MSCOCO dataset만 직접 다운로드한다.
3. 기존 TextVQA 실행에 사용한 ALBEF checkpoint 경로를 확인한다.
4. `--limit 2` smoke 실행으로 4개 GPU 분산과 로그 출력을 확인한다.
5. full 실행으로 train/val 전체 question을 공격한다.
6. merge와 finalization 후 검증 스크립트를 실행한다.

## 완료 기준

- train/val question JSON의 모든 question에 공격 결과가 존재한다.
- 최종 question JSON의 question 수가 원본과 같다.
- 각 question 항목에 `original_question`이 보존된다.
- annotation JSON의 annotation 수가 원본과 같다.
- train/val 이미지 파일 이름과 개수가 원본과 같다.
- shard별 `run.log`에 오류가 없고 merge 결과의 text/metadata count가 일치한다.

## 검증 명령

```bash
python verify_mscoco_vqa_assets.py \
  --mscoco-root /var/tmp/jnuadmin_vlm/VLM/dataset/MSCOCO

python validate_mscoco_vqattack_output.py \
  --mscoco-root /var/tmp/jnuadmin_vlm/VLM/dataset/MSCOCO \
  --output-dir /var/tmp/jnuadmin_vlm/VLM/outputs/mscoco_vqattack_4gpu/merged/original_format \
  --splits train,val
```
