# TextVQA 서버 실행 안내

이 repo는 서버에서 직접 다운로드하고 실행하는 흐름을 기준으로 한다. 로컬에서 데이터셋이나 가중치를 `scp`로 옮기지 않는다.

## GitHub에 포함되는 것

- VQAttack 코드 수정분
- TextVQA 준비/공격/merge/finalize 스크립트
- 서버 환경 파일
- 가중치 다운로드 스크립트
- TextVQA Hugging Face 다운로드 및 `original_format` export 스크립트

## GitHub에 포함하지 않는 것

- `ALBEF.pth`, `vqa.pth`
- TextVQA parquet/image/json 데이터
- `.pt` 공격 결과
- 공격 이미지 결과
- 로그와 output 폴더

## 서버에서 clone

```bash
mkdir -p ~/VLM/Attack
cd ~/VLM/Attack
git clone https://github.com/<user>/<repo>.git VQAttack
cd VQAttack
```

## 환경 생성

PyTorch는 서버 GPU/driver에 맞는 wheel을 먼저 설치하는 것을 권장한다.

```bash
conda create -n vqattack-textvqa python=3.10 -y
conda activate vqattack-textvqa

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2" pillow nltk scikit-learn
pip install transformers==4.26.1 ruamel.yaml timm==0.4.12 opencv-python pycocotools pyarrow
```

## 가중치 URL 준비

`ALBEF.pth`, `vqa.pth`는 GitHub repo에 commit하지 않는다. GitHub Releases, Hugging Face, 학교 서버 HTTP 링크 중 하나에 올리고 URL만 사용한다.

GitHub Release 예시:

```text
https://github.com/<user>/<repo>/releases/download/weights/ALBEF.pth
https://github.com/<user>/<repo>/releases/download/weights/vqa.pth
```

## 서버에서 가중치와 TextVQA 다운로드

아래 명령은 서버 내부에서 직접 실행한다.

```bash
ALBEF_URL="https://github.com/<user>/<repo>/releases/download/weights/ALBEF.pth" \
VQA_URL="https://github.com/<user>/<repo>/releases/download/weights/vqa.pth" \
bash scripts/setup_server_assets.sh \
  --textvqa-root ~/VLM/dataset/textvqa \
  --checkpoint-dir ~/VLM/Attack/VQAttack/checkpoints/albef \
  --force-export
```

위 명령이 하는 일:

- `scripts/download_weights.sh`로 `ALBEF.pth`, `vqa.pth` 다운로드
- `textvqa_tools/download_textvqa_hf.py`로 `lmms-lab/textvqa` parquet 다운로드
- `textvqa_tools/export_textvqa_original_format.py`로 아래 구조 생성

```text
~/VLM/dataset/textvqa/original_format/
  train_images/
  validation_images/
  test_images/
  TextVQA_0.5.1_train.json
  TextVQA_0.5.1_val.json
  TextVQA_0.5.1_test.json
```

가중치만 다시 받을 때:

```bash
ALBEF_URL="..." VQA_URL="..." bash scripts/download_weights.sh
```

데이터셋만 받을 때:

```bash
bash scripts/setup_server_assets.sh \
  --skip-weights \
  --textvqa-root ~/VLM/dataset/textvqa \
  --force-export
```

## 4GPU 전체 공격 실행

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash textvqa_tools/run_textvqa_vqattack_4gpu.sh \
  --textvqa-root ~/VLM/dataset/textvqa \
  --checkpoint-root ~/VLM/Attack/VQAttack/checkpoints/albef \
  --work-root ~/VLM/outputs/textvqa_vqattack_4gpu \
  --gpu-ids 0,1,2,3 \
  --convert-images
```

최종 출력:

```text
~/VLM/outputs/textvqa_vqattack_4gpu/merged/original_format/
  train_images/
  validation_images/
  TextVQA_0.5.1_train.json
  TextVQA_0.5.1_val.json
  attack_original_format_manifest.json
```

JSON의 모든 question은 공격된 question으로 바뀌며, `original_question` 필드에 원문을 보존한다. 이미지 파일명과 이미지 개수는 원본 `original_format`과 동일하게 유지한다.
