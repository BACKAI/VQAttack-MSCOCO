# TextVQA VQAttack Server Package

TextVQA train/validation split에 대해 VQAttack을 서버 GPU에서 실행하기 위한 패키지입니다.  
로컬에서 데이터셋이나 가중치를 서버로 옮기지 않고, 서버 내부에서 GitHub clone 후 다운로드와 공격 실행을 진행합니다.

## 1. GitHub Repository 만들기

GitHub 웹사이트에서 새 repository를 만듭니다.

권장 설정:

```text
Repository name: VQAttack-TextVQA
Visibility: Private 또는 Public
Add a README file: 체크하지 않음
Add .gitignore: None
Choose a license: None
```

중요: README, `.gitignore`, license를 GitHub에서 자동 생성하지 마세요.  
이 폴더 안에 이미 필요한 파일이 들어 있습니다.

## 2. Git Bash에서 업로드

Windows에서 Git Bash를 열고 아래를 실행합니다.

```bash
cd /d/VLM/Attack/VQAttack_textvqa_github_package

git init
git add .
git commit -m "Add TextVQA VQAttack server package"
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_ID>/VQAttack-TextVQA.git
git push -u origin main
```

`<YOUR_GITHUB_ID>`는 본인 GitHub ID로 바꿔야 합니다.

이미 remote를 잘못 추가했다면:

```bash
git remote remove origin
git remote add origin https://github.com/<YOUR_GITHUB_ID>/VQAttack-TextVQA.git
git push -u origin main
```

업로드 후 GitHub repository 화면에 이 README가 보이면 코드 업로드는 완료된 것입니다.

## 3. 가중치 파일 올리기

가중치 파일은 repository에 commit하지 않습니다.

필요한 파일:

```text
ALBEF.pth
vqa.pth
```

권장 방식은 GitHub Releases입니다.

GitHub repository 화면에서:

```text
Releases -> Create a new release
Tag version: weights
Release title: weights
Attach binaries: ALBEF.pth, vqa.pth 업로드
Publish release
```

업로드 후 각 파일의 다운로드 URL은 보통 이런 형태입니다.

```text
https://github.com/<YOUR_GITHUB_ID>/VQAttack-TextVQA/releases/download/weights/ALBEF.pth
https://github.com/<YOUR_GITHUB_ID>/VQAttack-TextVQA/releases/download/weights/vqa.pth
```

브라우저에서 아래 링크를 눌렀을 때 다운로드가 시작되면 정상입니다.

- [ALBEF.pth 다운로드](https://github.com/<YOUR_GITHUB_ID>/VQAttack-TextVQA/releases/download/weights/ALBEF.pth)
- [vqa.pth 다운로드](https://github.com/<YOUR_GITHUB_ID>/VQAttack-TextVQA/releases/download/weights/vqa.pth)

위 링크의 `<YOUR_GITHUB_ID>`는 본인 GitHub ID로 바꾼 뒤 사용하세요.

## 4. 서버에서 clone

서버 터미널에서 실행합니다.

```bash
mkdir -p ~/VLM/Attack
cd ~/VLM/Attack

git clone https://github.com/<YOUR_GITHUB_ID>/VQAttack-TextVQA.git VQAttack
cd VQAttack
```

Private repository라면 GitHub token 또는 SSH key 설정이 필요할 수 있습니다.

## 5. 서버 Python 환경 생성

서버에서 실행합니다.

```bash
conda create -n vqattack-textvqa python=3.10 -y
conda activate vqattack-textvqa

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2" pillow nltk scikit-learn
pip install transformers==4.26.1 ruamel.yaml timm==0.4.12 opencv-python pycocotools pyarrow
```

서버의 NVIDIA driver가 충분히 최신이면 CUDA 12.1 wheel은 보통 동작합니다.  
서버가 A100이고 driver가 최신이면 이 방식이 가장 단순합니다.

## 6. 서버에서 가중치와 TextVQA 다운로드

아래 명령은 서버 내부에서 직접 다운로드합니다.

```bash
ALBEF_URL="https://github.com/<YOUR_GITHUB_ID>/VQAttack-TextVQA/releases/download/weights/ALBEF.pth" \
VQA_URL="https://github.com/<YOUR_GITHUB_ID>/VQAttack-TextVQA/releases/download/weights/vqa.pth" \
bash scripts/setup_server_assets.sh \
  --textvqa-root ~/VLM/dataset/textvqa \
  --checkpoint-dir ~/VLM/Attack/VQAttack/checkpoints/albef \
  --force-export
```

이 명령이 수행하는 작업:

```text
1. ALBEF.pth, vqa.pth 다운로드
2. Hugging Face lmms-lab/textvqa parquet 다운로드
3. TextVQA original_format 생성
```

생성되는 데이터 구조:

```text
~/VLM/dataset/textvqa/original_format/
  train_images/
  validation_images/
  test_images/
  TextVQA_0.5.1_train.json
  TextVQA_0.5.1_val.json
  TextVQA_0.5.1_test.json
```

## 7. 4GPU로 train/validation 전체 공격 실행

서버에서 실행합니다.

```bash
conda activate vqattack-textvqa
cd ~/VLM/Attack/VQAttack

CUDA_VISIBLE_DEVICES=0,1,2,3 bash textvqa_tools/run_textvqa_vqattack_4gpu.sh \
  --textvqa-root ~/VLM/dataset/textvqa \
  --checkpoint-root ~/VLM/Attack/VQAttack/checkpoints/albef \
  --work-root ~/VLM/outputs/textvqa_vqattack_4gpu \
  --gpu-ids 0,1,2,3 \
  --convert-images
```

최종 결과:

```text
~/VLM/outputs/textvqa_vqattack_4gpu/merged/original_format/
  train_images/
  validation_images/
  TextVQA_0.5.1_train.json
  TextVQA_0.5.1_val.json
  attack_original_format_manifest.json
```

최종 결과의 특징:

```text
원본과 같은 train_images / validation_images 구조 유지
원본과 같은 이미지 파일명 유지
원본과 같은 JSON question 개수 유지
question은 공격된 question으로 변경
original_question 필드에 원본 question 보존
test split은 공격하지 않음
```

## 8. tmux로 오래 실행하기

긴 작업은 tmux 안에서 실행하는 것을 권장합니다.

```bash
tmux new -s textvqa_attack
conda activate vqattack-textvqa
cd ~/VLM/Attack/VQAttack

CUDA_VISIBLE_DEVICES=0,1,2,3 bash textvqa_tools/run_textvqa_vqattack_4gpu.sh \
  --textvqa-root ~/VLM/dataset/textvqa \
  --checkpoint-root ~/VLM/Attack/VQAttack/checkpoints/albef \
  --work-root ~/VLM/outputs/textvqa_vqattack_4gpu \
  --gpu-ids 0,1,2,3 \
  --convert-images
```

tmux에서 빠져나오기:

```text
Ctrl+b 누른 뒤 d
```

다시 들어가기:

```bash
tmux attach -t textvqa_attack
```

## 9. 이 repository에 포함하지 않는 파일

아래 파일은 GitHub에 올리지 않습니다.

```text
*.pth
*.pt
*.ckpt
*.safetensors
*.parquet
*.jpg
*.png
checkpoints/
data/
datasets/
outputs/
logs/
__pycache__/
```

즉, GitHub에는 코드만 올리고, 서버가 직접 가중치와 데이터셋을 다운로드합니다.
