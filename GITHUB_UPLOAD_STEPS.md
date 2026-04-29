# GitHub 업로드 절차

## 1. GitHub에서 repository 생성

GitHub 웹사이트에서 `New repository`를 누른다.

권장 설정:

```text
Repository name: VQAttack-TextVQA
Visibility: Private 또는 Public
Add a README file: 체크하지 않음
Add .gitignore: None
Choose a license: None
```

생성 후 repository URL을 확인한다.

예시:

```text
https://github.com/<YOUR_GITHUB_ID>/VQAttack-TextVQA.git
```

## 2. Git Bash에서 패키지 폴더로 이동

```bash
cd /d/VLM/Attack/VQAttack_textvqa_github_package
```

## 3. Git 초기화와 업로드

```bash
git init
git add .
git commit -m "Add TextVQA VQAttack server package"
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_ID>/VQAttack-TextVQA.git
git push -u origin main
```

## 4. GitHub Release에 가중치 업로드

GitHub repository 화면에서:

```text
Releases -> Create a new release
Tag version: weights
Release title: weights
Attach binaries: ALBEF.pth, vqa.pth
Publish release
```

다운로드 URL 예시:

```text
https://github.com/<YOUR_GITHUB_ID>/VQAttack-TextVQA/releases/download/weights/ALBEF.pth
https://github.com/<YOUR_GITHUB_ID>/VQAttack-TextVQA/releases/download/weights/vqa.pth
```

## 5. 서버에서 실행

서버에서:

```bash
mkdir -p ~/VLM/Attack
cd ~/VLM/Attack
git clone https://github.com/<YOUR_GITHUB_ID>/VQAttack-TextVQA.git VQAttack
cd VQAttack
```

이후 `README.md`의 서버 실행 절차를 따르면 된다.
