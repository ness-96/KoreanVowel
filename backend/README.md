# 한국어 모음 F1·F2 분석 API

Whisper로 목표 단어를 확인한 뒤, 목표 단어 안의 첫 번째 유성·포먼트 안정 구간에서 중앙 20%를 선택하고 Parselmouth(Praat Formant Burg)로 F1·F2를 산출하는 무료 로컬 분석 서버입니다.

## 실행

1. Python 3.11 이상과 ffmpeg를 설치합니다.
2. 이 파일들과 `app.py`를 같은 폴더에 두고 실행합니다.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

첫 분석 때 faster-whisper 모델을 내려받습니다. 기본 모델은 `small`이며, 느린 컴퓨터에서는 `WHISPER_MODEL=base`로 바꿀 수 있습니다.

## 프론트엔드 연결

프론트엔드가 `analysisApi` 주소를 사용하도록 연결한 뒤 다음처럼 실행합니다.

```text
?analysisApi=http://localhost:8000
```

실제 배포에서는 분석 API를 HTTPS로 제공하고 `CORS_ORIGINS`에 GitHub Pages 주소를 지정해야 합니다.

## 분석 절차

1. Whisper가 목표 단어와 단어 구간을 확인합니다.
2. 목표 단어 안에서 첫 번째 유성·포먼트 안정 구간을 찾습니다.
3. 해당 모음 구간의 중앙 20%를 선택합니다.
4. Parselmouth의 Praat Formant Burg로 여러 프레임의 F1·F2를 측정합니다.
5. 중앙값과 측정 구간·IQR을 JSON으로 반환합니다.

현재 목표항목이 모음+다 형식이라는 점을 이용한 예비 자동 구간 추출입니다. 연구용으로 사용하기 전에는 동일 음성을 Praat GUI에서 수동 측정하여 오차를 검증해야 합니다.

























