# 한국어 모음 F1·F2 분석 API

선택한 한국어 고립 모음의 음성 활동을 확인한 뒤, Praat/Parselmouth Formant Burg로 F1·F2를 측정하는 FastAPI 서버입니다. 전사나 음성인식은 사용하지 않습니다.

## 실행

1. Python 3.11 이상과 ffmpeg를 설치합니다.
2. 가상환경을 만들고 의존성을 설치합니다.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

## 프론트엔드 연결

사이트 주소에 다음처럼 API 주소를 지정할 수 있습니다.

```
?analysisApi=http://localhost:8000
```

실제 배포에서는 HTTPS API와 CORS 설정이 필요합니다.

## 분석 절차

1. 사용자가 선택한 모음만 고립 발화합니다.
2. 음성 활동과 지속적인 유성 구간을 확인합니다.
3. ffmpeg로 녹음 파일을 WAV로 변환합니다.
4. 고립 모음 녹음에서 가장 긴 유성·포먼트 안정 구간을 찾습니다.
5. 안정 구간의 중앙 20%를 측정합니다.
6. Formant Burg의 5,000/5,500Hz 결과가 크게 다르면 불안정값으로 보고하지 않습니다.

현재 자동 구간 검출은 예비 구현입니다. 연구용 사용 전에는 동일 발화를 Praat GUI에서 수동 측정해 구간과 값을 비교해야 합니다.
