from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import parselmouth
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

app = FastAPI(title="Korean vowel formant analyzer", version="0.1.0")

origins = [item.strip() for item in os.getenv("CORS_ORIGINS", "*").split(",") if item.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["*"]
)

WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny")
FORMANT_CEILING = float(os.getenv("FORMANT_CEILING", "5500"))
_whisper_model: WhisperModel | None = None


def normalize_text(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", value or "").lower()


def get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(
            WHISPER_MODEL_NAME,
            device=os.getenv("WHISPER_DEVICE", "cpu"),
            compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        )
    return _whisper_model


def convert_to_wav(input_path: Path, output_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg가 설치되어 있지 않습니다.")
    completed = subprocess.run(
        [ffmpeg, "-y", "-i", str(input_path), "-ac", "1", "-ar", "16000", "-f", "wav", str(output_path)],
        capture_output=True,
        text=True
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr[-1000:] or "음성 파일 변환에 실패했습니다.")


def transcribe_target(wav_path: Path, target_word: str) -> dict[str, Any]:
    model = get_whisper_model()
    segments, _info = model.transcribe(
        str(wav_path),
        language="ko",
        word_timestamps=True,
        vad_filter=True,
        beam_size=1
    )
    segments = list(segments)
    transcript = " ".join(segment.text.strip() for segment in segments).strip()
    target_norm = normalize_text(target_word)
    words: list[dict[str, Any]] = []
    for segment in segments:
        for word in (segment.words or []):
            words.append({
                "text": word.word.strip(),
                "start": float(word.start),
                "end": float(word.end)
            })

    matched = next((word for word in words if target_norm and target_norm in normalize_text(word["text"])), None)
    if matched:
        return {"transcript": transcript, "targetMatched": True, "word": matched, "segments": segments}

    if target_norm and target_norm in normalize_text(transcript) and segments:
        return {
            "transcript": transcript,
            "targetMatched": True,
            "word": {"text": target_word, "start": float(segments[0].start), "end": float(segments[-1].end)},
            "segments": segments
        }
    return {"transcript": transcript, "targetMatched": False, "word": None, "segments": segments}


def finite_formant(formant: parselmouth.Formant, number: int, time: float) -> float | None:
    value = float(formant.get_value_at_time(number, time))
    return value if np.isfinite(value) else None


def find_first_vowel_run(sound: parselmouth.Sound, word_start: float, word_end: float) -> list[dict[str, float]]:
    pitch = sound.to_pitch(time_step=0.005, pitch_floor=60, pitch_ceiling=500)
    formant = sound.to_formant_burg(
        time_step=0.005,
        max_number_of_formants=5,
        maximum_formant=FORMANT_CEILING,
        window_length=0.025,
        pre_emphasis_from=50
    )
    times = np.arange(max(0.0, word_start), min(sound.duration, word_end), 0.005)
    valid: list[dict[str, float]] = []
    for time in times:
        f0 = float(pitch.get_value_at_time(time))
        f1 = finite_formant(formant, 1, float(time))
        f2 = finite_formant(formant, 2, float(time))
        if not np.isfinite(f0) or f0 <= 0 or f1 is None or f2 is None:
            continue
        if not (200 <= f1 <= 1200 and 500 <= f2 <= 3500):
            continue
        valid.append({"time": float(time), "f1": f1, "f2": f2})

    runs: list[list[dict[str, float]]] = []
    current: list[dict[str, float]] = []
    for item in valid:
        if current and item["time"] - current[-1]["time"] > 0.015:
            runs.append(current)
            current = []
        current.append(item)
    if current:
        runs.append(current)

    for run in runs:
        if len(run) >= 4 and run[-1]["time"] - run[0]["time"] >= 0.04:
            return run
    raise ValueError("목표모음으로 추정할 안정적인 유성 구간을 찾지 못했습니다.")


def median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=float)))


def analyse_formants(wav_path: Path, word: dict[str, Any]) -> dict[str, Any]:
    sound = parselmouth.Sound(str(wav_path))
    run = find_first_vowel_run(sound, float(word["start"]), float(word["end"]))
    vowel_start = run[0]["time"]
    vowel_end = run[-1]["time"] + 0.005
    duration = vowel_end - vowel_start
    measurement_start = vowel_start + duration * 0.40
    measurement_end = vowel_start + duration * 0.60
    selected = [item for item in run if measurement_start <= item["time"] <= measurement_end]
    if len(selected) < 2:
        raise ValueError("중앙 20% 구간의 유효한 분석 프레임이 부족합니다.")

    f1_values = [item["f1"] for item in selected]
    f2_values = [item["f2"] for item in selected]
    return {
        "f1": round(median(f1_values), 1),
        "f2": round(median(f2_values), 1),
        "method": "Praat Formant Burg via Parselmouth",
        "wordInterval": {"start": round(float(word["start"]), 4), "end": round(float(word["end"]), 4)},
        "vowelInterval": {"start": round(vowel_start, 4), "end": round(vowel_end, 4)},
        "measurementInterval": {"start": round(measurement_start, 4), "end": round(measurement_end, 4)},
        "frameCount": len(selected),
        "quality": {
            "f1Iqr": round(float(np.percentile(f1_values, 75) - np.percentile(f1_values, 25)), 1),
            "f2Iqr": round(float(np.percentile(f2_values, 75) - np.percentile(f2_values, 25)), 1),
            "segmentation": "first voiced formant-stable run within target word"
        }
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "korean-vowel-formant-analyzer"}


@app.get("/warmup")
def warmup() -> dict[str, str]:
    get_whisper_model()
    return {"status": "ready", "model": WHISPER_MODEL_NAME}


@app.post("/analyze")
async def analyze(
    audio: UploadFile = File(...),
    target_vowel: str = Form(...),
    target_word: str = Form(...)
) -> dict[str, Any]:
    suffix = Path(audio.filename or "recording.webm").suffix or ".webm"
    try:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / f"input{suffix}"
            wav_path = root / "audio.wav"
            input_path.write_bytes(await audio.read())
            convert_to_wav(input_path, wav_path)
            recognition = transcribe_target(wav_path, target_word)
            if not recognition["targetMatched"] or not recognition["word"]:
                raise HTTPException(status_code=422, detail={
                    "message": "Whisper 전사에서 목표항목을 확인하지 못했습니다.",
                    "transcript": recognition["transcript"],
                    "targetWord": target_word
                })
            result = analyse_formants(wav_path, recognition["word"])
            return {
                "ok": True,
                "targetVowel": target_vowel,
                "targetWord": target_word,
                "transcript": recognition["transcript"],
                "targetMatched": True,
                **result
            }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=422, detail={"message": str(error)}) from error




















































































































































































