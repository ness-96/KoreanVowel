from __future__ import annotations

import hmac
import json
import uuid
import urllib.error
import urllib.parse
import urllib.request
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import parselmouth
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Korean vowel formant analyzer", version="0.1.0")
FORMANT_CEILING = float(os.getenv("FORMANT_CEILING", "5500"))
REFERENCE_TOLERANCE_HZ = 500.0
DEFAULT_REFERENCE_AVERAGES = {
    "ㅣ": {"female": {"f1": 360.0, "f2": 2442.0}, "male": {"f1": 236.0, "f2": 2183.0}},
    "ㅔ": {"female": {"f1": 500.0, "f2": 2168.0}, "male": {"f1": 415.0, "f2": 1848.0}},
    "ㅐ": {"female": {"f1": 500.0, "f2": 2168.0}, "male": {"f1": 415.0, "f2": 1848.0}},
    "ㅏ": {"female": {"f1": 825.0, "f2": 1688.0}, "male": {"f1": 651.0, "f2": 1156.0}},
    "ㅓ": {"female": {"f1": 595.0, "f2": 1286.0}, "male": {"f1": 445.0, "f2": 845.0}},
    "ㅡ": {"female": {"f1": 410.0, "f2": 1706.0}, "male": {"f1": 317.0, "f2": 1218.0}},
    "ㅗ": {"female": {"f1": 450.0, "f2": 1103.0}, "male": {"f1": 320.0, "f2": 587.0}},
    "ㅜ": {"female": {"f1": 357.0, "f2": 1203.0}, "male": {"f1": 324.0, "f2": 595.0}}
}

origins = [item.strip() for item in os.getenv("CORS_ORIGINS", "*").split(",") if item.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["DELETE", "GET", "POST", "OPTIONS"],
    allow_headers=["*"]
)


SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
SHARED_DATA_PASSWORD = os.getenv("SHARED_DATA_PASSWORD", "")
SUPABASE_TIMEOUT_SECONDS = 20
SHARED_VOWELS = {"ㅣ", "ㅔ", "ㅐ", "ㅏ", "ㅓ", "ㅡ", "ㅗ", "ㅜ"}
SHARED_RECORD_FIELDS = ("id", "speakerId", "region", "country", "nationality", "gender", "age", "vowel", "f1", "f2", "tokenNo", "source", "batchId", "createdAt")


def normalize_shared_speaker_records(records: Any) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        return []
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in records:
        if not isinstance(raw, dict):
            continue
        record_id = str(raw.get("id", "")).strip()
        speaker_id = str(raw.get("speakerId", raw.get("speaker_id", ""))).strip()
        region = str(raw.get("region", "")).strip()
        gender = str(raw.get("gender", "")).strip()
        age = str(raw.get("age", "")).strip()
        vowel = str(raw.get("vowel", "")).strip()
        try:
            f1 = float(raw.get("f1"))
            f2 = float(raw.get("f2"))
            token_no = int(raw.get("tokenNo", raw.get("token_no", 0)))
        except (TypeError, ValueError):
            continue
        if not record_id or not speaker_id or not age or gender not in {"female", "male"} or vowel not in SHARED_VOWELS:
            continue
        if not np.isfinite(f1) or not np.isfinite(f2) or not (0 < f1 < 5000) or not (0 < f2 < 5000) or token_no < 1 or record_id in seen_ids:
            continue
        item = {
            "id": record_id,
            "speakerId": speaker_id,
            "region": region,
            "country": str(raw.get("country", "")).strip(),
            "nationality": str(raw.get("nationality", "")).strip(),
            "gender": gender,
            "age": age,
            "vowel": vowel,
            "f1": f1,
            "f2": f2,
            "tokenNo": token_no,
            "source": "manual",
        }
        for optional_key in ("batchId", "createdAt"):
            if raw.get(optional_key):
                item[optional_key] = raw[optional_key]
        normalized.append(item)
        seen_ids.add(record_id)
    return normalized


def empty_shared_data() -> dict[str, list[dict[str, Any]]]:
    return {"speakerData": [], "averageData": []}


def supabase_request(method: str, endpoint: str, body: Any | None = None, query: dict[str, str] | None = None) -> Any:
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise HTTPException(status_code=503, detail="Supabase 연결 정보가 서버에 설정되지 않았습니다.")
    url = SUPABASE_URL + endpoint
    if query:
        url += "?" + urllib.parse.urlencode(query)
    request = urllib.request.Request(url, method=method)
    request.add_header("Accept", "application/json")
    request.add_header("Content-Type", "application/json")
    request.add_header("apikey", SUPABASE_SERVICE_KEY)
    request.add_header("Authorization", f"Bearer {SUPABASE_SERVICE_KEY}")
    if body is not None:
        request.data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    try:
        with urllib.request.urlopen(request, timeout=SUPABASE_TIMEOUT_SECONDS) as response:
            response_body = response.read().decode("utf-8")
            return json.loads(response_body) if response_body else {}
    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        if error.code == 409:
            raise HTTPException(status_code=409, detail="공유 데이터가 동시에 변경되었습니다. 잠시 후 다시 시도해 주세요.") from error
        if 400 <= error.code < 500:
            raise HTTPException(status_code=422, detail="공유 데이터 요청을 Supabase가 거부했습니다.") from error
        raise HTTPException(status_code=503, detail="Supabase 공유 데이터 저장소에 연결하지 못했습니다.") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise HTTPException(status_code=503, detail="Supabase 공유 데이터 저장소에 연결하지 못했습니다.") from error


def shared_row_to_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row.get("id", "")),
        "speakerId": str(row.get("speaker_id", "")),
        "region": str(row.get("region", "")),
        "country": str(row.get("country") or ""),
        "nationality": str(row.get("nationality") or ""),
        "gender": str(row.get("gender", "")),
        "age": str(row.get("age", "")),
        "vowel": str(row.get("vowel", "")),
        "f1": row.get("f1"),
        "f2": row.get("f2"),
        "tokenNo": row.get("token_no"),
        "source": str(row.get("source") or "manual"),
        "batchId": row.get("batch_id"),
        "createdAt": row.get("created_at"),
    }


def read_shared_data() -> dict[str, list[dict[str, Any]]]:
    rows = supabase_request(
        "GET",
        "/rest/v1/vowel_tokens",
        query={
            "select": "id,speaker_id,region,country,nationality,gender,age,vowel,f1,f2,token_no,source,batch_id,created_at",
            "deleted_at": "is.null",
            "order": "created_at.asc,id.asc",
        },
    )
    records = [shared_row_to_record(row) for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    return {"speakerData": normalize_shared_speaker_records(records), "averageData": []}


def require_shared_write_password(candidate: str | None) -> None:
    if not SHARED_DATA_PASSWORD:
        raise HTTPException(status_code=503, detail="공유 데이터 쓰기 비밀번호가 서버에 설정되지 않았습니다.")
    if not candidate or not hmac.compare_digest(candidate, SHARED_DATA_PASSWORD):
        raise HTTPException(status_code=401, detail="공유 데이터 쓰기 권한이 없습니다.")


@app.get("/shared-data")
def get_shared_data() -> dict[str, Any]:
    return {"ok": True, **read_shared_data()}


@app.post("/shared-data")
def append_shared_data(payload: dict[str, Any], x_shared_data_password: str | None = Header(default=None)) -> dict[str, Any]:
    require_shared_write_password(x_shared_data_password)
    speaker_records = normalize_shared_speaker_records(payload.get("speakerData", []))
    if not speaker_records:
        raise HTTPException(status_code=422, detail="공유할 새 화자 토큰이 없습니다.")
    batch_id = str(payload.get("batchId") or f"batch-{uuid.uuid4().hex}").strip()
    client_request_id = str(payload.get("clientRequestId") or batch_id).strip()
    rpc_payload = {
        "p_batch_id": batch_id,
        "p_client_request_id": client_request_id,
        "p_records": [
            {
                "id": item["id"],
                "speaker_id": item["speakerId"],
                "region": item["region"],
                "country": item.get("country", ""),
                "nationality": item.get("nationality", ""),
                "gender": item["gender"],
                "age": item["age"],
                "vowel": item["vowel"],
                "f1": item["f1"],
                "f2": item["f2"],
                "token_no": item["tokenNo"],
                "source": "manual",
            }
            for item in speaker_records
        ],
    }
    result = supabase_request("POST", "/rest/v1/rpc/append_vowel_batch", rpc_payload)
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise HTTPException(status_code=503, detail="Supabase에 일괄 입력을 저장하지 못했습니다.")
    return {
        "ok": True,
        "batchId": result.get("batchId", batch_id),
        "insertedCount": int(result.get("insertedCount", len(speaker_records)) or 0),
        "speakerData": normalize_shared_speaker_records(result.get("speakerData", [])),
        "averageData": [],
    }


@app.delete("/shared-data")
def delete_shared_data(payload: dict[str, Any], x_shared_data_password: str | None = Header(default=None)) -> dict[str, Any]:
    require_shared_write_password(x_shared_data_password)
    speaker_ids = [str(item) for item in payload.get("speakerIds", []) if item] if isinstance(payload.get("speakerIds", []), list) else []
    result = supabase_request(
        "POST",
        "/rest/v1/rpc/delete_vowel_tokens",
        {"p_ids": speaker_ids, "p_all": payload.get("all") is True},
    )
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise HTTPException(status_code=503, detail="Supabase에서 공유 데이터를 삭제하지 못했습니다.")
    return {
        "ok": True,
        "deletedCount": int(result.get("deletedCount", 0) or 0),
        "speakerData": normalize_shared_speaker_records(result.get("speakerData", [])),
        "averageData": [],
    }

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


def has_voice_activity(wav_path: Path) -> bool:
    sound = parselmouth.Sound(str(wav_path))
    values = np.asarray(sound.values, dtype=float)
    if values.size == 0:
        return False
    samples = values[0] if values.ndim > 1 else values
    peak = float(np.max(np.abs(samples)))
    if peak < 0.01:
        return False
    frame_length = max(1, int(sound.sampling_frequency * 0.025))
    hop_length = max(1, int(sound.sampling_frequency * 0.010))
    pitch = sound.to_pitch(time_step=0.010, pitch_floor=70, pitch_ceiling=500)
    active_frames = 0
    longest_active_run = 0
    current_run = 0
    for start in range(0, max(1, len(samples) - frame_length + 1), hop_length):
        frame = samples[start:start + frame_length]
        if len(frame) < frame_length // 2:
            break
        rms = float(np.sqrt(np.mean(np.square(frame))))
        center_time = (start + len(frame) / 2) / sound.sampling_frequency
        f0 = float(pitch.get_value_at_time(center_time))
        voiced = np.isfinite(f0) and 70 <= f0 <= 500
        if rms > 0.0035 and voiced:
            active_frames += 1
            current_run += 1
            longest_active_run = max(longest_active_run, current_run)
        else:
            current_run = 0
    return active_frames >= 8 and longest_active_run >= 6


def finite_formant(formant: parselmouth.Formant, number: int, time: float) -> float | None:
    value = float(formant.get_value_at_time(number, time))
    return value if np.isfinite(value) else None


def find_longest_voiced_run(
    sound: parselmouth.Sound,
    word_start: float,
    word_end: float
) -> list[dict[str, float]]:
    pitch = sound.to_pitch(time_step=0.005, pitch_floor=60, pitch_ceiling=500)
    values = np.asarray(sound.values, dtype=float)
    samples = values[0] if values.ndim > 1 else values
    frame_length = max(1, int(sound.sampling_frequency * 0.025))
    start = max(0.0, float(word_start) - 0.12)
    end = min(sound.duration, float(word_end) + 0.12)
    times = np.arange(start, end, 0.005)
    voiced: list[dict[str, float]] = []
    for time in times:
        center_index = int(float(time) * sound.sampling_frequency)
        frame_start = max(0, center_index - frame_length // 2)
        frame = samples[frame_start:frame_start + frame_length]
        if len(frame) < frame_length // 2:
            continue
        rms = float(np.sqrt(np.mean(np.square(frame))))
        f0 = float(pitch.get_value_at_time(float(time)))
        if rms > 0.0035 and np.isfinite(f0) and 70 <= f0 <= 500:
            voiced.append({"time": float(time)})

    runs: list[list[dict[str, float]]] = []
    current: list[dict[str, float]] = []
    for item in voiced:
        if current and item["time"] - current[-1]["time"] > 0.015:
            runs.append(current)
            current = []
        current.append(item)
    if current:
        runs.append(current)
    qualified = [
        run for run in runs
        if len(run) >= 8 and run[-1]["time"] - run[0]["time"] >= 0.08
    ]
    if not qualified:
        raise ValueError("고립 모음 녹음에서 충분히 긴 유성 구간을 찾지 못했습니다.")
    return max(qualified, key=lambda run: (run[-1]["time"] - run[0]["time"], len(run)))

def median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=float)))


def analyse_with_ceiling(
    sound: parselmouth.Sound,
    voiced_run: list[dict[str, float]],
    maximum_formant: float
) -> dict[str, Any]:
    formant = sound.to_formant_burg(
        time_step=0.005,
        max_number_of_formants=5,
        maximum_formant=maximum_formant,
        window_length=0.025,
        pre_emphasis_from=50
    )
    vowel_start = voiced_run[0]["time"]
    vowel_end = voiced_run[-1]["time"] + 0.005
    duration = vowel_end - vowel_start
    measurement_start = vowel_start + duration * 0.40
    measurement_end = vowel_start + duration * 0.60
    selected: list[dict[str, float]] = []
    for item in voiced_run:
        time = item["time"]
        if not measurement_start <= time <= measurement_end:
            continue
        f1 = finite_formant(formant, 1, time)
        f2 = finite_formant(formant, 2, time)
        if f1 is None or f2 is None:
            continue
        if not (200 <= f1 <= 1200 and 500 <= f2 <= 3500):
            continue
        selected.append({"time": time, "f1": f1, "f2": f2})
    if len(selected) < 3:
        raise ValueError("모음 안정 구간의 중앙 20%에서 유효한 F1·F2 프레임이 부족합니다.")
    stable_runs: list[list[dict[str, float]]] = []
    current_stable: list[dict[str, float]] = []
    for item in selected:
        if current_stable:
            previous = current_stable[-1]
            gap = item["time"] - previous["time"]
            jump_f1 = abs(item["f1"] - previous["f1"])
            jump_f2 = abs(item["f2"] - previous["f2"])
            if gap > 0.015 or jump_f1 > 90 or jump_f2 > 180:
                stable_runs.append(current_stable)
                current_stable = []
        current_stable.append(item)
    if current_stable:
        stable_runs.append(current_stable)
    qualified_stable_runs = [
        run for run in stable_runs
        if len(run) >= 8 and run[-1]["time"] - run[0]["time"] >= 0.03
    ]
    if not qualified_stable_runs:
        raise ValueError("공통 음성 구간에서 연속적으로 안정된 F1·F2 추적을 찾지 못했습니다. 모음을 더 길고 일정하게 발음한 뒤 다시 녹음해 주세요.")
    selected = max(qualified_stable_runs, key=lambda run: (run[-1]["time"] - run[0]["time"], len(run)))

    f1_values = [item["f1"] for item in selected]
    f2_values = [item["f2"] for item in selected]
    f1_iqr = float(np.percentile(f1_values, 75) - np.percentile(f1_values, 25))
    f2_iqr = float(np.percentile(f2_values, 75) - np.percentile(f2_values, 25))
    return {
        "f1": round(median(f1_values), 1),
        "f2": round(median(f2_values), 1),
        "method": "Praat Formant Burg via Parselmouth (common voiced interval, central 20%)",
        "formantCeiling": maximum_formant,
        "vowelInterval": {"start": round(vowel_start, 4), "end": round(vowel_end, 4)},
        "measurementInterval": {"start": round(measurement_start, 4), "end": round(measurement_end, 4)},
        "frameCount": len(selected),
        "quality": {
            "f1Iqr": round(f1_iqr, 1),
            "f2Iqr": round(f2_iqr, 1),
            "f1Range": [round(min(f1_values), 1), round(max(f1_values), 1)],
            "f2Range": [round(min(f2_values), 1), round(max(f2_values), 1)],
            "frames": [
                {"time": round(item["time"], 4), "f1": round(item["f1"], 1), "f2": round(item["f2"], 1)}
                for item in selected
            ],
            "runDuration": round(duration, 4),
            "segmentation": "common longest voiced interval; formant candidates compared on the same central 20%"
        }
    }

def analyse_formants(
    wav_path: Path,
    target_vowel: str,
    reference_averages: list[dict[str, Any]] | None = None,
    word_start: float = 0.0,
    word_end: float | None = None
) -> dict[str, Any]:
    sound = parselmouth.Sound(str(wav_path))
    if word_end is None:
        word_end = sound.duration
    voiced_run = find_longest_voiced_run(sound, word_start, word_end)
    candidates: list[dict[str, Any]] = []
    for ceiling in (5000.0, 5500.0, FORMANT_CEILING):
        if ceiling in {item.get("formantCeiling") for item in candidates}:
            continue
        try:
            candidates.append(analyse_with_ceiling(sound, voiced_run, ceiling))
        except ValueError:
            continue
    if not candidates:
        raise ValueError("모든 Formant ceiling에서 연속적으로 안정된 F1·F2 추적을 찾지 못했습니다. 모음을 더 길고 일정하게 발음해 다시 녹음해 주세요.")

    f1_spread = max(item["f1"] for item in candidates) - min(item["f1"] for item in candidates)
    f2_spread = max(item["f2"] for item in candidates) - min(item["f2"] for item in candidates)
    configured_refs = reference_averages or []
    if not configured_refs:
        configured_refs = [
            {"gender": gender, "vowel": target_vowel, **values}
            for gender, values in DEFAULT_REFERENCE_AVERAGES.get(target_vowel, {}).items()
        ]
    vowel_refs = [
        ref for ref in configured_refs
        if ref.get("vowel") == target_vowel
        and ref.get("gender") in {"female", "male"}
        and np.isfinite(float(ref.get("f1", float("nan"))))
        and np.isfinite(float(ref.get("f2", float("nan"))))
    ]
    reference_matches: dict[int, list[dict[str, Any]]] = {}
    for item in candidates:
        reference_matches[id(item)] = [
            ref for ref in vowel_refs
            if abs(item["f1"] - float(ref["f1"])) <= REFERENCE_TOLERANCE_HZ
            and abs(item["f2"] - float(ref["f2"])) <= REFERENCE_TOLERANCE_HZ
        ]
    reference_candidates = [item for item in candidates if reference_matches[id(item)]]
    if not reference_candidates:
        raise ValueError("측정값이 남성·여성 평균값에서 F1·F2 각각 ±500 Hz 범위를 벗어났습니다. 모음을 더 길고 일정하게 발음한 뒤 다시 녹음해 주세요.")
    selected = min(
        reference_candidates,
        key=lambda item: item["quality"]["f1Iqr"] + item["quality"]["f2Iqr"] * 0.25
    )
    matched_reference = min(
        reference_matches[id(selected)],
        key=lambda ref: abs(selected["f1"] - float(ref["f1"])) + abs(selected["f2"] - float(ref["f2"]))
    )
    warnings: list[str] = []
    if f1_spread > 90 or f2_spread > 180:
        warnings.append("Formant ceiling에 따라 후보값이 달라져 자동값을 수동 Praat 확인과 함께 사용해야 합니다.")
    if selected["quality"]["f1Iqr"] > 90 or selected["quality"]["f2Iqr"] > 180:
        warnings.append("안정 구간 안에서도 포먼트 변동이 커 수동 Praat 확인이 필요합니다.")
    selected["quality"]["ceilingSpread"] = {
        "f1": round(f1_spread, 1),
        "f2": round(f2_spread, 1),
        "warning": " ".join(warnings) or None
    }
    selected["quality"]["referenceRange"] = {
        "toleranceHz": REFERENCE_TOLERANCE_HZ,
        "matchedGender": matched_reference["gender"],
        "referenceF1": float(matched_reference["f1"]),
        "referenceF2": float(matched_reference["f2"])
    }
    selected["quality"]["ceilingCandidates"] = [
        {
            "formantCeiling": item["formantCeiling"],
            "f1": item["f1"],
            "f2": item["f2"],
            "f1Iqr": item["quality"]["f1Iqr"],
            "f2Iqr": item["quality"]["f2Iqr"],
            "f1Range": item["quality"]["f1Range"],
            "f2Range": item["quality"]["f2Range"],
            "vowelInterval": item["vowelInterval"],
            "measurementInterval": item["measurementInterval"],
            "frameCount": item["frameCount"],
            "selected": item is selected
        }
        for item in candidates
    ]
    return selected


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "korean-vowel-formant-analyzer"}


@app.post("/analyze")
async def analyze(
    audio: UploadFile = File(...),
    target_vowel: str = Form(...),
    reference_averages: str | None = Form(None)
) -> dict[str, Any]:
    suffix = Path(audio.filename or "recording.webm").suffix or ".webm"
    try:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / f"input{suffix}"
            wav_path = root / "audio.wav"
            input_path.write_bytes(await audio.read())
            convert_to_wav(input_path, wav_path)
            if not has_voice_activity(wav_path):
                raise HTTPException(status_code=422, detail={
                    "message": "음성 활동이 감지되지 않아 Praat/Parselmouth 분석을 실행하지 않았습니다.",
                    "reason": "no_voice_activity"
                })
            parsed_references: list[dict[str, Any]] | None = None
            if reference_averages:
                try:
                    candidate_references = json.loads(reference_averages)
                    if isinstance(candidate_references, list):
                        parsed_references = candidate_references
                except (TypeError, ValueError, json.JSONDecodeError):
                    parsed_references = None
            result = analyse_formants(wav_path, target_vowel, parsed_references)
            return {
                "ok": True,
                "targetVowel": target_vowel,
                **result
            }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=422, detail={"message": str(error)}) from error


