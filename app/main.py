"""
Whisper Large V3 Transcription API.
Accepts any FFmpeg-supported input file, extracts 16kHz mono audio, transcribes with Whisper.
Returns cleaned output (text, language, segments). Optional output filter and gen_cache.
"""
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import whisper
from fastapi import FastAPI, File, Query, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse

app = FastAPI(
    title="Whisper Large V3 Transcription API",
    description="Upload any FFmpeg-supported audio/video file; get Whisper large-v3 transcript. Use ?output=text,segment,gen_cache to filter response. API docs: /docs",
)

MODEL_NAME = os.environ.get("WHISPER_MODEL", "large-v3")
CACHE_DIR = os.environ.get("CACHE_DIR", "/app/cache")
ALLOWED_OUTPUT_KEYS = {"full", "text", "segment", "segments", "language", "gen_cache"}

model = None


def get_model():
    global model
    if model is None:
        model = whisper.load_model(
            MODEL_NAME,
            device="cuda",
            download_root="/app/models",
        )
    return model


@app.on_event("startup")
async def load_model_on_startup():
    """Load Whisper model at startup (downloads if needed)."""
    get_model()
    os.makedirs(CACHE_DIR, exist_ok=True)


def _clean_response(result: dict) -> dict:
    """Return only text, language, and segments with id, start, end, text (no tokens/logprobs)."""
    segments = [
        {
            "id": s["id"],
            "start": s["start"],
            "end": s["end"],
            "text": s["text"].strip(),
        }
        for s in result.get("segments", [])
    ]
    return {
        "text": (result.get("text") or "").strip(),
        "language": result.get("language"),
        "segments": segments,
    }


def _parse_output_keys(output: str) -> list[str]:
    """Parse comma-separated output param; return list of normalized keys. Raises HTTPException if invalid."""
    if not output or not output.strip():
        return ["full"]
    tokens = [k.strip().lower() for k in output.split(",") if k.strip()]
    invalid = [t for t in tokens if t not in ALLOWED_OUTPUT_KEYS]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid output key(s): {invalid}. Allowed: full, text, segment, segments, language, gen_cache",
        )
    return tokens


def _filter_response(full: dict, keys: list[str], gen_cache_obj: dict | None) -> dict:
    """Build response dict with only requested keys. keys already validated."""
    if "full" in keys or not keys:
        out = dict(full)
        if gen_cache_obj is not None:
            out["gen_cache"] = gen_cache_obj
        return out
    out = {}
    if "text" in keys:
        out["text"] = full["text"]
    if "language" in keys:
        out["language"] = full["language"]
    if "segment" in keys or "segments" in keys:
        out["segments"] = full["segments"]
    if gen_cache_obj is not None:
        out["gen_cache"] = gen_cache_obj
    return out


def extract_audio_ffmpeg(input_path: str, output_path: str) -> None:
    """Convert any FFmpeg-supported input to 16kHz mono WAV."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise HTTPException(
            status_code=400,
            detail=f"FFmpeg failed: {result.stderr or result.stdout or 'Unknown error'}",
        )


@app.get("/")
async def root():
    """Redirect to Swagger UI."""
    return RedirectResponse(url="/docs")


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    output: str = Query(
        "full",
        description="Comma-separated: full | text | segment | language | gen_cache. Response includes only these keys.",
    ),
):
    """
    Upload any audio/video file supported by FFmpeg.
    Returns cleaned output. Use query param `output` to get only specific keys:
    - full (default): text, language, segments
    - text, segment, language: single or multiple keys
    - gen_cache: write a timestamp-named JSON cache file and add gen_cache metadata to response
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    keys = _parse_output_keys(output)
    suffix = Path(file.filename).suffix or ".bin"
    input_path = tempfile.mktemp(suffix=suffix)
    wav_path = tempfile.mktemp(suffix=".wav")

    try:
        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)

        extract_audio_ffmpeg(input_path, wav_path)

        whisper_model = get_model()
        result = whisper_model.transcribe(
            wav_path,
            word_timestamps=False,
            condition_on_previous_text=False,
        )

        full = _clean_response(result)
        gen_cache_obj = None

        if "gen_cache" in keys:
            os.makedirs(CACHE_DIR, exist_ok=True)
            temp_cache_file_id = f"{time.time():.10f}.json"
            date_and_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            file_name = file.filename or "unknown"
            cache_path = os.path.join(CACHE_DIR, temp_cache_file_id)
            cache_content = {
                **full,
                "date_and_time": date_and_time,
                "file_name": file_name,
            }
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_content, f, ensure_ascii=False, indent=2)
            gen_cache_obj = {
                "temp_cache_file_id": temp_cache_file_id,
                "date_and_time": date_and_time,
                "file_name": file_name,
            }

        return _filter_response(full, keys, gen_cache_obj)
    finally:
        for path in (input_path, wav_path):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


def _list_cache_files() -> list[dict]:
    """Scan CACHE_DIR for *.json and return list of {temp_cache_file_id, date_and_time, file_name}."""
    files = []
    if not os.path.isdir(CACHE_DIR):
        return files
    for name in sorted(os.listdir(CACHE_DIR), reverse=True):
        if not name.endswith(".json"):
            continue
        path = os.path.join(CACHE_DIR, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            files.append({
                "temp_cache_file_id": name,
                "date_and_time": data.get("date_and_time", ""),
                "file_name": data.get("file_name", ""),
            })
        except (json.JSONDecodeError, OSError):
            mtime = os.path.getmtime(path) if os.path.isfile(path) else 0
            files.append({
                "temp_cache_file_id": name,
                "date_and_time": datetime.utcfromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S") if mtime else "",
                "file_name": "",
            })
    return files


@app.get("/cache-files")
async def cache_files_json():
    """Return JSON list of available cache files (temp_cache_file_id, date_and_time, file_name)."""
    return {"files": _list_cache_files()}


@app.get("/cache-files/list", response_class=HTMLResponse)
async def cache_files_list():
    """Return HTML page listing cache files in a table."""
    files = _list_cache_files()
    rows = "".join(
        f"<tr><td>{f['temp_cache_file_id']}</td><td>{f['date_and_time']}</td><td>{f['file_name']}</td></tr>"
        for f in files
    )
    if not rows:
        rows = "<tr><td colspan=\"3\">No cache files</td></tr>"
    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Cache files</title></head>
<body>
<h1>Cache files</h1>
<p><a href="/docs">API docs (Swagger)</a></p>
<table border="1" cellpadding="6">
<thead><tr><th>temp_cache_file_id</th><th>date_and_time</th><th>file_name</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</body>
</html>"""
    return HTMLResponse(html)


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME}
