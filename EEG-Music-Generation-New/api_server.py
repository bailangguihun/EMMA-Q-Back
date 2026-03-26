from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from emmaq_generation_adapter import run_pretuned_generation
from emmaq_generation_service import DATA_DIR, BridgeError, build_legacy_preferences, save_preferences_submission
from maseval_pipeline_adapter import resolve_maseval_backend_script, run_evaluation_for_job, run_generation_evaluation_pipeline


def error_payload(message: str, *, details: Any = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"message": message}
    if details is not None:
        payload["details"] = details
    return payload


def parse_allowed_origins() -> list[str]:
    raw = str(os.getenv("EMMAQ_ALLOW_ORIGINS") or "*").strip()
    if not raw or raw == "*":
        return ["*"]
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    return origins or ["*"]


app = FastAPI(title="EMMA-Q EEG Backend", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_allowed_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(BridgeError)
async def handle_bridge_error(_: Request, exc: BridgeError) -> JSONResponse:
    return JSONResponse(status_code=400, content=error_payload(str(exc)))


@app.exception_handler(HTTPException)
async def handle_http_error(_: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, (dict, list)) else None
    message = exc.detail if isinstance(exc.detail, str) else "Request failed."
    return JSONResponse(status_code=exc.status_code, content=error_payload(message, details=detail))


@app.exception_handler(RequestValidationError)
async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content=error_payload("Request validation failed.", details=exc.errors()))


@app.exception_handler(Exception)
async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content=error_payload(f"Unexpected server error: {type(exc).__name__}: {exc}"))


async def parse_json_request(request: Request) -> Dict[str, Any]:
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise BridgeError(f"Request body is not valid JSON: {exc}")
    if not isinstance(payload, dict):
        raise BridgeError("JSON body must be an object.")
    return payload


async def read_upload(upload: UploadFile) -> bytes:
    file_name = upload.filename or "upload.csv"
    content = await upload.read()
    if not content:
        raise BridgeError(f"Uploaded file '{file_name}' is empty.")
    return content


def resolve_job_artifact(job_id: str, artifact_key: str) -> Tuple[Path, Dict[str, Any]]:
    manifest_path = (DATA_DIR / "jobs" / job_id / "manifest.json").resolve()
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts") or {}
    artifact = artifacts.get(artifact_key)
    if not isinstance(artifact, dict):
        raise HTTPException(status_code=404, detail=f"Artifact '{artifact_key}' not found for job {job_id}")
    relative_path = artifact.get("relative_path")
    if not isinstance(relative_path, str) or not relative_path:
        raise HTTPException(status_code=500, detail=f"Artifact '{artifact_key}' for job {job_id} has no path")
    resolved = (DATA_DIR / relative_path).resolve()
    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"Artifact '{artifact_key}' for job {job_id} is missing on disk")
    return resolved, artifact


@app.get("/api/health")
@app.get("/api/healthz")
@app.get("/api/status")
def health() -> Dict[str, Any]:
    adapter_script = resolve_maseval_backend_script()
    jobs_dir = DATA_DIR / "jobs"
    tmp_dir = DATA_DIR / "tmp"
    return {
        "status": "ok",
        "service": "emmaq-eeg-backend",
        "data_dir": str(DATA_DIR.resolve()),
        "cors": {
            "allowed_origins": parse_allowed_origins(),
        },
        "paths": {
            "data_dir_exists": DATA_DIR.exists(),
            "jobs_dir_exists": jobs_dir.exists(),
            "tmp_dir_exists": tmp_dir.exists(),
        },
        "evaluation_backend": {
            "adapter_found": adapter_script is not None,
            "adapter_script": str(adapter_script) if adapter_script else None,
            "deepseek_api_key_present": bool(os.getenv("DEEPSEEK_API_KEY")),
            "deepseek_base_url": os.getenv("DEEPSEEK_BASE_URL") or None,
        },
    }


@app.post("/api/preferences")
@app.post("/api/config/preferences")
@app.post("/api/prefs")
async def save_preferences(request: Request) -> Dict[str, Any]:
    payload = await parse_json_request(request)
    ini_content = payload.get("preferences_ini") if isinstance(payload.get("preferences_ini"), str) else ""
    return save_preferences_submission(payload, preferences_ini=ini_content)


@app.post("/api/generate")
@app.post("/api/jobs/generate")
@app.post("/api/music/generate")
async def generate_music(
    preferences: str = Form(...),
    preferences_ini: str = Form(""),
    eeg_file: UploadFile = File(...),
) -> Dict[str, Any]:
    try:
        payload = json.loads(preferences)
    except json.JSONDecodeError as exc:
        raise BridgeError(f"Form field 'preferences' is not valid JSON: {exc}")
    if not isinstance(payload, dict):
        raise BridgeError("Form field 'preferences' must decode to a JSON object.")
    file_bytes = await read_upload(eeg_file)
    return run_pretuned_generation(
        payload,
        preferences_ini=preferences_ini,
        eeg_file_name=eeg_file.filename or "upload.csv",
        eeg_bytes=file_bytes,
    )


@app.post("/api/pipeline/run")
async def run_pipeline(
    preferences: str = Form(...),
    preferences_ini: str = Form(""),
    eeg_file: UploadFile = File(...),
) -> Dict[str, Any]:
    try:
        payload = json.loads(preferences)
    except json.JSONDecodeError as exc:
        raise BridgeError(f"Form field 'preferences' is not valid JSON: {exc}")
    if not isinstance(payload, dict):
        raise BridgeError("Form field 'preferences' must decode to a JSON object.")
    file_bytes = await read_upload(eeg_file)
    return run_generation_evaluation_pipeline(
        payload,
        preferences_ini=preferences_ini,
        eeg_file_name=eeg_file.filename or "upload.csv",
        eeg_bytes=file_bytes,
    )


@app.post("/api/evaluate")
async def evaluate_job(request: Request) -> Dict[str, Any]:
    payload = await parse_json_request(request)
    job_id = str(payload.get("job_id") or "").strip()
    if not job_id:
        raise BridgeError("job_id is required.")
    return run_evaluation_for_job(job_id)


@app.post("/api/jobs/{job_id}/evaluate")
def evaluate_existing_job(job_id: str) -> Dict[str, Any]:
    return run_evaluation_for_job(job_id)


@app.post("/api/v1/eeg-to-midi")
async def legacy_eeg_to_midi(
    file: UploadFile = File(...),
    channel: int = Form(1),
    seconds: float = Form(60.0),
    rule: str = Form("p2p_r2v"),
) -> Dict[str, Any]:
    file_bytes = await read_upload(file)
    result = run_pretuned_generation(
        build_legacy_preferences(channel=channel, seconds=seconds, rule=rule),
        preferences_ini="",
        eeg_file_name=file.filename or "upload.csv",
        eeg_bytes=file_bytes,
    )
    return {
        "success": result["success"],
        "status": result["status"],
        "job_id": result["job_id"],
        "download_url": result.get("download_url"),
        "message": result["message"],
        "final_midi": result.get("output"),
        "output_path": result.get("output_path"),
        "used_parameters": result.get("used_parameters"),
    }


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    manifest_path = (DATA_DIR / "jobs" / job_id / "manifest.json").resolve()
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


@app.get("/api/jobs/{job_id}/artifacts/{artifact_key}")
def download_artifact(job_id: str, artifact_key: str) -> FileResponse:
    path, artifact = resolve_job_artifact(job_id, artifact_key)
    return FileResponse(
        str(path),
        media_type=str(artifact.get("media_type") or "application/octet-stream"),
        filename=str(artifact.get("file_name") or path.name),
    )

