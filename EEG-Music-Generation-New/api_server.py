from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Tuple

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from emmaq_generation_adapter import preview_parameter_tuning, run_pretuned_generation
from emmaq_generation_service import DATA_DIR, BridgeError, build_legacy_preferences, now_iso, save_preferences_submission
from maseval_pipeline_adapter import (
    build_generation_result_from_manifest,
    load_job_manifest,
    resolve_maseval_backend_script,
    run_evaluation_for_job,
    write_job_manifest,
)


def error_payload(message: str, *, details: Any = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"message": message}
    if details is not None:
        payload["details"] = details
    return payload


def build_error(code: str, message: str) -> Dict[str, Any]:
    return {"code": code, "message": message}


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
    if isinstance(exc.detail, str):
        message = exc.detail
    elif isinstance(exc.detail, dict):
        message = str(exc.detail.get("message") or "Request failed.")
    else:
        message = "Request failed."
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


def poll_url_for_job(job_id: str) -> str:
    return f"/api/jobs/{job_id}"


def load_job_snapshot(job_id: str) -> Dict[str, Any]:
    try:
        return load_job_manifest(job_id)
    except BridgeError as exc:
        if str(exc).startswith("Job not found:"):
            raise HTTPException(status_code=404, detail=str(exc))
        raise HTTPException(status_code=400, detail=str(exc))
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "pending",
                "job_id": job_id,
                "message": "Job status is being updated. Please retry.",
            },
        )


def safe_job_snapshot(job_id: str, fallback: Dict[str, Any] | None = None) -> Dict[str, Any]:
    try:
        return load_job_manifest(job_id)
    except Exception:
        return dict(fallback or {})


def derive_job_status(manifest: Dict[str, Any]) -> str:
    generation = manifest.get("generation") if isinstance(manifest.get("generation"), dict) else {}
    evaluation = manifest.get("evaluation") if isinstance(manifest.get("evaluation"), dict) else {}
    generation_status = str(generation.get("status") or "")
    evaluation_status = str(evaluation.get("status") or "")

    if generation_status in {"queued", "running", "error"}:
        return generation_status
    if evaluation_status in {"queued", "running"}:
        return "evaluating"
    if evaluation_status == "error":
        return "partial_failure"
    if generation_status == "completed":
        return "completed"
    return manifest.get("status") or "pending"


def update_phase(
    manifest: Dict[str, Any],
    phase: str,
    *,
    status: str,
    message: str,
    success: bool | None = None,
    error: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    phase_payload = manifest.get(phase) if isinstance(manifest.get(phase), dict) else {}
    phase_payload["status"] = status
    phase_payload["message"] = message
    if success is not None:
        phase_payload["success"] = success
    if error is None:
        phase_payload.pop("error", None)
    else:
        phase_payload["error"] = error
    manifest[phase] = phase_payload
    manifest["updated_at"] = now_iso()
    manifest["status"] = derive_job_status(manifest)
    return manifest


def build_job_stub(
    *,
    job_id: str,
    eeg_file_name: str,
    prepared_preview: Any,
    auto_evaluate: bool,
) -> Dict[str, Any]:
    created_at = now_iso()
    manifest: Dict[str, Any] = {
        "job_id": job_id,
        "created_at": created_at,
        "updated_at": created_at,
        "status": "queued",
        "eeg_file_name": eeg_file_name,
        "prepared": prepared_preview,
        "artifacts": {},
        "stages": [],
        "generation": {
            "status": "queued",
            "message": "Generation queued.",
        },
        "evaluation": {
            "status": "queued" if auto_evaluate else "idle",
            "message": "Evaluation queued after generation." if auto_evaluate else "Evaluation not started.",
        },
    }
    return manifest


def start_background_worker(target: Any, *args: Any, **kwargs: Any) -> None:
    worker = threading.Thread(target=target, args=args, kwargs=kwargs, daemon=True)
    worker.start()


def write_failure_manifest(job_id: str, manifest: Dict[str, Any], *, phase: str, message: str, code: str) -> None:
    update_phase(manifest, phase, status="error", message=message, success=False, error=build_error(code, message))
    write_job_manifest(job_id, manifest)


def run_evaluation_background(job_id: str) -> None:
    manifest = safe_job_snapshot(job_id, {"job_id": job_id})
    update_phase(manifest, "evaluation", status="running", message="Evaluation in progress.")
    write_job_manifest(job_id, manifest)
    try:
        run_evaluation_for_job(job_id)
    except Exception as exc:
        manifest = safe_job_snapshot(job_id, manifest)
        manifest["status"] = "partial_failure"
        write_failure_manifest(
            job_id,
            manifest,
            phase="evaluation",
            message=f"Evaluation failed. Please retry. {type(exc).__name__}: {exc}",
            code="evaluation_failed",
        )


def run_generation_background(
    *,
    job_id: str,
    payload: Dict[str, Any],
    preferences_ini: str,
    eeg_file_name: str,
    eeg_bytes: bytes,
    auto_evaluate: bool,
) -> None:
    manifest = safe_job_snapshot(job_id, {"job_id": job_id})
    update_phase(manifest, "generation", status="running", message="Generation in progress.")
    if auto_evaluate:
        update_phase(manifest, "evaluation", status="queued", message="Evaluation will start after generation.")
    write_job_manifest(job_id, manifest)

    try:
        run_pretuned_generation(
            payload,
            preferences_ini=preferences_ini,
            eeg_file_name=eeg_file_name,
            eeg_bytes=eeg_bytes,
            job_id=job_id,
        )
    except Exception as exc:
        manifest = safe_job_snapshot(job_id, manifest)
        write_failure_manifest(
            job_id,
            manifest,
            phase="generation",
            message=f"Generation failed. Please retry. {type(exc).__name__}: {exc}",
            code="generation_failed",
        )
        if auto_evaluate:
            manifest = safe_job_snapshot(job_id, manifest)
            update_phase(
                manifest,
                "evaluation",
                status="skipped",
                message="Evaluation did not start because generation failed.",
                success=False,
                error=build_error("generation_failed", "Evaluation did not start because generation failed."),
            )
            write_job_manifest(job_id, manifest)
        return

    if auto_evaluate:
        run_evaluation_background(job_id)


def create_generation_job(
    *,
    payload: Dict[str, Any],
    preferences_ini: str,
    eeg_file_name: str,
    eeg_bytes: bytes,
    auto_evaluate: bool = False,
) -> Dict[str, Any]:
    preview = preview_parameter_tuning(
        payload,
        preferences_ini=preferences_ini,
        eeg_file_name=eeg_file_name,
    )
    job_id = str(preview["job_id"])
    manifest = build_job_stub(
        job_id=job_id,
        eeg_file_name=eeg_file_name,
        prepared_preview=preview.get("used_parameters"),
        auto_evaluate=auto_evaluate,
    )
    write_job_manifest(job_id, manifest)
    start_background_worker(
        run_generation_background,
        job_id=job_id,
        payload=payload,
        preferences_ini=preferences_ini,
        eeg_file_name=eeg_file_name,
        eeg_bytes=eeg_bytes,
        auto_evaluate=auto_evaluate,
    )
    return {
        "success": True,
        "status": "accepted",
        "job_id": job_id,
        "message": "Generation started." if not auto_evaluate else "Generation and evaluation started.",
        "poll_url": poll_url_for_job(job_id),
        "generation": manifest["generation"],
        "evaluation": manifest["evaluation"],
        "output": preview.get("output"),
        "used_parameters": preview.get("used_parameters"),
    }


def enqueue_evaluation(job_id: str) -> Dict[str, Any]:
    manifest = load_job_snapshot(job_id)
    generation = manifest.get("generation") if isinstance(manifest.get("generation"), dict) else {}
    evaluation = manifest.get("evaluation") if isinstance(manifest.get("evaluation"), dict) else {}
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}

    if str(generation.get("status") or "") != "completed" or not isinstance(artifacts.get("final-midi"), dict):
        raise HTTPException(
            status_code=409,
            detail={
                "status": "pending",
                "job_id": job_id,
                "message": "Generation is still in progress.",
                "generation": generation,
                "evaluation": evaluation,
            },
        )

    evaluation_status = str(evaluation.get("status") or "")
    if evaluation_status in {"queued", "running"}:
        return {
            "success": True,
            "status": "accepted",
            "job_id": job_id,
            "message": evaluation.get("message") or "Evaluation is already in progress.",
            "poll_url": poll_url_for_job(job_id),
            "evaluation": evaluation,
        }
    if evaluation_status == "completed" and bool(evaluation.get("success")):
        return {
            "success": True,
            "status": "completed",
            "job_id": job_id,
            "message": evaluation.get("message") or "Evaluation already completed.",
            "poll_url": poll_url_for_job(job_id),
            "evaluation": evaluation,
            "generation_result": build_generation_result_from_manifest(manifest),
        }

    update_phase(manifest, "evaluation", status="queued", message="Evaluation queued.")
    write_job_manifest(job_id, manifest)
    start_background_worker(run_evaluation_background, job_id)
    return {
        "success": True,
        "status": "accepted",
        "job_id": job_id,
        "message": "Evaluation started.",
        "poll_url": poll_url_for_job(job_id),
        "evaluation": manifest["evaluation"],
        "generation_result": build_generation_result_from_manifest(manifest),
    }


def resolve_job_artifact(job_id: str, artifact_key: str) -> Tuple[Path, Dict[str, Any]]:
    manifest = load_job_snapshot(job_id)
    artifacts = manifest.get("artifacts") or {}
    artifact = artifacts.get(artifact_key)
    generation = manifest.get("generation") if isinstance(manifest.get("generation"), dict) else {}
    evaluation = manifest.get("evaluation") if isinstance(manifest.get("evaluation"), dict) else {}

    if not isinstance(artifact, dict):
        pending_generation_keys = {"final-midi", "stage-01-rule", "stage-02-fx", "stage-03-scale", "stage-04-quantize", "stage-05-postprocess", "stage-06-chords"}
        pending_evaluation_keys = {"evaluation-report", "evaluation-log"}
        if artifact_key in pending_generation_keys and str(generation.get("status") or "") in {"queued", "running"}:
            raise HTTPException(
                status_code=409,
                detail={
                    "status": "pending",
                    "job_id": job_id,
                    "message": "Generation is still in progress.",
                    "generation": generation,
                },
            )
        if artifact_key in pending_evaluation_keys and str(evaluation.get("status") or "") in {"queued", "running"}:
            raise HTTPException(
                status_code=409,
                detail={
                    "status": "pending",
                    "job_id": job_id,
                    "message": "Evaluation is still in progress.",
                    "evaluation": evaluation,
                },
            )
        raise HTTPException(status_code=404, detail=f"Artifact '{artifact_key}' not found for job {job_id}")

    relative_path = artifact.get("relative_path")
    if not isinstance(relative_path, str) or not relative_path:
        raise HTTPException(status_code=500, detail=f"Artifact '{artifact_key}' for job {job_id} has no path")
    resolved = (DATA_DIR / relative_path).resolve()
    if not resolved.exists():
        if artifact_key == "final-midi" and str(generation.get("status") or "") in {"queued", "running"}:
            raise HTTPException(
                status_code=409,
                detail={
                    "status": "pending",
                    "job_id": job_id,
                    "message": "Generation is still in progress.",
                    "generation": generation,
                },
            )
        if artifact_key in {"evaluation-report", "evaluation-log"} and str(evaluation.get("status") or "") in {"queued", "running"}:
            raise HTTPException(
                status_code=409,
                detail={
                    "status": "pending",
                    "job_id": job_id,
                    "message": "Evaluation is still in progress.",
                    "evaluation": evaluation,
                },
            )
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
    return create_generation_job(
        payload=payload,
        preferences_ini=preferences_ini,
        eeg_file_name=eeg_file.filename or "upload.csv",
        eeg_bytes=file_bytes,
        auto_evaluate=False,
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
    return create_generation_job(
        payload=payload,
        preferences_ini=preferences_ini,
        eeg_file_name=eeg_file.filename or "upload.csv",
        eeg_bytes=file_bytes,
        auto_evaluate=True,
    )


@app.post("/api/evaluate")
async def evaluate_job(request: Request) -> Dict[str, Any]:
    payload = await parse_json_request(request)
    job_id = str(payload.get("job_id") or "").strip()
    if not job_id:
        raise BridgeError("job_id is required.")
    return enqueue_evaluation(job_id)


@app.post("/api/jobs/{job_id}/evaluate")
def evaluate_existing_job(job_id: str) -> Dict[str, Any]:
    return enqueue_evaluation(job_id)


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
    return load_job_snapshot(job_id)


@app.get("/api/jobs/{job_id}/artifacts/{artifact_key}")
def download_artifact(job_id: str, artifact_key: str) -> FileResponse:
    path, artifact = resolve_job_artifact(job_id, artifact_key)
    return FileResponse(
        str(path),
        media_type=str(artifact.get("media_type") or "application/octet-stream"),
        filename=str(artifact.get("file_name") or path.name),
    )
