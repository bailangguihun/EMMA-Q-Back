from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from emmaq_generation_service import DATA_DIR, BridgeError, artifact_entry, now_iso, run_generation_job, write_json_atomic

BASE_DIR = Path(__file__).resolve().parent


def build_error(code: str, message: str, *, details: Any = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"code": code, "message": message}
    if details is not None:
        payload["details"] = details
    return payload


def resolve_maseval_backend_script() -> Optional[Path]:
    configured = os.getenv("EMMAQ_MASEVAL_ADAPTER")
    candidates: List[Path] = []
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            BASE_DIR.parent / "MASEval" / "MASEval" / "maseval_backend_adapter.py",
            BASE_DIR.parent / "maseval" / "maseval_backend_adapter.py",
            BASE_DIR.parent / "maseval" / "MASEval" / "maseval_backend_adapter.py",
        ]
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return None


def resolve_maseval_python(script_path: Path) -> str:
    configured = os.getenv("EMMAQ_MASEVAL_PYTHON")
    if configured:
        return configured
    for candidate in (
        script_path.parent / ".venv" / "Scripts" / "python.exe",
        script_path.parent / ".venv" / "bin" / "python",
        script_path.parent.parent / ".venv" / "Scripts" / "python.exe",
        script_path.parent.parent / ".venv" / "bin" / "python",
    ):
        if candidate.exists():
            return str(candidate.resolve())
    return sys.executable


def parse_json_output(stdout: str) -> Dict[str, Any]:
    text = str(stdout or "").strip()
    if not text:
        raise BridgeError("Evaluation backend returned empty stdout.")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise BridgeError(f"Evaluation backend did not return valid JSON. Raw stdout: {text[-1000:]}")
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise BridgeError("Evaluation backend JSON must be an object.")
    return parsed


def normalize_job_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(manifest or {})
    artifacts = normalized.get("artifacts") if isinstance(normalized.get("artifacts"), dict) else {}
    evaluation = normalized.get("evaluation") if isinstance(normalized.get("evaluation"), dict) else {}
    generation = normalized.get("generation") if isinstance(normalized.get("generation"), dict) else {}

    has_final_midi = isinstance(artifacts.get("final-midi"), dict)
    if not generation:
        if has_final_midi:
            generation = {
                "success": True,
                "status": "completed",
                "message": "Generation completed.",
                "job_id": normalized.get("job_id"),
            }
        else:
            generation = {
                "success": False,
                "status": "pending",
                "message": "Generation status is pending.",
            }

    evaluation_status = str(evaluation.get("status") or "")
    if not evaluation:
        evaluation = {
            "status": "idle",
            "message": "Evaluation not started.",
        }
        evaluation_status = "idle"

    if not normalized.get("updated_at"):
        normalized["updated_at"] = normalized.get("created_at") or now_iso()

    if not normalized.get("status"):
        if str(generation.get("status") or "") in {"queued", "running", "error"}:
            normalized["status"] = str(generation.get("status"))
        elif evaluation_status in {"queued", "running"}:
            normalized["status"] = "evaluating"
        elif evaluation_status == "error":
            normalized["status"] = "partial_failure"
        elif str(generation.get("status") or "") == "completed":
            normalized["status"] = "completed"
        else:
            normalized["status"] = "pending"

    normalized["artifacts"] = artifacts
    normalized["generation"] = generation
    normalized["evaluation"] = evaluation
    normalized["stages"] = normalized.get("stages") if isinstance(normalized.get("stages"), list) else []
    return normalized


def load_job_manifest(job_id: str, *, data_dir: Optional[Path] = None) -> Dict[str, Any]:
    data_root = (data_dir or DATA_DIR).resolve()
    manifest_path = data_root / "jobs" / job_id / "manifest.json"
    if not manifest_path.exists():
        raise BridgeError(f"Job not found: {job_id}")
    return normalize_job_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))


def write_job_manifest(job_id: str, manifest: Dict[str, Any], *, data_dir: Optional[Path] = None) -> None:
    data_root = (data_dir or DATA_DIR).resolve()
    manifest_path = data_root / "jobs" / job_id / "manifest.json"
    write_json_atomic(manifest_path, normalize_job_manifest(manifest))


def resolve_artifact_path(artifact: Dict[str, Any], *, data_dir: Optional[Path] = None) -> Path:
    data_root = (data_dir or DATA_DIR).resolve()
    relative_path = artifact.get("relative_path")
    if not isinstance(relative_path, str) or not relative_path:
        raise BridgeError("Artifact is missing relative_path.")
    return (data_root / relative_path).resolve()


def build_generation_result(result: Dict[str, Any]) -> Dict[str, Any]:
    final_midi = result.get("final_midi") if isinstance(result.get("final_midi"), dict) else {}
    return {
        "success": True,
        "status": "completed",
        "job_id": result.get("job_id"),
        "message": result.get("message") or "Generation completed.",
        "final_midi": final_midi,
        "output_path": final_midi.get("relative_path") or final_midi.get("file_name"),
        "download_url": result.get("download_url"),
        "prepared": result.get("prepared"),
        "estimated_key": result.get("estimated_key"),
        "stages": result.get("stages") or [],
    }


def build_generation_result_from_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_job_manifest(manifest)
    artifacts = normalized.get("artifacts") if isinstance(normalized.get("artifacts"), dict) else {}
    final_midi = artifacts.get("final-midi") if isinstance(artifacts.get("final-midi"), dict) else {}
    generation = normalized.get("generation") if isinstance(normalized.get("generation"), dict) else {}
    return {
        "success": bool(generation.get("success", bool(final_midi))),
        "status": str(generation.get("status") or ("completed" if final_midi else "pending")),
        "job_id": normalized.get("job_id"),
        "message": generation.get("message") or "Generation artifact loaded from existing job.",
        "final_midi": final_midi,
        "output_path": final_midi.get("relative_path") or final_midi.get("file_name"),
        "download_url": final_midi.get("download_url"),
        "prepared": normalized.get("prepared"),
        "estimated_key": normalized.get("estimated_key"),
        "stages": normalized.get("stages") or [],
    }


def infer_target_style(prepared_preview: Any) -> str:
    if isinstance(prepared_preview, dict):
        chord = prepared_preview.get("chord")
        if isinstance(chord, dict):
            emotion = chord.get("emotion")
            if emotion:
                return str(emotion).lower()
    return "general"


def evaluate_generated_job(
    *,
    job_id: str,
    final_midi_artifact: Dict[str, Any],
    prepared_preview: Any,
    data_dir: Optional[Path] = None,
    intended_use: str = "interactive_generation",
) -> Dict[str, Any]:
    data_root = (data_dir or DATA_DIR).resolve()
    evaluation_dir = data_root / "jobs" / job_id / "evaluation"
    report_path = evaluation_dir / "evaluation_report.json"
    cache_dir = evaluation_dir / "cache"
    log_path = evaluation_dir / "evaluation_backend.log"
    adapter_path = resolve_maseval_backend_script()
    if adapter_path is None:
        return {
            "success": False,
            "status": "error",
            "job_id": job_id,
            "message": "MASEval backend adapter was not found.",
            "report": None,
            "final_score_total": None,
            "warnings": [],
            "error": build_error("adapter_not_found", "MASEval backend adapter script was not found."),
            "artifacts": {},
        }

    midi_path = resolve_artifact_path(final_midi_artifact, data_dir=data_root)
    command = [
        resolve_maseval_python(adapter_path),
        str(adapter_path),
        "--midi",
        str(midi_path),
        "--out",
        str(report_path.resolve()),
        "--cache-dir",
        str(cache_dir.resolve()),
        "--target-style",
        infer_target_style(prepared_preview),
        "--intended-use",
        intended_use,
        "--log-file",
        str(log_path.resolve()),
    ]

    try:
        completed = subprocess.run(
            command,
            cwd=str(adapter_path.parent),
            capture_output=True,
            text=True,
            timeout=int(os.getenv("EMMAQ_MASEVAL_TIMEOUT_SECONDS", "600") or "600"),
            check=False,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "success": False,
            "status": "error",
            "job_id": job_id,
            "message": f"Evaluation timed out after {exc.timeout} seconds.",
            "report": None,
            "final_score_total": None,
            "warnings": [],
            "error": build_error("evaluation_timeout", f"Evaluation timed out after {exc.timeout} seconds."),
            "artifacts": {},
        }
    except Exception as exc:
        return {
            "success": False,
            "status": "error",
            "job_id": job_id,
            "message": f"Evaluation failed to start: {type(exc).__name__}: {exc}",
            "report": None,
            "final_score_total": None,
            "warnings": [],
            "error": build_error("evaluation_start_failed", f"Evaluation failed to start: {type(exc).__name__}: {exc}"),
            "artifacts": {},
        }

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    try:
        payload = parse_json_output(stdout)
    except Exception as exc:
        return {
            "success": False,
            "status": "error",
            "job_id": job_id,
            "message": f"Evaluation backend returned invalid JSON: {type(exc).__name__}: {exc}",
            "report": None,
            "final_score_total": None,
            "warnings": [],
            "error": build_error("invalid_backend_json", f"Evaluation backend returned invalid JSON: {type(exc).__name__}: {exc}"),
            "artifacts": {},
            "stdout": stdout[-4000:],
            "stderr": stderr[-4000:],
        }

    warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
    backend_eval = payload.get("evaluation_result") if isinstance(payload.get("evaluation_result"), dict) else {}
    artifacts: Dict[str, Any] = {}
    if report_path.exists():
        artifacts["report"] = artifact_entry(key="evaluation-report", path=report_path, data_dir=data_root, job_id=job_id, media_type="application/json")
    if log_path.exists():
        artifacts["log"] = artifact_entry(key="evaluation-log", path=log_path, data_dir=data_root, job_id=job_id, media_type="text/plain")

    message = backend_eval.get("message") or (payload.get("error") or {}).get("message") or ("Evaluation completed." if payload.get("success") else "Evaluation failed.")
    result = {
        "success": bool(payload.get("success")),
        "status": str(payload.get("status") or ("completed" if payload.get("success") else "error")),
        "job_id": job_id,
        "message": message,
        "report": backend_eval.get("report"),
        "final_score_total": backend_eval.get("final_score_total"),
        "cache_key": backend_eval.get("cache_key"),
        "warnings": warnings,
        "error": payload.get("error"),
        "artifacts": artifacts,
        "stdout": stdout[-4000:],
        "stderr": stderr[-4000:],
        "backend": {
            "adapter_script": str(adapter_path),
            "python": resolve_maseval_python(adapter_path),
            "returncode": completed.returncode,
        },
    }
    return result


def update_manifest_with_evaluation(job_id: str, evaluation_result: Dict[str, Any], *, data_dir: Optional[Path] = None) -> Dict[str, Any]:
    manifest = load_job_manifest(job_id, data_dir=data_dir)
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    eval_artifacts = evaluation_result.get("artifacts") if isinstance(evaluation_result.get("artifacts"), dict) else {}
    if isinstance(eval_artifacts.get("report"), dict):
        artifacts["evaluation-report"] = eval_artifacts["report"]
    if isinstance(eval_artifacts.get("log"), dict):
        artifacts["evaluation-log"] = eval_artifacts["log"]
    manifest["artifacts"] = artifacts
    manifest["evaluation"] = evaluation_result
    manifest["updated_at"] = now_iso()
    if str(evaluation_result.get("status") or "") == "completed" and bool(evaluation_result.get("success")):
        manifest["status"] = "completed"
    elif str(evaluation_result.get("status") or "") in {"queued", "running"}:
        manifest["status"] = "evaluating"
    else:
        manifest["status"] = "partial_failure"
    write_job_manifest(job_id, manifest, data_dir=data_dir)
    return manifest


def build_pipeline_response(generation_result: Dict[str, Any], evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
    success = bool(generation_result.get("success")) and bool(evaluation_result.get("success"))
    if success:
        status = "completed"
        error = None
    elif generation_result.get("success"):
        status = "partial_failure"
        error = evaluation_result.get("error") or build_error("evaluation_failed", evaluation_result.get("message") or "Evaluation failed.")
    else:
        status = "error"
        error = generation_result.get("error") or build_error("generation_failed", generation_result.get("message") or "Generation failed.")
    warnings: List[str] = []
    if isinstance(evaluation_result.get("warnings"), list):
        warnings.extend(str(item) for item in evaluation_result.get("warnings") if item)
    return {
        "success": success,
        "status": status,
        "generation_result": generation_result,
        "evaluation_result": evaluation_result,
        "error": error,
        "warnings": warnings,
    }


def run_generation_evaluation_pipeline(
    preferences_payload: Optional[Dict[str, Any]],
    *,
    preferences_ini: str,
    eeg_file_name: str,
    eeg_bytes: bytes,
    data_dir: Optional[Path] = None,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    generation_raw = run_generation_job(
        preferences_payload,
        preferences_ini=preferences_ini,
        eeg_file_name=eeg_file_name,
        eeg_bytes=eeg_bytes,
        data_dir=data_dir,
        job_id=job_id,
        enable_evaluation=False,
    )
    generation_result = build_generation_result(generation_raw)
    manifest = load_job_manifest(str(generation_result["job_id"]), data_dir=data_dir)
    final_midi = generation_result.get("final_midi") if isinstance(generation_result.get("final_midi"), dict) else {}
    evaluation_result = evaluate_generated_job(
        job_id=str(generation_result["job_id"]),
        final_midi_artifact=final_midi,
        prepared_preview=manifest.get("prepared"),
        data_dir=data_dir,
    )
    update_manifest_with_evaluation(str(generation_result["job_id"]), evaluation_result, data_dir=data_dir)
    return build_pipeline_response(generation_result, evaluation_result)


def run_evaluation_for_job(job_id: str, *, data_dir: Optional[Path] = None) -> Dict[str, Any]:
    manifest = load_job_manifest(job_id, data_dir=data_dir)
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    final_midi = artifacts.get("final-midi") if isinstance(artifacts.get("final-midi"), dict) else None
    if not isinstance(final_midi, dict):
        raise BridgeError(f"Job {job_id} does not have a final MIDI artifact.")
    generation_result = build_generation_result_from_manifest(manifest)
    evaluation_result = evaluate_generated_job(
        job_id=job_id,
        final_midi_artifact=final_midi,
        prepared_preview=manifest.get("prepared"),
        data_dir=data_dir,
    )
    update_manifest_with_evaluation(job_id, evaluation_result, data_dir=data_dir)
    return build_pipeline_response(generation_result, evaluation_result)


