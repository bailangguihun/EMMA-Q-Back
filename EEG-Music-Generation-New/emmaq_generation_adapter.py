from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from emmaq_generation_service import (
    DATA_DIR,
    BridgeError,
    build_preview,
    normalize_preferences,
    prepare_generation,
    run_generation_job,
)


def build_error_result(message: str, *, error_type: Optional[str] = None, details: Any = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "success": False,
        "status": "error",
        "error": message,
    }
    if error_type:
        payload["error_type"] = error_type
    if details is not None:
        payload["details"] = details
    return payload


def build_output_plan(job_id: str, output_prefix: str, *, data_dir: Optional[Path] = None) -> Dict[str, Any]:
    relative_path = (Path("jobs") / job_id / "outputs" / f"{output_prefix}_final.mid").as_posix()
    plan = {
        "file_name": f"{output_prefix}_final.mid",
        "relative_path": relative_path,
    }
    if data_dir is not None:
        plan["absolute_path"] = str((data_dir.resolve() / relative_path).resolve())
    return plan


def preview_parameter_tuning(
    preferences_payload: Optional[Dict[str, Any]],
    *,
    preferences_ini: str = "",
    eeg_file_name: str = "draft.csv",
    job_id: Optional[str] = None,
    data_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    data_root = (data_dir or DATA_DIR).resolve()
    normalized = normalize_preferences(preferences_payload, ini_content=preferences_ini, eeg_file_name=eeg_file_name)
    prepared = prepare_generation(normalized, job_id=job_id)
    return {
        "success": True,
        "status": "validated",
        "message": "Parameters validated.",
        "job_id": prepared.job_id,
        "output": build_output_plan(prepared.job_id, prepared.output_prefix, data_dir=data_root),
        "used_parameters": build_preview(prepared),
        "normalized_preferences": normalized,
    }


def run_pretuned_generation(
    preferences_payload: Optional[Dict[str, Any]],
    *,
    preferences_ini: str,
    eeg_file_name: str,
    eeg_bytes: bytes,
    data_dir: Optional[Path] = None,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    result = run_generation_job(
        preferences_payload,
        preferences_ini=preferences_ini,
        eeg_file_name=eeg_file_name,
        eeg_bytes=eeg_bytes,
        data_dir=data_dir,
        job_id=job_id,
        enable_evaluation=False,
    )
    final_midi = result.get("final_midi") if isinstance(result.get("final_midi"), dict) else {}
    output_path = final_midi.get("relative_path") or final_midi.get("file_name")
    return {
        "success": True,
        "status": "completed",
        "message": str(result.get("message") or "Generation completed."),
        "job_id": result.get("job_id"),
        "output_path": output_path,
        "output": final_midi,
        "download_url": result.get("download_url"),
        "used_parameters": result.get("prepared"),
        "estimated_key": result.get("estimated_key"),
        "stages": result.get("stages") or [],
    }


def _load_json_input(inline_value: Optional[str], file_path: Optional[str]) -> Dict[str, Any]:
    if inline_value:
        payload = json.loads(inline_value)
    elif file_path:
        payload = json.loads(Path(file_path).read_text(encoding="utf-8"))
    else:
        payload = {}
    if not isinstance(payload, dict):
        raise BridgeError("Preferences JSON must decode to an object.")
    return payload


def _load_text_input(inline_value: Optional[str], file_path: Optional[str]) -> str:
    if inline_value is not None and inline_value != "":
        return inline_value
    if file_path:
        return Path(file_path).read_text(encoding="utf-8")
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate parameter tuning and run EEG-to-MIDI generation without Streamlit or browser launch."
    )
    parser.add_argument("--csv", help="Path to EEG CSV file.")
    parser.add_argument("--eeg-file-name", help="Optional EEG file name for preview mode.")
    parser.add_argument("--preferences-json", help="Inline JSON payload.")
    parser.add_argument("--preferences-json-file", help="Path to JSON payload file.")
    parser.add_argument("--preferences-ini", help="Inline INI content.")
    parser.add_argument("--preferences-ini-file", help="Path to INI content file.")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Directory for generated job outputs.")
    parser.add_argument("--job-id", help="Optional fixed job id.")
    parser.add_argument("--preview-only", action="store_true", help="Validate parameters and show the derived generation plan only.")
    args = parser.parse_args()

    try:
        preferences_payload = _load_json_input(args.preferences_json, args.preferences_json_file)
        preferences_ini = _load_text_input(args.preferences_ini, args.preferences_ini_file)
        data_dir = Path(args.data_dir)
        if args.preview_only:
            eeg_file_name = args.eeg_file_name or (Path(args.csv).name if args.csv else "draft.csv")
            result = preview_parameter_tuning(
                preferences_payload,
                preferences_ini=preferences_ini,
                eeg_file_name=eeg_file_name,
                job_id=args.job_id,
                data_dir=data_dir,
            )
        else:
            if not args.csv:
                raise BridgeError("--csv is required unless --preview-only is used.")
            csv_path = Path(args.csv)
            if not csv_path.exists():
                raise BridgeError(f"CSV file not found: {csv_path}")
            result = run_pretuned_generation(
                preferences_payload,
                preferences_ini=preferences_ini,
                eeg_file_name=csv_path.name,
                eeg_bytes=csv_path.read_bytes(),
                data_dir=data_dir,
                job_id=args.job_id,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        message = str(exc) or f"{type(exc).__name__}"
        print(json.dumps(build_error_result(message, error_type=type(exc).__name__), ensure_ascii=False, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
