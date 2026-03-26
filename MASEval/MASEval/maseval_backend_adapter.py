from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = BASE_DIR / "music_eval_competition_light.py"
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"


class AdapterError(RuntimeError):
    def __init__(self, code: str, message: str, *, details: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_error(code: str, message: str, *, details: Any = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"code": code, "message": message}
    if details is not None:
        payload["details"] = details
    return payload


def append_log(logs: List[str], message: str) -> None:
    logs.append(f"[{now_iso()}] {message}")


def write_log(log_path: Path, logs: List[str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(logs).strip()
    if text:
        text += "\n"
    log_path.write_text(text, encoding="utf-8")


def build_template_module_name(template_path: Path) -> str:
    resolved = template_path.resolve()
    safe_stem = "".join(ch if ch.isalnum() else "_" for ch in resolved.stem).strip("_") or "template"
    path_hash = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:12]
    return f"maseval_template_{safe_stem}_{path_hash}"


def load_template_module(template_path: Path):
    resolved_path = template_path.resolve()
    module_name = build_template_module_name(resolved_path)
    spec = importlib.util.spec_from_file_location(module_name, str(resolved_path))
    if spec is None or spec.loader is None:
        raise AdapterError(
            "template_load_failed",
            "Evaluation module failed to load. Please retry or check backend logs.",
            details={"template_script": str(resolved_path), "module_name": module_name},
        )
    module = importlib.util.module_from_spec(spec)
    had_previous = module_name in sys.modules
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        if had_previous:
            sys.modules[module_name] = previous_module
        else:
            sys.modules.pop(module_name, None)
        raise AdapterError(
            "template_load_failed",
            "Evaluation module failed to load. Please retry or check backend logs.",
            details={
                "template_script": str(resolved_path),
                "module_name": module_name,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        ) from exc
    if not hasattr(module, "evaluate_one"):
        if had_previous:
            sys.modules[module_name] = previous_module
        else:
            sys.modules.pop(module_name, None)
        raise AdapterError(
            "template_missing_entry",
            "Evaluation module is unavailable. Please check backend deployment.",
            details={"template_script": str(resolved_path), "module_name": module_name},
        )
    return module


def ensure_environment(*, warnings: List[str], logs: List[str]) -> None:
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise AdapterError(
            "missing_api_key",
            "DEEPSEEK_API_KEY is not set. Configure it in the backend environment before running MASEval.",
        )
    if not os.getenv("DEEPSEEK_BASE_URL"):
        os.environ["DEEPSEEK_BASE_URL"] = DEFAULT_BASE_URL
        warnings.append(f"DEEPSEEK_BASE_URL was not set. Using default {DEFAULT_BASE_URL}.")
        append_log(logs, f"DEEPSEEK_BASE_URL not set; defaulting to {DEFAULT_BASE_URL}.")


def normalize_auto_edit(value: str) -> str:
    cleaned = "".join(ch for ch in (value or "QNV").upper() if ch in "QNVD")
    return cleaned or "QNV"


def build_evaluate_kwargs(
    evaluate_one: Any,
    *,
    out_path: Path,
    cache_dir: Path,
    model_judge: str,
    model_chair: str,
    max_measures: int,
    target_style: str,
    intended_use: str,
    rule_weight: float,
    auto_edit: str,
    do_after: bool,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "out_path": out_path,
        "cache_dir": cache_dir,
        "model_judge": model_judge,
        "model_chair": model_chair,
        "max_measures": max_measures,
        "target_style": str(target_style),
        "intended_use": str(intended_use),
        "rule_weight": float(rule_weight),
        "auto_edit": auto_edit,
        "do_after": bool(do_after),
    }
    supported = inspect.signature(evaluate_one).parameters
    return {key: value for key, value in kwargs.items() if key in supported}


def normalize_report(report: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(report)
    final_score_total = normalized.get("final_score_total")
    score_total = normalized.get("score_total")
    if final_score_total is None and score_total is not None:
        normalized["final_score_total"] = score_total
    if score_total is None and final_score_total is not None:
        normalized["score_total"] = final_score_total
    overall_comment = normalized.get("overall_comment")
    if not isinstance(overall_comment, str):
        normalized["overall_comment"] = ""
    return normalized


def run_maseval_backend(
    *,
    midi_path: Path,
    out_path: Path,
    cache_dir: Path,
    target_style: str,
    intended_use: str,
    log_file: Optional[Path] = None,
    template_script: Optional[Path] = None,
    model_judge: Optional[str] = None,
    model_chair: Optional[str] = None,
    max_measures: int = 8,
    rule_weight: float = 0.25,
    auto_edit: str = "QNV",
    do_after: bool = False,
) -> Dict[str, Any]:
    logs: List[str] = []
    warnings: List[str] = []
    midi_path = midi_path.resolve()
    out_path = out_path.resolve()
    cache_dir = cache_dir.resolve()
    template_path = (template_script or DEFAULT_TEMPLATE).resolve()
    log_path = (log_file or (out_path.parent / "evaluation_backend.log")).resolve()
    max_measures = max(1, int(max_measures))
    auto_edit = normalize_auto_edit(auto_edit)

    evaluation_result: Dict[str, Any] = {
        "midi_path": str(midi_path),
        "report_path": str(out_path),
        "cache_dir": str(cache_dir),
        "template_script": str(template_path),
        "target_style": str(target_style),
        "intended_use": str(intended_use),
        "log_file": str(log_path),
    }
    payload: Dict[str, Any] = {
        "success": False,
        "status": "error",
        "generation_result": None,
        "evaluation_result": evaluation_result,
        "error": None,
        "warnings": warnings,
    }

    append_log(logs, f"Starting MASEval backend adapter for MIDI: {midi_path}")
    append_log(logs, f"Report path: {out_path}")
    append_log(logs, f"Cache dir: {cache_dir}")

    try:
        if not midi_path.exists() or not midi_path.is_file():
            raise AdapterError("midi_not_found", f"MIDI file not found: {midi_path}")
        if not template_path.exists() or not template_path.is_file():
            raise AdapterError("template_not_found", f"Template script not found: {template_path}")

        ensure_environment(warnings=warnings, logs=logs)
        module = load_template_module(template_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        if model_judge is None:
            model_judge = getattr(module, "DEFAULT_MODEL", "deepseek-chat")
        if model_chair is None:
            model_chair = getattr(module, "DEFAULT_MODEL", "deepseek-chat")

        append_log(logs, f"Loaded template module from {template_path}")
        append_log(logs, f"Using model_judge={model_judge}, model_chair={model_chair}, target_style={target_style}, intended_use={intended_use}")

        evaluate_kwargs = build_evaluate_kwargs(
            module.evaluate_one,
            out_path=out_path,
            cache_dir=cache_dir,
            model_judge=model_judge,
            model_chair=model_chair,
            max_measures=max_measures,
            target_style=str(target_style),
            intended_use=str(intended_use),
            rule_weight=float(rule_weight),
            auto_edit=auto_edit,
            do_after=bool(do_after),
        )
        report = module.evaluate_one(midi_path, **evaluate_kwargs)
        if not isinstance(report, dict):
            raise AdapterError(
                "invalid_template_result",
                "Evaluation service did not return a valid result. Please retry.",
                details={
                    "template_script": str(template_path),
                    "result_type": type(report).__name__,
                },
            )
        report = normalize_report(report)

        evaluation_result.update(
            {
                "report": report,
                "report_exists": out_path.exists(),
                "final_score_total": report.get("final_score_total"),
                "score_total": report.get("score_total"),
                "summary": report.get("overall_comment") or "",
                "cache_key": report.get("cache_key"),
                "after_enabled": bool((report.get("after") or {}).get("enabled")) if isinstance(report, dict) else False,
                "message": "Evaluation completed.",
            }
        )
        append_log(logs, f"Evaluation completed. final_score_total={report.get('final_score_total')}")
        payload.update(
            {
                "success": True,
                "status": "completed",
                "evaluation_result": evaluation_result,
                "error": None,
            }
        )
    except AdapterError as exc:
        append_log(logs, f"Adapter error: {exc.code}: {exc}")
        if exc.details is not None:
            append_log(logs, f"Adapter error details: {json.dumps(exc.details, ensure_ascii=False)}")
        evaluation_result["message"] = str(exc)
        payload["error"] = make_error(exc.code, str(exc), details=exc.details)
    except SystemExit as exc:
        message = str(exc) or "MASEval template exited unexpectedly."
        append_log(logs, f"Template exited: {message}")
        user_message = "Evaluation failed. Please retry or check backend logs."
        evaluation_result["message"] = user_message
        payload["error"] = make_error("template_exit", user_message, details={"raw_message": message})
    except Exception as exc:
        append_log(logs, f"Unhandled error: {type(exc).__name__}: {exc}")
        append_log(logs, traceback.format_exc())
        user_message = "Evaluation failed. Please retry or check backend logs."
        evaluation_result["message"] = user_message
        payload["error"] = make_error(
            "unhandled_exception",
            user_message,
            details={"error_type": type(exc).__name__, "error_message": str(exc)},
        )
    finally:
        evaluation_result["log_tail"] = logs[-20:]
        write_log(log_path, logs)

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MASEval template through a stable JSON backend adapter.")
    parser.add_argument("--midi", required=True)
    parser.add_argument("--out", default="report_v2.json")
    parser.add_argument("--cache-dir", default=".cache_music_eval_v4_stages_v2")
    parser.add_argument("--target-style", default="general")
    parser.add_argument("--intended-use", default="demo")
    parser.add_argument("--log-file")
    parser.add_argument("--template-script")
    parser.add_argument("--model-judge")
    parser.add_argument("--model-chair")
    parser.add_argument("--max-measures", type=int, default=8)
    parser.add_argument("--rule-weight", type=float, default=0.25)
    parser.add_argument("--auto-edit", default="QNV")
    parser.add_argument("--after", action="store_true")
    args = parser.parse_args()

    result = run_maseval_backend(
        midi_path=Path(args.midi),
        out_path=Path(args.out),
        cache_dir=Path(args.cache_dir),
        target_style=args.target_style,
        intended_use=args.intended_use,
        log_file=Path(args.log_file) if args.log_file else None,
        template_script=Path(args.template_script) if args.template_script else None,
        model_judge=args.model_judge,
        model_chair=args.model_chair,
        max_measures=args.max_measures,
        rule_weight=args.rule_weight,
        auto_edit=args.auto_edit,
        do_after=bool(args.after),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
