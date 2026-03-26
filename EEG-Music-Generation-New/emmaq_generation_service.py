from __future__ import annotations

import argparse
import configparser
import inspect
import json
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from miditoolkit.midi import parser as mid_parser
from miditoolkit.midi.containers import Instrument, Note

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / ".api_data"
PREFERENCES_DIR = DATA_DIR / "preferences"
JOBS_DIR = DATA_DIR / "jobs"
TMP_DIR = DATA_DIR / "tmp"

for directory in (DATA_DIR, PREFERENCES_DIR, JOBS_DIR, TMP_DIR):
    directory.mkdir(parents=True, exist_ok=True)

DEFAULT_FLAT_PREFERENCES: Dict[str, Any] = {
    "startTime": 0,
    "duration": 30,
    "samplingRate": 64,
    "maxPoints": 8000,
    "melodyChannel": 0,
    "rule": "Rule1",
    "timeSignature": "4/4",
    "pitchCenter": 66,
    "span": 8,
    "spanMode": "Mode1",
    "pitch1": 74,
    "pitch2": 66,
    "pitch3": 58,
    "dynamicsPreset": "Standard",
    "vel1": 90,
    "vel2": 70,
    "vel3": 55,
    "magnet": 4,
    "loudnessOffset": 0,
    "dynamicsLimitStrategy": "Balanced",
    "rhythmQuantization": "Off",
    "swingAmount": 0.12,
    "noteDurationMultiplier": 1.2,
    "sparsity": "0",
    "scaleConstraint": "Diatonic",
    "pitchCoherence": "None",
    "pitchRangeMin": 48,
    "pitchRangeMax": 84,
    "maxLeap": 9,
    "transpose": 0,
    "melodyMinVelocity": 45,
    "melodyMaxVelocity": 90,
    "chordTrackEnabled": False,
    "keyRoot": "Auto",
    "majorMinor": "Auto",
    "emotion": "Calm",
    "chordStyle": "Block",
    "chordChangeFrequency": 8,
    "chordOctaveOffset": 0,
    "chordVelocity": 55,
    "apiBaseUrl": "http://127.0.0.1:8000/api",
}

SECTION_TO_FLAT: Dict[str, Dict[str, str]] = {
    "input": {
        "eeg_file": "eegFileName",
        "start_time": "startTime",
        "duration": "duration",
        "sampling_rate": "samplingRate",
        "max_points": "maxPoints",
        "melody_channel": "melodyChannel",
    },
    "rule": {
        "rule": "rule",
        "time_signature": "timeSignature",
        "pitch_center": "pitchCenter",
        "span": "span",
        "span_mode": "spanMode",
        "pitch_1": "pitch1",
        "pitch_2": "pitch2",
        "pitch_3": "pitch3",
        "magnet": "magnet",
    },
    "dynamics": {
        "preset": "dynamicsPreset",
        "vel_1": "vel1",
        "vel_2": "vel2",
        "vel_3": "vel3",
        "loudness_offset": "loudnessOffset",
        "limit_strategy": "dynamicsLimitStrategy",
    },
    "advanced": {
        "rhythm_quantization": "rhythmQuantization",
        "pitch_coherence": "pitchCoherence",
        "swing_amount": "swingAmount",
        "note_duration_multiplier": "noteDurationMultiplier",
        "sparsity": "sparsity",
        "scale_constraint": "scaleConstraint",
        "pitch_range_min": "pitchRangeMin",
        "pitch_range_max": "pitchRangeMax",
        "max_leap": "maxLeap",
        "transpose": "transpose",
        "melody_min_velocity": "melodyMinVelocity",
        "melody_max_velocity": "melodyMaxVelocity",
    },
    "chord": {
        "chord_track_enabled": "chordTrackEnabled",
        "key_root": "keyRoot",
        "major_minor": "majorMinor",
        "emotion": "emotion",
        "chord_style": "chordStyle",
        "chord_change_frequency": "chordChangeFrequency",
        "chord_octave_offset": "chordOctaveOffset",
        "chord_velocity": "chordVelocity",
    },
}

ALLOWED_TIME_SIGNATURES = {"2/4", "3/4", "4/4", "3/8", "6/8"}
ALLOWED_RULES = {"Rule1", "Rule2"}
ALLOWED_SPAN_MODES = {"Mode1", "Mode2"}
ALLOWED_DYNAMICS_PRESETS = {"Light", "Standard", "Strong", "Custom"}
ALLOWED_DYNAMICS_LIMITS = {"Conservative", "Balanced", "Aggressive"}
ALLOWED_QUANTIZATION = {"Off", "1/8", "1/16", "1/32"}
ALLOWED_SPARSITY = {"0", "1", "2", "3", "4"}
ALLOWED_SCALE_CONSTRAINTS = {"Diatonic", "Chromatic", "Penta"}
ALLOWED_PITCH_COHERENCE = {"None", "FX1", "FX2", "FX3", "FX4"}
ALLOWED_KEY_ROOTS = {"Auto", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"}
ALLOWED_MAJOR_MINOR = {"Auto", "Major", "Minor"}
ALLOWED_EMOTIONS = {"Calm", "Excited", "Sad", "Tense"}
ALLOWED_CHORD_STYLE = {"Block", "Arp"}

RULE_TO_GENERATOR = {"Rule1": "rule1", "Rule2": "rule2"}
PRESET_VELOCITIES = {
    "Light": (78, 62, 48),
    "Standard": (90, 70, 55),
    "Strong": (110, 90, 70),
}
LIMIT_CAPS = {"Conservative": 95, "Balanced": 110, "Aggressive": 127}
FX_MAP = {"None": "none", "FX1": "FX1", "FX2": "FX2", "FX3": "FX3", "FX4": "FX4"}
SCALE_MAP = {"Diatonic": "diatonic", "Chromatic": "chromatic", "Penta": "penta"}
QUANTIZATION_MAP = {"Off": "off", "1/8": "1/8", "1/16": "1/16", "1/32": "1/32"}
EMOTION_TO_CHORD_TAG = {"Calm": "Calm", "Excited": "Happy", "Sad": "Sad", "Tense": "Tense"}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NAME_TO_PC = {name: index for index, name in enumerate(NOTE_NAMES)}
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]
PENTA_MAJOR = [0, 2, 4, 7, 9]
PENTA_MINOR = [0, 3, 5, 7, 10]
EMOTION_DEGREES_MAJOR = {
    "Happy": ["I", "V", "vi", "IV"],
    "Sad": ["vi", "IV", "I", "V"],
    "Calm": ["I", "iii", "IV", "V"],
    "Tense": ["ii", "V", "I", "vi"],
}
EMOTION_DEGREES_MINOR = {
    "Happy": ["i", "VII", "VI", "VII"],
    "Sad": ["i", "VI", "VII", "i"],
    "Calm": ["i", "III", "VI", "VII"],
    "Tense": ["ii_deg", "V", "i", "VII"],
}


class BridgeError(ValueError):
    pass


@dataclass
class PreparedGeneration:
    job_id: str
    eeg_file_name: str
    start_time: int
    duration: int
    target_sampling_rate: int
    max_points: int
    melody_channel: int
    rule_key: str
    numerator: int
    denominator: int
    magnet: int
    pitches: Tuple[int, int, int]
    velocities: Tuple[int, int, int]
    fx_choice: str
    scale_mode: str
    quantization: str
    swing_amount: float
    note_duration_multiplier: float
    sparsity: int
    pitch_range_min: int
    pitch_range_max: int
    max_leap: int
    transpose: int
    melody_min_velocity: int
    melody_max_velocity: int
    chord_track_enabled: bool
    key_root: str
    major_minor: str
    emotion: str
    chord_style: str
    chord_change_frequency: int
    chord_octave_offset: int
    chord_velocity: int
    output_prefix: str


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sanitize_name(value: str, fallback: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value or "").strip("._-")
    return slug or fallback


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def parse_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_text(value: Any, default: str) -> str:
    text = str(value).strip() if value is not None else ""
    return text or default


def bool_from_env(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}

def flatten_section_preferences(sections: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for section_name, mapping in SECTION_TO_FLAT.items():
        section = sections.get(section_name)
        if not isinstance(section, dict):
            continue
        for source_key, target_key in mapping.items():
            if source_key in section:
                flat[target_key] = section[source_key]
    return flat


def parse_preferences_ini(ini_content: str) -> Dict[str, Any]:
    if not str(ini_content or "").strip():
        return {}
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    parser.read_string(ini_content)
    sections: Dict[str, Dict[str, Any]] = {}
    for section_name in parser.sections():
        sections[section_name] = dict(parser.items(section_name))
    return flatten_section_preferences(sections)


def extract_flat_preferences(payload: Optional[Dict[str, Any]], ini_content: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    if isinstance(payload, dict):
        payload_flat = payload.get("flat")
        if isinstance(payload_flat, dict):
            flat.update(payload_flat)
        sections = payload.get("preferences")
        if isinstance(sections, dict):
            for key, value in flatten_section_preferences(sections).items():
                flat.setdefault(key, value)
    for key, value in parse_preferences_ini(ini_content).items():
        flat.setdefault(key, value)
    return flat


def normalize_preferences(
    payload: Optional[Dict[str, Any]],
    *,
    ini_content: str = "",
    eeg_file_name: str = "",
) -> Dict[str, Any]:
    raw_flat = extract_flat_preferences(payload, ini_content=ini_content)
    normalized = dict(DEFAULT_FLAT_PREFERENCES)
    normalized["eegFileName"] = eeg_file_name or parse_text(raw_flat.get("eegFileName"), "")

    for key, default in DEFAULT_FLAT_PREFERENCES.items():
        raw_value = raw_flat.get(key, default)
        if isinstance(default, bool):
            normalized[key] = parse_bool(raw_value)
        elif isinstance(default, int):
            normalized[key] = parse_int(raw_value, default)
        elif isinstance(default, float):
            normalized[key] = parse_float(raw_value, default)
        else:
            normalized[key] = parse_text(raw_value, default)

    if eeg_file_name:
        normalized["eegFileName"] = eeg_file_name
    return normalized


def require_range(errors: List[str], label: str, value: float, minimum: float, maximum: float) -> None:
    if value < minimum or value > maximum:
        errors.append(f"{label} must be between {minimum} and {maximum}.")


def require_choice(errors: List[str], label: str, value: str, allowed: set[str]) -> None:
    if value not in allowed:
        errors.append(f"{label} must be one of: {', '.join(sorted(allowed))}.")


def validate_preferences(normalized: Dict[str, Any], *, require_file_name: bool = False) -> None:
    errors: List[str] = []
    require_range(errors, "startTime", normalized["startTime"], 0, 120)
    require_range(errors, "duration", normalized["duration"], 30, 120)
    require_range(errors, "samplingRate", normalized["samplingRate"], 16, 128)
    require_range(errors, "maxPoints", normalized["maxPoints"], 1000, 30000)
    require_range(errors, "melodyChannel", normalized["melodyChannel"], 0, 6)
    require_range(errors, "pitchCenter", normalized["pitchCenter"], 60, 72)
    require_range(errors, "span", normalized["span"], 4, 12)
    require_range(errors, "magnet", normalized["magnet"], 0, 12)
    require_range(errors, "loudnessOffset", normalized["loudnessOffset"], -30, 30)
    require_range(errors, "swingAmount", normalized["swingAmount"], 0, 0.6)
    require_range(errors, "noteDurationMultiplier", normalized["noteDurationMultiplier"], 0.25, 2.5)
    require_range(errors, "pitchRangeMin", normalized["pitchRangeMin"], 12, 84)
    require_range(errors, "pitchRangeMax", normalized["pitchRangeMax"], 48, 120)
    require_range(errors, "maxLeap", normalized["maxLeap"], 3, 24)
    require_range(errors, "transpose", normalized["transpose"], -10, 10)
    require_range(errors, "melodyMinVelocity", normalized["melodyMinVelocity"], 30, 60)
    require_range(errors, "melodyMaxVelocity", normalized["melodyMaxVelocity"], 75, 105)
    require_range(errors, "chordChangeFrequency", normalized["chordChangeFrequency"], 1, 32)
    require_range(errors, "chordOctaveOffset", normalized["chordOctaveOffset"], -2, 2)
    require_range(errors, "chordVelocity", normalized["chordVelocity"], 20, 90)
    require_choice(errors, "rule", normalized["rule"], ALLOWED_RULES)
    require_choice(errors, "timeSignature", normalized["timeSignature"], ALLOWED_TIME_SIGNATURES)
    require_choice(errors, "spanMode", normalized["spanMode"], ALLOWED_SPAN_MODES)
    require_choice(errors, "dynamicsPreset", normalized["dynamicsPreset"], ALLOWED_DYNAMICS_PRESETS)
    require_choice(errors, "dynamicsLimitStrategy", normalized["dynamicsLimitStrategy"], ALLOWED_DYNAMICS_LIMITS)
    require_choice(errors, "rhythmQuantization", normalized["rhythmQuantization"], ALLOWED_QUANTIZATION)
    require_choice(errors, "sparsity", normalized["sparsity"], ALLOWED_SPARSITY)
    require_choice(errors, "scaleConstraint", normalized["scaleConstraint"], ALLOWED_SCALE_CONSTRAINTS)
    require_choice(errors, "pitchCoherence", normalized["pitchCoherence"], ALLOWED_PITCH_COHERENCE)
    require_choice(errors, "keyRoot", normalized["keyRoot"], ALLOWED_KEY_ROOTS)
    require_choice(errors, "majorMinor", normalized["majorMinor"], ALLOWED_MAJOR_MINOR)
    require_choice(errors, "emotion", normalized["emotion"], ALLOWED_EMOTIONS)
    require_choice(errors, "chordStyle", normalized["chordStyle"], ALLOWED_CHORD_STYLE)

    if normalized["pitchRangeMin"] >= normalized["pitchRangeMax"]:
        errors.append("pitchRangeMin must be smaller than pitchRangeMax.")
    if normalized["melodyMinVelocity"] >= normalized["melodyMaxVelocity"]:
        errors.append("melodyMinVelocity must be smaller than melodyMaxVelocity.")
    if normalized["spanMode"] == "Mode2":
        require_range(errors, "pitch1", normalized["pitch1"], 0, 127)
        require_range(errors, "pitch2", normalized["pitch2"], 0, 127)
        require_range(errors, "pitch3", normalized["pitch3"], 0, 127)
    if normalized["dynamicsPreset"] == "Custom":
        require_range(errors, "vel1", normalized["vel1"], 1, 127)
        require_range(errors, "vel2", normalized["vel2"], 1, 127)
        require_range(errors, "vel3", normalized["vel3"], 1, 127)
    if require_file_name and not normalized.get("eegFileName"):
        errors.append("EEG file name is required.")
    if errors:
        raise BridgeError("Invalid preferences: " + " ".join(errors))


def parse_time_signature(value: str) -> Tuple[int, int]:
    if value not in ALLOWED_TIME_SIGNATURES:
        raise BridgeError(f"Unsupported time signature: {value}")
    numerator, denominator = value.split("/", 1)
    return int(numerator), int(denominator)


def derive_pitches(normalized: Dict[str, Any]) -> Tuple[int, int, int]:
    if normalized["spanMode"] == "Mode2":
        return (
            int(np.clip(normalized["pitch1"], 0, 127)),
            int(np.clip(normalized["pitch2"], 0, 127)),
            int(np.clip(normalized["pitch3"], 0, 127)),
        )
    center = int(normalized["pitchCenter"])
    span = int(normalized["span"])
    return (
        int(np.clip(center + span, 0, 127)),
        int(np.clip(center, 0, 127)),
        int(np.clip(center - span, 0, 127)),
    )


def derive_velocities(normalized: Dict[str, Any]) -> Tuple[int, int, int]:
    if normalized["dynamicsPreset"] == "Custom":
        base = (int(normalized["vel1"]), int(normalized["vel2"]), int(normalized["vel3"]))
    else:
        base = PRESET_VELOCITIES[normalized["dynamicsPreset"]]
    cap = LIMIT_CAPS[normalized["dynamicsLimitStrategy"]]
    offset = int(normalized["loudnessOffset"])
    adjusted = [int(np.clip(value + offset, 1, cap)) for value in base]
    return int(adjusted[0]), int(adjusted[1]), int(adjusted[2])


def prepare_generation(normalized: Dict[str, Any], *, job_id: Optional[str] = None) -> PreparedGeneration:
    validate_preferences(normalized, require_file_name=True)
    numerator, denominator = parse_time_signature(normalized["timeSignature"])
    eeg_file_name = parse_text(normalized.get("eegFileName"), "eeg.csv")
    base_name = sanitize_name(Path(eeg_file_name).stem, "eeg")
    derived_job_id = job_id or uuid.uuid4().hex
    return PreparedGeneration(
        job_id=derived_job_id,
        eeg_file_name=eeg_file_name,
        start_time=int(normalized["startTime"]),
        duration=int(normalized["duration"]),
        target_sampling_rate=int(normalized["samplingRate"]),
        max_points=int(normalized["maxPoints"]),
        melody_channel=int(normalized["melodyChannel"]),
        rule_key=RULE_TO_GENERATOR[normalized["rule"]],
        numerator=numerator,
        denominator=denominator,
        magnet=int(normalized["magnet"]),
        pitches=derive_pitches(normalized),
        velocities=derive_velocities(normalized),
        fx_choice=FX_MAP[normalized["pitchCoherence"]],
        scale_mode=SCALE_MAP[normalized["scaleConstraint"]],
        quantization=QUANTIZATION_MAP[normalized["rhythmQuantization"]],
        swing_amount=float(normalized["swingAmount"]),
        note_duration_multiplier=float(normalized["noteDurationMultiplier"]),
        sparsity=int(normalized["sparsity"]),
        pitch_range_min=int(normalized["pitchRangeMin"]),
        pitch_range_max=int(normalized["pitchRangeMax"]),
        max_leap=int(normalized["maxLeap"]),
        transpose=int(normalized["transpose"]),
        melody_min_velocity=int(normalized["melodyMinVelocity"]),
        melody_max_velocity=int(normalized["melodyMaxVelocity"]),
        chord_track_enabled=bool(normalized["chordTrackEnabled"]),
        key_root=str(normalized["keyRoot"]),
        major_minor=str(normalized["majorMinor"]),
        emotion=str(normalized["emotion"]),
        chord_style="block" if normalized["chordStyle"] == "Block" else "arp",
        chord_change_frequency=int(normalized["chordChangeFrequency"]),
        chord_octave_offset=int(normalized["chordOctaveOffset"]),
        chord_velocity=int(normalized["chordVelocity"]),
        output_prefix=f"{base_name}_job_{derived_job_id[:8]}",
    )


def build_preview(prepared: PreparedGeneration) -> Dict[str, Any]:
    return {
        "job_id": prepared.job_id,
        "eeg_file_name": prepared.eeg_file_name,
        "rule_key": prepared.rule_key,
        "time_signature": f"{prepared.numerator}/{prepared.denominator}",
        "melody_channel": prepared.melody_channel,
        "slice": {
            "start_time": prepared.start_time,
            "duration": prepared.duration,
            "target_sampling_rate": prepared.target_sampling_rate,
            "max_points": prepared.max_points,
        },
        "rule_stage": {
            "pitches": prepared.pitches,
            "velocities": prepared.velocities,
            "magnet": prepared.magnet,
        },
        "postprocess": {
            "fx_choice": prepared.fx_choice,
            "scale_mode": prepared.scale_mode,
            "quantization": prepared.quantization,
            "swing_amount": prepared.swing_amount,
            "note_duration_multiplier": prepared.note_duration_multiplier,
            "sparsity": prepared.sparsity,
            "pitch_range_min": prepared.pitch_range_min,
            "pitch_range_max": prepared.pitch_range_max,
            "max_leap": prepared.max_leap,
            "transpose": prepared.transpose,
            "melody_min_velocity": prepared.melody_min_velocity,
            "melody_max_velocity": prepared.melody_max_velocity,
        },
        "chord": {
            "enabled": prepared.chord_track_enabled,
            "key_root": prepared.key_root,
            "major_minor": prepared.major_minor,
            "emotion": prepared.emotion,
            "chord_style": prepared.chord_style,
            "chord_change_frequency": prepared.chord_change_frequency,
            "chord_octave_offset": prepared.chord_octave_offset,
            "chord_velocity": prepared.chord_velocity,
        },
    }


def relative_to_data_dir(path: Path, data_dir: Path) -> str:
    try:
        return path.resolve().relative_to(data_dir.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def artifact_entry(*, key: str, path: Path, data_dir: Path, job_id: Optional[str] = None, media_type: str = "application/octet-stream") -> Dict[str, Any]:
    entry = {
        "key": key,
        "file_name": path.name,
        "relative_path": relative_to_data_dir(path, data_dir),
        "media_type": media_type,
    }
    if job_id:
        entry["download_url"] = f"/api/jobs/{job_id}/artifacts/{key}"
    return entry


def save_preferences_submission(
    preferences_payload: Optional[Dict[str, Any]],
    *,
    preferences_ini: str = "",
    data_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    data_root = (data_dir or DATA_DIR).resolve()
    pref_dir = data_root / "preferences"
    pref_dir.mkdir(parents=True, exist_ok=True)
    normalized = normalize_preferences(preferences_payload, ini_content=preferences_ini)
    validate_preferences(normalized)
    preview = build_preview(prepare_generation({**normalized, "eegFileName": normalized.get("eegFileName") or "draft.csv"}))
    saved_at = now_iso()
    payload_path = pref_dir / "latest_preferences.payload.json"
    normalized_path = pref_dir / "latest_preferences.normalized.json"
    ini_path = pref_dir / "latest_preferences.ini"
    payload_path.write_text(json.dumps(preferences_payload or {}, ensure_ascii=False, indent=2), encoding="utf-8")
    normalized_path.write_text(json.dumps({"saved_at": saved_at, "normalized": normalized, "preview": preview}, ensure_ascii=False, indent=2), encoding="utf-8")
    ini_path.write_text(preferences_ini or "", encoding="utf-8")
    return {
        "saved_at": saved_at,
        "message": "Preferences validated and saved.",
        "preview": preview,
        "artifacts": {
            "payload": artifact_entry(key="preferences-payload", path=payload_path, data_dir=data_root, media_type="application/json"),
            "normalized": artifact_entry(key="preferences-json", path=normalized_path, data_dir=data_root, media_type="application/json"),
            "ini": artifact_entry(key="preferences-ini", path=ini_path, data_dir=data_root, media_type="text/plain"),
        },
    }

def _to_numeric_series(series: pd.Series) -> pd.Series:
    result = pd.to_numeric(series, errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


def _sanitize_instrument_names(music_obj: Any) -> None:
    for index, instrument in enumerate(getattr(music_obj, "instruments", [])):
        name = getattr(instrument, "name", "")
        if name is None:
            instrument.name = f"track_{index}"
        elif not isinstance(name, str):
            instrument.name = str(name)
        elif not name.strip():
            instrument.name = f"track_{index}"


def _clip_midi_pitch(value: int) -> int:
    return int(max(0, min(127, int(value))))


def _sanitize_midi_notes(music_obj: Any) -> None:
    for instrument in getattr(music_obj, "instruments", []):
        for note in getattr(instrument, "notes", []):
            note.pitch = _clip_midi_pitch(int(getattr(note, "pitch", 60)))
            note.velocity = int(max(1, min(127, int(getattr(note, "velocity", 64)))))
            start = int(max(0, int(getattr(note, "start", 0))))
            end = int(max(start + 1, int(getattr(note, "end", start + 1))))
            note.start = start
            note.end = end


def robust_fs_from_t(time_values: np.ndarray) -> Optional[float]:
    if time_values is None or len(time_values) < 3:
        return None
    deltas = np.diff(time_values.astype(np.float64, copy=False))
    deltas = deltas[np.isfinite(deltas)]
    deltas = deltas[deltas > 0]
    if len(deltas) == 0:
        return None
    delta_t = float(np.median(deltas))
    if (not np.isfinite(delta_t)) or delta_t <= 0:
        return None
    fs_value = 1.0 / delta_t
    if (not np.isfinite(fs_value)) or fs_value <= 0:
        return None
    return float(fs_value)


def safe_slice_segment(time_values: np.ndarray, channels: np.ndarray, start_s: float, duration_s: float, fs_estimate: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
    time_values = time_values.astype(np.float64, copy=False)
    if len(time_values) != channels.shape[0]:
        raise BridgeError("Timestamp length does not match data rows.")
    ok = np.isfinite(time_values)
    filtered_time = time_values[ok]
    filtered_channels = channels[ok, :]
    if len(filtered_time) < 10:
        raise BridgeError("Too few valid timestamps.")
    segment_start = float(filtered_time[0]) + float(start_s)
    segment_end = segment_start + float(duration_s)
    mask = (filtered_time >= segment_start) & (filtered_time <= segment_end)
    if np.any(mask) and int(mask.sum()) >= 10:
        return filtered_time[mask], filtered_channels[mask, :]
    fallback_fs = fs_estimate if fs_estimate and np.isfinite(fs_estimate) and fs_estimate > 0 else 512.0
    start_index = int(max(0, round(float(start_s) * fallback_fs)))
    length = int(max(10, round(float(duration_s) * fallback_fs)))
    end_index = min(len(filtered_time), start_index + length)
    if end_index - start_index < 10:
        start_index = max(0, len(filtered_time) - 10)
        end_index = len(filtered_time)
    return filtered_time[start_index:end_index], filtered_channels[start_index:end_index, :]


def downsample_limit(time_values: np.ndarray, channels: np.ndarray, fs_estimate: Optional[float], target_hz: int, max_points: int) -> Tuple[np.ndarray, np.ndarray, float]:
    if fs_estimate is None or (not np.isfinite(fs_estimate)) or fs_estimate <= 0:
        fs_estimate = robust_fs_from_t(time_values)
    if fs_estimate is None or (not np.isfinite(fs_estimate)) or fs_estimate <= 0:
        fs_estimate = 512.0
    stride = max(1, int(round(float(fs_estimate) / float(target_hz))))
    sampled_time = time_values[::stride]
    sampled_channels = channels[::stride, :]
    if len(sampled_time) > int(max_points):
        stride2 = max(1, int(np.ceil(len(sampled_time) / int(max_points))))
        sampled_time = sampled_time[::stride2]
        sampled_channels = sampled_channels[::stride2, :]
    sampled_fs = float(fs_estimate) / float(stride)
    check_fs = robust_fs_from_t(sampled_time)
    if check_fs is not None and np.isfinite(check_fs) and check_fs > 0:
        sampled_fs = float(check_fs)
    return sampled_time, sampled_channels, float(sampled_fs)


def patch_tool_generate_safe() -> None:
    import tool_generate as tool_generate

    if getattr(tool_generate, "_SAFE_PATCHED_EMMAQ", False):
        return

    if hasattr(tool_generate, "get_TPB_BPM"):
        original = tool_generate.get_TPB_BPM

        def safe_get_tpb_bpm(time_arr: Any, track_brain: Any, *args: Any, **kwargs: Any) -> Tuple[int, float]:
            try:
                tpb, bpm = original(time_arr, track_brain, *args, **kwargs)
                if (not np.isfinite(tpb)) or tpb <= 0:
                    tpb = 480
                if (not np.isfinite(bpm)) or bpm <= 0:
                    bpm = 120
                return int(tpb), float(bpm)
            except Exception:
                return 480, 120.0

        tool_generate.get_TPB_BPM = safe_get_tpb_bpm

    def wrap_named(function: Any) -> Any:
        def wrapped(time_arr: Any, track_brain: Any, *args: Any, **kwargs: Any) -> Any:
            track = track_brain
            if not hasattr(track, "name"):
                try:
                    track = pd.Series(np.asarray(track, dtype=float), name="track")
                except Exception:
                    track = pd.Series(list(track), name="track")
            if getattr(track, "name", None) is None:
                try:
                    track = track.rename("track")
                except Exception:
                    pass
            return function(time_arr, track, *args, **kwargs)

        return wrapped

    for candidate in ("EEG_MIDI_p2p_r2v", "EEG_MIDI_r2p_p2v"):
        if hasattr(tool_generate, candidate):
            setattr(tool_generate, candidate, wrap_named(getattr(tool_generate, candidate)))

    tool_generate._SAFE_PATCHED_EMMAQ = True


def pick_gen_func(rule_key: str) -> Tuple[Any, str]:
    patch_tool_generate_safe()
    import tool_generate as tool_generate

    candidate_names = ("EEG_MIDI_p2p_r2v",) if rule_key == "rule1" else ("EEG_MIDI_r2p_p2v",)
    for name in candidate_names:
        if hasattr(tool_generate, name):
            return getattr(tool_generate, name), name
    raise BridgeError(f"Could not find generator function for {rule_key}.")


def midi_to_bytes(midi_obj: Any) -> bytes:
    temp_path = TMP_DIR / f"tmp_{uuid.uuid4().hex}.mid"
    _sanitize_instrument_names(midi_obj)
    _sanitize_midi_notes(midi_obj)
    try:
        if hasattr(midi_obj, "dump"):
            midi_obj.dump(str(temp_path))
        elif hasattr(midi_obj, "save"):
            midi_obj.save(str(temp_path))
        else:
            raise BridgeError("Generated MIDI object does not support dump/save.")
        return temp_path.read_bytes()
    finally:
        temp_path.unlink(missing_ok=True)


def _read_midi_from_bytes(midi_bytes: bytes) -> Any:
    temp_path = TMP_DIR / f"in_{uuid.uuid4().hex}.mid"
    temp_path.write_bytes(midi_bytes)
    try:
        return mid_parser.MidiFile(str(temp_path))
    finally:
        temp_path.unlink(missing_ok=True)


def _midi_to_bytes(midi: Any) -> bytes:
    temp_path = TMP_DIR / f"out_{uuid.uuid4().hex}.mid"
    _sanitize_instrument_names(midi)
    _sanitize_midi_notes(midi)
    try:
        midi.dump(str(temp_path))
        return temp_path.read_bytes()
    finally:
        temp_path.unlink(missing_ok=True)


def _pick_target_instrument(midi: Any) -> Optional[Instrument]:
    instruments = [instrument for instrument in midi.instruments if not instrument.is_drum and instrument.notes]
    if not instruments:
        return None
    instruments.sort(key=lambda instrument: len(instrument.notes), reverse=True)
    return instruments[0]


def apply_fix(midi_bytes: bytes, fx_name: str) -> bytes:
    if fx_name == "none":
        return midi_bytes
    midi = _read_midi_from_bytes(midi_bytes)
    instrument = _pick_target_instrument(midi)
    if instrument is None:
        return midi_bytes
    notes = sorted(instrument.notes, key=lambda note: (note.start, note.pitch))
    if len(notes) < 4:
        return midi_bytes
    center = 60
    if fx_name == "FX1":
        pitches = np.array([note.pitch for note in notes], dtype=np.float64)
        median = float(np.median(pitches))
        mad = float(np.median(np.abs(pitches - median))) + 1e-9
        kept = []
        for note in notes:
            z_value = abs(float(note.pitch) - median) / (1.4826 * mad)
            if z_value <= 3.5:
                kept.append(note)
        if len(kept) >= 2:
            instrument.notes = kept
    elif fx_name == "FX2":
        for note in notes:
            pitch = int(note.pitch)
            while pitch > center + 12:
                pitch -= 12
            while pitch < center - 12:
                pitch += 12
            note.pitch = int(pitch)
    elif fx_name == "FX3":
        previous = int(notes[0].pitch)
        for note in notes[1:]:
            pitch = int(note.pitch)
            candidates = [pitch + 12 * step for step in (-3, -2, -1, 0, 1, 2, 3)]
            best = min(candidates, key=lambda candidate: abs(int(candidate) - previous))
            note.pitch = int(best)
            previous = int(note.pitch)
    elif fx_name == "FX4":
        for index in range(1, len(notes) - 1):
            left = int(notes[index - 1].pitch)
            current = int(notes[index].pitch)
            right = int(notes[index + 1].pitch)
            if (current > left and current > right) or (current < left and current < right):
                target = int(round((left + right) / 2))
                candidates = [target + 12 * step for step in (-3, -2, -1, 0, 1, 2, 3)]
                notes[index].pitch = int(min(candidates, key=lambda candidate: abs(int(candidate) - current)))
    instrument.notes = sorted(instrument.notes, key=lambda note: (note.start, note.pitch))
    return _midi_to_bytes(midi)


def quantize_midi(midi_bytes: bytes, grid: str, swing: float) -> bytes:
    if grid == "off":
        return midi_bytes
    midi = _read_midi_from_bytes(midi_bytes)
    ticks_per_beat = int(getattr(midi, "ticks_per_beat", 480) or 480)
    division = {"1/8": 2, "1/16": 4, "1/32": 8}.get(grid, 4)
    step = max(1, int(round(ticks_per_beat / division)))
    instrument = _pick_target_instrument(midi)
    if instrument is None:
        return midi_bytes
    swing = float(max(0.0, min(0.75, swing)))
    for note in instrument.notes:
        start = int(round(note.start / step) * step)
        end = int(round(note.end / step) * step)
        if division >= 4:
            index = int(round(start / step))
            if index % 2 == 1:
                start = int(start + swing * step)
                end = int(end + swing * step)
        if end <= start:
            end = start + step
        note.start = int(max(0, start))
        note.end = int(max(note.start + 1, end))
    instrument.notes = sorted(instrument.notes, key=lambda note: (note.start, note.pitch))
    return _midi_to_bytes(midi)


def _pcs_for_scale(mode: str, scale: str) -> List[int]:
    if scale == "diatonic":
        return MAJOR_SCALE if mode == "major" else MINOR_SCALE
    if scale == "penta":
        return PENTA_MAJOR if mode == "major" else PENTA_MINOR
    return list(range(12))


def apply_scale_constraint(midi_bytes: bytes, root_pc: int, mode: str, scale: str, lower_bound: int, upper_bound: int) -> bytes:
    allowed = {(int(root_pc) + pitch_class) % 12 for pitch_class in _pcs_for_scale(mode, scale)}
    midi = _read_midi_from_bytes(midi_bytes)
    instrument = _pick_target_instrument(midi)
    if instrument is None:
        return midi_bytes
    lower = int(lower_bound)
    upper = int(upper_bound)
    if lower >= upper:
        lower, upper = 48, 84
    for note in instrument.notes:
        pitch = int(max(lower, min(upper, int(note.pitch))))
        if scale == "chromatic":
            note.pitch = pitch
            continue
        if pitch % 12 in allowed:
            note.pitch = pitch
            continue
        best = None
        distance = 10 ** 9
        for delta in range(-9, 10):
            candidate = pitch + delta
            if candidate < lower or candidate > upper:
                continue
            if candidate % 12 in allowed:
                candidate_distance = abs(candidate - pitch)
                if candidate_distance < distance:
                    best = candidate
                    distance = candidate_distance
        note.pitch = int(pitch if best is None else best)
    instrument.notes = sorted(instrument.notes, key=lambda note: (note.start, note.pitch))
    return _midi_to_bytes(midi)


def estimate_key_from_melody(midi_bytes: bytes) -> Tuple[int, str]:
    midi = _read_midi_from_bytes(midi_bytes)
    instrument = _pick_target_instrument(midi)
    if instrument is None or len(instrument.notes) < 8:
        return 0, "major"
    counts = np.zeros(12, dtype=np.float64)
    for note in instrument.notes:
        counts[int(note.pitch) % 12] += 1.0
    best_score = float("-inf")
    best_root = 0
    best_mode = "major"
    for root in range(12):
        major_pcs = {(root + pitch_class) % 12 for pitch_class in MAJOR_SCALE}
        minor_pcs = {(root + pitch_class) % 12 for pitch_class in MINOR_SCALE}
        major_score = float(sum(counts[p] for p in major_pcs)) - 0.6 * float(sum(counts[p] for p in range(12) if p not in major_pcs))
        minor_score = float(sum(counts[p] for p in minor_pcs)) - 0.6 * float(sum(counts[p] for p in range(12) if p not in minor_pcs))
        if major_score > best_score:
            best_score = major_score
            best_root = root
            best_mode = "major"
        if minor_score > best_score:
            best_score = minor_score
            best_root = root
            best_mode = "minor"
    return int(best_root), str(best_mode)

def build_diatonic_triads(root_pc: int, mode: str) -> Dict[str, List[int]]:
    if mode == "major":
        degrees = {
            "I": [0, 4, 7],
            "ii": [2, 5, 9],
            "iii": [4, 7, 11],
            "IV": [5, 9, 0],
            "V": [7, 11, 2],
            "vi": [9, 0, 4],
            "vii_deg": [11, 2, 5],
        }
    else:
        degrees = {
            "i": [0, 3, 7],
            "ii_deg": [2, 5, 8],
            "III": [3, 7, 10],
            "iv": [5, 8, 0],
            "V": [7, 11, 2],
            "VI": [8, 0, 3],
            "VII": [10, 2, 5],
        }
    return {key: [int((int(root_pc) + pitch_class) % 12) for pitch_class in values] for key, values in degrees.items()}


def voice_triad(pitch_classes: List[int], base: int, spread: int) -> List[int]:
    root, third, fifth = [int(value) % 12 for value in pitch_classes[:3]]
    root_note = base + ((root - base) % 12)
    third_note = root_note + spread + ((third - (root_note + spread)) % 12)
    fifth_note = root_note + (2 * spread) + ((fifth - (root_note + (2 * spread))) % 12)
    return [int(root_note), int(third_note), int(fifth_note)]


def add_diatonic_chords(midi_bytes: bytes, *, emotion: str, root_pc: int, mode: str, chord_every_beats: float, velocity: int, octave_shift: int, style: str) -> bytes:
    midi = _read_midi_from_bytes(midi_bytes)
    ticks_per_beat = int(getattr(midi, "ticks_per_beat", 480) or 480)
    end_tick = 0
    for instrument in midi.instruments:
        for note in instrument.notes:
            if note.end > end_tick:
                end_tick = note.end
    if end_tick <= 0:
        end_tick = ticks_per_beat * 16
    total_beats = max(1, int(end_tick / ticks_per_beat))
    emotion_tag = EMOTION_TO_CHORD_TAG.get(emotion, "Calm")
    triads = build_diatonic_triads(int(root_pc), str(mode))
    if mode == "major":
        degrees = EMOTION_DEGREES_MAJOR.get(emotion_tag, EMOTION_DEGREES_MAJOR["Calm"])
        tonic_fallback = "I"
    else:
        degrees = EMOTION_DEGREES_MINOR.get(emotion_tag, EMOTION_DEGREES_MINOR["Calm"])
        tonic_fallback = "i"
    chord_instrument = Instrument(program=0, is_drum=False, name=f"CHORD_{emotion_tag}_{NOTE_NAMES[int(root_pc)]}_{mode}")
    step = max(1, int(round(float(chord_every_beats))))
    base_pitch = int(max(24, min(72, 48 + int(octave_shift) * 12)))
    clipped_velocity = int(max(1, min(127, int(velocity))))
    for beat_index in range(0, total_beats, step):
        degree = degrees[(beat_index // step) % len(degrees)]
        pitch_classes = triads.get(degree, triads.get(tonic_fallback, [0, 4, 7]))
        pitches = voice_triad(pitch_classes, base=base_pitch, spread=5)
        start = int(beat_index * ticks_per_beat)
        end = int(min((beat_index + step) * ticks_per_beat, end_tick))
        if end <= start:
            continue
        if style == "block":
            for pitch in pitches:
                chord_instrument.notes.append(Note(velocity=clipped_velocity, pitch=int(pitch), start=start, end=end))
        else:
            pattern = [0, 1, 2, 1]
            duration = max(1, int(ticks_per_beat / 2))
            cursor = start
            while cursor < end:
                index = pattern[int((cursor - start) / duration) % len(pattern)]
                pitch = pitches[index]
                chord_instrument.notes.append(Note(velocity=clipped_velocity, pitch=int(pitch), start=int(cursor), end=int(min(cursor + duration, end))))
                cursor += duration
    midi.instruments.append(chord_instrument)
    return _midi_to_bytes(midi)


def postprocess_midi(midi_bytes: bytes, *, transpose: int, vel_min: int, vel_max: int, duration_multiplier: float, thin: int, max_leap: int) -> bytes:
    midi = _read_midi_from_bytes(midi_bytes)
    instrument = _pick_target_instrument(midi)
    if instrument is None:
        return midi_bytes
    ticks_per_beat = int(getattr(midi, "ticks_per_beat", 480) or 480)
    vel_min = int(max(1, min(127, vel_min)))
    vel_max = int(max(1, min(127, vel_max)))
    if vel_min > vel_max:
        vel_min, vel_max = 40, 90
    duration_multiplier = float(max(0.25, min(2.5, duration_multiplier)))
    transpose = int(max(-24, min(24, transpose)))
    max_leap = int(max(3, min(24, max_leap)))
    notes = sorted(instrument.notes, key=lambda note: (note.start, note.pitch))
    for note in notes:
        note.pitch = int(note.pitch) + transpose
        velocity = int(note.velocity)
        if velocity < vel_min:
            velocity = vel_min
        if velocity > vel_max:
            velocity = vel_max
        note.velocity = int(velocity)
        duration = max(1, int(note.end) - int(note.start))
        note.end = int(note.start) + int(max(1, round(duration * duration_multiplier)))
    if len(notes) >= 2:
        previous = int(notes[0].pitch)
        for note in notes[1:]:
            pitch = int(note.pitch)
            if abs(pitch - previous) > max_leap:
                candidates = [pitch + 12 * step for step in (-3, -2, -1, 0, 1, 2, 3)]
                note.pitch = int(min(candidates, key=lambda candidate: abs(int(candidate) - previous)))
            previous = int(note.pitch)
    if thin > 0:
        step = max(1, int(round(ticks_per_beat / 4)))
        buckets: Dict[int, List[Note]] = {}
        for note in notes:
            buckets.setdefault(int(note.start // step), []).append(note)
        kept: List[Note] = []
        for bucket_notes in buckets.values():
            kept.extend(sorted(bucket_notes, key=lambda note: (-(note.velocity), note.pitch))[: int(thin)])
        notes = sorted(kept, key=lambda note: (note.start, note.pitch))
    instrument.notes = notes
    return _midi_to_bytes(midi)


def build_uniform_time(track_len: int, fs_estimate: float) -> np.ndarray:
    if (not np.isfinite(fs_estimate)) or fs_estimate <= 0:
        fs_estimate = 512.0
    return np.arange(int(track_len), dtype=np.float64) / float(fs_estimate)


def generate_rule_midi(time_values: np.ndarray, channels: np.ndarray, *, channel_idx: int, rule_key: str, numerator: int, denominator: int, magnet: int, pitches: Tuple[int, int, int], velocities: Tuple[int, int, int], fs_estimate: float) -> Tuple[bytes, str, str]:
    raw = channels[:, int(channel_idx)].astype(np.float64, copy=False)
    series = pd.Series(raw, name=f"ch_{int(channel_idx)}")
    series = series.iloc[np.where(np.isfinite(series.values))[0]].reset_index(drop=True)
    if len(series) < 20:
        raise BridgeError("Too few valid data points after channel filtering.")
    if float(np.nanstd(series.values)) < 1e-9:
        raise BridgeError("Selected EEG channel has almost no variation.")
    uniform_time = build_uniform_time(len(series), float(fs_estimate))
    gen_func, gen_name = pick_gen_func(rule_key)
    signature = inspect.signature(gen_func)
    parameters = signature.parameters
    kwargs: Dict[str, Any] = {}
    if "numerator" in parameters:
        kwargs["numerator"] = int(numerator)
    if "denominator" in parameters:
        kwargs["denominator"] = int(denominator)
    elif "denomimator" in parameters:
        kwargs["denomimator"] = int(denominator)
    elif "denom" in parameters:
        kwargs["denom"] = int(denominator)
    elif "den" in parameters:
        kwargs["den"] = int(denominator)
    if "magnet" in parameters:
        kwargs["magnet"] = int(magnet)
    if "pitches" in parameters:
        kwargs["pitches"] = tuple(int(value) for value in pitches)
    elif "pithces" in parameters:
        kwargs["pithces"] = tuple(int(value) for value in pitches)
    elif "pitch" in parameters:
        kwargs["pitch"] = tuple(int(value) for value in pitches)
    if "velocities" in parameters:
        kwargs["velocities"] = tuple(int(value) for value in velocities)
    elif "velocity" in parameters:
        kwargs["velocity"] = tuple(int(value) for value in velocities)
    if "fs" in parameters:
        kwargs["fs"] = float(fs_estimate)
    if "sr" in parameters:
        kwargs["sr"] = float(fs_estimate)
    midi_obj = gen_func(uniform_time, series, **kwargs)
    _sanitize_instrument_names(midi_obj)
    return midi_to_bytes(midi_obj), gen_name, str(signature)


def load_eeg_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    dataframe = pd.read_csv(csv_path, header=None)
    if dataframe.shape[1] < 2:
        raise BridgeError("CSV must have at least two columns: timestamp plus one EEG channel.")
    time_series = _to_numeric_series(dataframe.iloc[:, 0])
    channel_matrix = dataframe.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if time_series.isna().all():
        raise BridgeError("Timestamp column contains no numeric values.")
    return time_series.to_numpy(dtype=np.float64), channel_matrix.to_numpy(dtype=np.float64), dataframe


def resolve_maseval_script() -> Optional[Path]:
    configured = os.getenv("EMMAQ_MASEVAL_SCRIPT")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.append(BASE_DIR.parent / "MASEval" / "MASEval" / "music_eval_v4_single_cached_fixed_v2.template.py")
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


def run_optional_evaluation(*, final_midi_path: Path, evaluation_dir: Path, prepared: PreparedGeneration, data_dir: Path, job_id: str) -> Dict[str, Any]:
    if not bool_from_env("EMMAQ_ENABLE_MASEVAL", True):
        return {"status": "skipped", "message": "Evaluation disabled by EMMAQ_ENABLE_MASEVAL."}
    if not os.getenv("DEEPSEEK_API_KEY"):
        return {"status": "skipped", "message": "DEEPSEEK_API_KEY is not set; evaluation skipped."}
    script_path = resolve_maseval_script()
    if script_path is None:
        return {"status": "skipped", "message": "MASEval script not found; evaluation skipped."}
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    report_path = evaluation_dir / "evaluation_report.json"
    cache_dir = evaluation_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    command = [
        resolve_maseval_python(script_path),
        str(script_path),
        "--midi", str(final_midi_path.resolve()),
        "--out", str(report_path.resolve()),
        "--cache_dir", str(cache_dir.resolve()),
        "--target_style", prepared.emotion.lower(),
        "--intended_use", "interactive_generation",
    ]
    try:
        completed = subprocess.run(command, cwd=str(script_path.parent), capture_output=True, text=True, timeout=parse_int(os.getenv("EMMAQ_MASEVAL_TIMEOUT_SECONDS"), 600), check=False, env=os.environ.copy())
    except subprocess.TimeoutExpired as error:
        return {"status": "error", "message": f"Evaluation timed out after {error.timeout} seconds."}
    except Exception as error:
        return {"status": "error", "message": f"Evaluation failed to start: {type(error).__name__}: {error}"}
    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    if completed.returncode != 0:
        return {"status": "error", "message": "Evaluation command failed.", "returncode": completed.returncode, "stdout": stdout[-4000:], "stderr": stderr[-4000:]}
    report = None
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            report = None
    return {
        "status": "completed",
        "message": "Evaluation completed.",
        "artifact": artifact_entry(key="evaluation-report", path=report_path, data_dir=data_dir, job_id=job_id, media_type="application/json") if report_path.exists() else None,
        "report": report,
        "stdout": stdout[-4000:],
        "stderr": stderr[-4000:],
    }


def run_generation_job(preferences_payload: Optional[Dict[str, Any]], *, preferences_ini: str, eeg_file_name: str, eeg_bytes: bytes, data_dir: Optional[Path] = None, job_id: Optional[str] = None, enable_evaluation: Optional[bool] = None) -> Dict[str, Any]:
    if not eeg_file_name.lower().endswith(".csv"):
        raise BridgeError("Only .csv EEG uploads are supported.")
    data_root = (data_dir or DATA_DIR).resolve()
    normalized = normalize_preferences(preferences_payload, ini_content=preferences_ini, eeg_file_name=eeg_file_name)
    prepared = prepare_generation(normalized, job_id=job_id)
    job_root = data_root / "jobs" / prepared.job_id
    upload_dir = job_root / "uploads"
    output_dir = job_root / "outputs"
    evaluation_dir = job_root / "evaluation"
    for directory in (upload_dir, output_dir):
        directory.mkdir(parents=True, exist_ok=True)
    input_csv_path = upload_dir / sanitize_name(eeg_file_name, f"{prepared.job_id}.csv")
    input_csv_path.write_bytes(eeg_bytes)
    preferences_payload_path = job_root / "preferences.payload.json"
    preferences_payload_path.write_text(json.dumps(preferences_payload or {}, ensure_ascii=False, indent=2), encoding="utf-8")
    preferences_normalized_path = job_root / "preferences.normalized.json"
    preferences_normalized_path.write_text(json.dumps({"normalized": normalized, "preview": build_preview(prepared)}, ensure_ascii=False, indent=2), encoding="utf-8")
    preferences_ini_path = job_root / "preferences.ini"
    preferences_ini_path.write_text(preferences_ini or "", encoding="utf-8")
    time_values, channels, dataframe = load_eeg_csv(input_csv_path)
    if prepared.melody_channel < 0 or prepared.melody_channel >= channels.shape[1]:
        raise BridgeError(f"melodyChannel must be between 0 and {channels.shape[1] - 1} for the uploaded EEG file.")
    fs_estimate = robust_fs_from_t(time_values)
    sliced_time, sliced_channels = safe_slice_segment(time_values, channels, start_s=prepared.start_time, duration_s=prepared.duration, fs_estimate=fs_estimate)
    run_time, run_channels, run_fs = downsample_limit(sliced_time, sliced_channels, fs_estimate=fs_estimate, target_hz=prepared.target_sampling_rate, max_points=prepared.max_points)

    artifacts: Dict[str, Dict[str, Any]] = {
        "input-csv": artifact_entry(key="input-csv", path=input_csv_path, data_dir=data_root, job_id=prepared.job_id, media_type="text/csv"),
        "preferences-json": artifact_entry(key="preferences-json", path=preferences_normalized_path, data_dir=data_root, job_id=prepared.job_id, media_type="application/json"),
        "preferences-ini": artifact_entry(key="preferences-ini", path=preferences_ini_path, data_dir=data_root, job_id=prepared.job_id, media_type="text/plain"),
    }
    stages: List[Dict[str, Any]] = []

    midi_bytes, generator_name, signature = generate_rule_midi(run_time, run_channels, channel_idx=prepared.melody_channel, rule_key=prepared.rule_key, numerator=prepared.numerator, denominator=prepared.denominator, magnet=prepared.magnet, pitches=prepared.pitches, velocities=prepared.velocities, fs_estimate=run_fs)

    def save_stage(key: str, label: str, current_bytes: bytes, extra: Optional[Dict[str, Any]] = None) -> bytes:
        stage_path = output_dir / f"{prepared.output_prefix}_{key}.mid"
        stage_path.write_bytes(current_bytes)
        artifacts[key] = artifact_entry(key=key, path=stage_path, data_dir=data_root, job_id=prepared.job_id, media_type="audio/midi")
        item = {"key": key, "label": label, "artifact": artifacts[key]}
        if extra:
            item.update(extra)
        stages.append(item)
        return current_bytes

    save_stage("stage-01-rule", "rule", midi_bytes, {"generator_name": generator_name, "signature": signature})
    if prepared.fx_choice != "none":
        midi_bytes = apply_fix(midi_bytes, prepared.fx_choice)
        save_stage("stage-02-fx", prepared.fx_choice, midi_bytes)
    estimated_root, estimated_mode = estimate_key_from_melody(midi_bytes)
    root_pc = int(estimated_root) if prepared.key_root == "Auto" else int(NAME_TO_PC[prepared.key_root])
    mode = str(estimated_mode) if prepared.major_minor == "Auto" else prepared.major_minor.lower()
    midi_bytes = apply_scale_constraint(midi_bytes, root_pc=root_pc, mode=mode, scale=prepared.scale_mode, lower_bound=prepared.pitch_range_min, upper_bound=prepared.pitch_range_max)
    save_stage("stage-03-scale", "scale", midi_bytes, {"applied_key": NOTE_NAMES[root_pc], "applied_mode": mode})
    if prepared.quantization != "off":
        midi_bytes = quantize_midi(midi_bytes, prepared.quantization, prepared.swing_amount)
        save_stage("stage-04-quantize", "quantize", midi_bytes, {"grid": prepared.quantization, "swing_amount": prepared.swing_amount})
    midi_bytes = postprocess_midi(midi_bytes, transpose=prepared.transpose, vel_min=prepared.melody_min_velocity, vel_max=prepared.melody_max_velocity, duration_multiplier=prepared.note_duration_multiplier, thin=prepared.sparsity, max_leap=prepared.max_leap)
    save_stage("stage-05-postprocess", "postprocess", midi_bytes)
    if prepared.chord_track_enabled:
        midi_bytes = add_diatonic_chords(midi_bytes, emotion=prepared.emotion, root_pc=root_pc, mode=mode, chord_every_beats=prepared.chord_change_frequency, velocity=prepared.chord_velocity, octave_shift=prepared.chord_octave_offset, style=prepared.chord_style)
        save_stage("stage-06-chords", "chords", midi_bytes, {"emotion": prepared.emotion})

    final_midi_path = output_dir / f"{prepared.output_prefix}_final.mid"
    final_midi_path.write_bytes(midi_bytes)
    artifacts["final-midi"] = artifact_entry(key="final-midi", path=final_midi_path, data_dir=data_root, job_id=prepared.job_id, media_type="audio/midi")

    evaluation_enabled = bool_from_env("EMMAQ_ENABLE_MASEVAL", True) if enable_evaluation is None else bool(enable_evaluation)
    evaluation = run_optional_evaluation(final_midi_path=final_midi_path, evaluation_dir=evaluation_dir, prepared=prepared, data_dir=data_root, job_id=prepared.job_id) if evaluation_enabled else {"status": "skipped", "message": "Evaluation disabled for this request."}
    if isinstance(evaluation.get("artifact"), dict):
        artifacts["evaluation-report"] = evaluation["artifact"]

    manifest = {
        "job_id": prepared.job_id,
        "created_at": now_iso(),
        "eeg_file_name": eeg_file_name,
        "row_count": int(dataframe.shape[0]),
        "channel_count": int(channels.shape[1]),
        "estimated_fs": float(fs_estimate) if fs_estimate is not None else None,
        "prepared": build_preview(prepared),
        "estimated_key": {"detected_root": NOTE_NAMES[int(estimated_root)], "detected_mode": estimated_mode, "applied_root": NOTE_NAMES[int(root_pc)], "applied_mode": mode},
        "artifacts": artifacts,
        "stages": stages,
        "evaluation": evaluation,
    }
    (job_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "job_id": prepared.job_id,
        "message": "Generation completed.",
        "download_url": artifacts["final-midi"]["download_url"],
        "final_midi": artifacts["final-midi"],
        "stages": stages,
        "estimated_key": manifest["estimated_key"],
        "prepared": manifest["prepared"],
        "evaluation": evaluation,
    }


def build_legacy_preferences(*, channel: int, seconds: float, rule: str) -> Dict[str, Any]:
    if rule == "p2p_r2v":
        ui_rule = "Rule1"
    elif rule == "r2p_p2v":
        ui_rule = "Rule2"
    else:
        raise BridgeError("Legacy rule must be p2p_r2v or r2p_p2v.")
    return {"flat": {**DEFAULT_FLAT_PREFERENCES, "melodyChannel": max(0, int(channel) - 1), "duration": int(seconds), "rule": ui_rule}}


def _load_json_file(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_text_file(path: Optional[str]) -> str:
    if not path:
        return ""
    return Path(path).read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EMMA-Q parameterized EEG-to-MIDI generation without launching a local web page.")
    parser.add_argument("--csv", required=True, help="Path to EEG CSV file")
    parser.add_argument("--preferences-json-file", help="Path to saved EMMA-Q JSON payload")
    parser.add_argument("--preferences-ini-file", help="Path to saved EMMA-Q INI file")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Data directory for job outputs")
    parser.add_argument("--job-id", help="Optional fixed job id")
    parser.add_argument("--skip-eval", action="store_true", help="Skip optional MASEval execution")
    args = parser.parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV file not found: {csv_path}")
    result = run_generation_job(
        _load_json_file(args.preferences_json_file),
        preferences_ini=_load_text_file(args.preferences_ini_file),
        eeg_file_name=csv_path.name,
        eeg_bytes=csv_path.read_bytes(),
        data_dir=Path(args.data_dir),
        job_id=args.job_id,
        enable_evaluation=not args.skip_eval,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


