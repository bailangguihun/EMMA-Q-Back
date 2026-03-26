from __future__ import annotations

import argparse
import ast
import json
import os
import re
import time
from bisect import bisect_right
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pretty_midi
from openai import OpenAI

DEFAULT_MODEL = "deepseek-chat"

JUDGES = [
    {"name": "harmony_judge", "focus": "harmony and voice leading"},
    {"name": "rhythm_judge", "focus": "rhythm and groove"},
    {"name": "structure_judge", "focus": "structure and phrasing"},
    {"name": "style_judge", "focus": "style consistency and listenability"},
]

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def clamp_int(x: Any, lo: int, hi: int, default: int = 0) -> int:
    try:
        v = int(x)
    except Exception:
        v = default
    return max(lo, min(hi, v))


def _strip_code_fence(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s.strip())
    return s


def _extract_balanced_json_candidates(s: str) -> List[str]:
    out: List[str] = []
    start = None
    depth = 0
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                out.append(s[start : i + 1])
                start = None
    return out


def safe_json_loads(text: str) -> Dict[str, Any]:
    s0 = _strip_code_fence(text)
    candidates = [s0, *_extract_balanced_json_candidates(s0)]
    for cand in candidates:
        if not cand:
            continue
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        try:
            obj = ast.literal_eval(cand)
            if isinstance(obj, dict):
                return json.loads(json.dumps(obj))
        except Exception:
            pass
    raise ValueError("Could not parse JSON object from model output.")


def ensure_env() -> None:
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise SystemExit("Missing env var: DEEPSEEK_API_KEY")
    if not os.getenv("DEEPSEEK_BASE_URL"):
        os.environ["DEEPSEEK_BASE_URL"] = "https://api.deepseek.com/v1"


def llm_client() -> OpenAI:
    ensure_env()
    return OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    )


def call_llm_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int = 950,
    max_try: int = 3,
) -> Dict[str, Any]:
    last_err: Exception | None = None
    for _ in range(max_try):
        raw = ""
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            raw = (resp.choices[0].message.content or "").strip()
            return safe_json_loads(raw)
        except Exception as e:
            last_err = e
            try:
                fixer = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You fix JSON. Return one strict JSON object only."},
                        {
                            "role": "user",
                            "content": "Repair the following output into strict JSON:\n\n" + raw,
                        },
                    ],
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                repaired = (fixer.choices[0].message.content or "").strip()
                return safe_json_loads(repaired)
            except Exception as e2:
                last_err = e2
    raise last_err or RuntimeError("LLM JSON parsing failed.")


def time_signature(pm: pretty_midi.PrettyMIDI) -> Tuple[int, int]:
    if pm.time_signature_changes:
        ts = pm.time_signature_changes[0]
        return int(ts.numerator), int(ts.denominator)
    return 4, 4


def tempo_at(pm: pretty_midi.PrettyMIDI, t: float) -> float:
    tempo_times, tempi = pm.get_tempo_changes()
    if len(tempi) == 0:
        return 120.0
    i = bisect_right(list(tempo_times), t) - 1
    if i < 0:
        return float(tempi[0])
    return float(tempi[i])


def pick_main_instruments(
    pm: pretty_midi.PrettyMIDI,
    keep: int = 2,
    include_drums: bool = False,
) -> List[pretty_midi.Instrument]:
    insts = [inst for inst in pm.instruments if include_drums or not inst.is_drum]
    insts.sort(key=lambda x: len(x.notes), reverse=True)
    return insts[: max(1, keep)]


def infer_triad(pitch_classes: List[int]) -> Optional[str]:
    pcs = set(pitch_classes)
    if not pcs:
        return None
    best_name = None
    best_score = 0
    for root in range(12):
        major = {root, (root + 4) % 12, (root + 7) % 12}
        minor = {root, (root + 3) % 12, (root + 7) % 12}
        major_score = len(pcs & major)
        minor_score = len(pcs & minor)
        if major_score >= 2 and major_score > best_score:
            best_score = major_score
            best_name = f"{NOTE_NAMES[root]}:maj"
        if minor_score >= 2 and minor_score > best_score:
            best_score = minor_score
            best_name = f"{NOTE_NAMES[root]}:min"
    return best_name


def midi_global_summary(pm: pretty_midi.PrettyMIDI, midi_path: str) -> Dict[str, Any]:
    end_t = float(pm.get_end_time())
    _, tempi = pm.get_tempo_changes()
    tempo0 = float(tempi[0]) if len(tempi) else None
    num, den = time_signature(pm)

    notes: List[pretty_midi.Note] = []
    drum_inst_count = 0
    for inst in pm.instruments:
        if inst.is_drum:
            drum_inst_count += 1
        notes.extend(inst.notes)

    note_count = len(notes)
    if note_count > 0:
        pitch_min = int(min(n.pitch for n in notes))
        pitch_max = int(max(n.pitch for n in notes))
        velocity_avg = float(sum(n.velocity for n in notes) / note_count)
    else:
        pitch_min = None
        pitch_max = None
        velocity_avg = None

    density = float(note_count / end_t) if end_t > 1e-9 else None

    return {
        "midi_path": midi_path,
        "duration_s": end_t,
        "tempo_bpm_first": tempo0,
        "time_signature": f"{num}/{den}",
        "instrument_count": len(pm.instruments),
        "drum_instrument_count": drum_inst_count,
        "note_count": note_count,
        "note_density": density,
        "pitch_min": pitch_min,
        "pitch_max": pitch_max,
        "velocity_avg": velocity_avg,
    }


def midi_symbolic_excerpt(
    pm: pretty_midi.PrettyMIDI,
    *,
    max_measures: int = 8,
    steps_per_beat: int = 4,
    keep_top_instruments: int = 2,
    max_events_total: int = 220,
    include_drums: bool = False,
) -> Dict[str, Any]:
    beats = list(pm.get_beats())
    end_t = float(pm.get_end_time())
    num, den = time_signature(pm)
    beats_per_measure = max(1, num)

    if not beats:
        beats = [0.0]

    measure_starts = [beats[i] for i in range(0, len(beats), beats_per_measure)]
    if not measure_starts:
        measure_starts = [0.0]

    mcount = min(max(1, max_measures), len(measure_starts))
    last_end = measure_starts[mcount] if mcount < len(measure_starts) else end_t

    measures: Dict[int, Dict[str, Any]] = {m: {"m": m, "events": []} for m in range(1, mcount + 1)}
    events_total = 0

    for inst in pick_main_instruments(pm, keep=keep_top_instruments, include_drums=include_drums):
        for note in inst.notes:
            if note.start >= last_end or note.end <= 0:
                continue
            mi = bisect_right(measure_starts[:mcount], note.start) - 1
            if mi < 0:
                mi = 0
            if mi >= mcount:
                continue
            measure_id = mi + 1

            bi = bisect_right(beats, note.start) - 1
            if bi < 0:
                bi = 0
            beat_in_measure = max(0, bi - mi * beats_per_measure)

            frac = 0.0
            if bi + 1 < len(beats):
                span = beats[bi + 1] - beats[bi]
                if span > 1e-9:
                    frac = (note.start - beats[bi]) / span
            beat_pos = float(beat_in_measure) + frac

            bpm = tempo_at(pm, note.start)
            sec_per_beat = 60.0 / max(1e-6, bpm)
            dur_beats = max(0.0, (note.end - note.start) / sec_per_beat)

            event = {
                "s": int(round(beat_pos * steps_per_beat)),
                "d": max(1, int(round(dur_beats * steps_per_beat))),
                "p": int(note.pitch),
                "n": pretty_midi.note_number_to_name(int(note.pitch)),
                "v": int(note.velocity),
                "prog": int(inst.program),
                "drum": bool(inst.is_drum),
            }
            measures[measure_id]["events"].append(event)
            events_total += 1
            if events_total >= max_events_total:
                break
        if events_total >= max_events_total:
            break

    out_measures: List[Dict[str, Any]] = []
    for m in range(1, mcount + 1):
        evs = measures[m]["events"]
        evs.sort(key=lambda x: (x["s"], x["p"]))
        evs = evs[:50]
        pcs = [int(e["p"]) % 12 for e in evs if not e.get("drum")]
        chord = infer_triad(pcs)
        pitch_range = [min(e["p"] for e in evs), max(e["p"] for e in evs)] if evs else None
        out_measures.append(
            {
                "m": m,
                "chord": chord,
                "pitch_range": pitch_range,
                "event_count": len(evs),
                "events": evs,
            }
        )

    _, tempi = pm.get_tempo_changes()
    tempo0 = float(tempi[0]) if len(tempi) else None

    return {
        "time_signature": f"{num}/{den}",
        "tempo_bpm_first": tempo0,
        "steps_per_beat": steps_per_beat,
        "max_measures": mcount,
        "measures": out_measures,
    }


def compress_peer(judge_out: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "judge_name": judge_out.get("judge_name"),
        "focus": judge_out.get("focus"),
        "score": judge_out.get("score"),
        "pros": (judge_out.get("pros") or [])[:4],
        "cons": (judge_out.get("cons") or [])[:4],
        "suggestions": (judge_out.get("suggestions") or [])[:4],
        "rationale": (judge_out.get("rationale") or "")[:260],
    }


def fallback_judge(name: str, focus: str, reason: str) -> Dict[str, Any]:
    return {
        "judge_name": name,
        "focus": focus,
        "score": 12,
        "pros": [],
        "cons": [f"LLM stage failed, fallback used: {reason}"],
        "suggestions": ["Retry with a shorter max_measures value."],
        "rationale": "fallback",
    }


def fallback_chair(scores: Dict[str, int], reason: str) -> Dict[str, Any]:
    total = sum(scores.values())
    return {
        "scores": scores,
        "score_total": clamp_int(total, 0, 100, total),
        "pros": [],
        "cons": [f"Chair stage failed, fallback used: {reason}"],
        "suggestions": ["Retry chair stage."],
        "overall_comment": "Final summary generated by fallback.",
    }


def run_multi_agent(
    global_meta: Dict[str, Any],
    symbolic: Dict[str, Any],
    *,
    target_style: str,
    intended_use: str,
    model_judge: str,
    model_chair: str,
) -> Dict[str, Any]:
    client = llm_client()

    global_s = json.dumps(global_meta, ensure_ascii=False)
    sym_s = json.dumps(symbolic, ensure_ascii=False)
    context = f"Target style: {target_style}\nIntended use: {intended_use}\nOutput language: Simplified Chinese."

    judge_system = (
        "You are a music judge. Return one JSON object only.\n"
        "Use score range 0-25 integer.\n"
        "Schema: {judge_name:str,focus:str,score:int,pros:[str],cons:[str],suggestions:[str],rationale:str}"
    )

    round1: Dict[str, Any] = {}
    for j in JUDGES:
        user = (
            f"{context}\nJudge focus: {j['focus']}\n\n"
            f"Global summary (JSON): {global_s}\n\n"
            f"Symbolic excerpt (JSON): {sym_s}\n\n"
            "Return strict JSON in the schema."
        )
        try:
            out = call_llm_json(client, model_judge, judge_system, user, max_tokens=900, max_try=3)
            out["judge_name"] = j["name"]
            out["focus"] = j["focus"]
            out["score"] = clamp_int(out.get("score", 12), 0, 25, 12)
            out.setdefault("pros", [])
            out.setdefault("cons", [])
            out.setdefault("suggestions", [])
            out.setdefault("rationale", "")
            round1[j["name"]] = out
        except Exception as e:
            round1[j["name"]] = fallback_judge(j["name"], j["focus"], repr(e))

    revise_system = (
        "You are a music judge. Return one JSON object only.\n"
        "You already did initial scoring. Review peer summaries and revise for consistency and actionability.\n"
        "Score range 0-25 integer. Keep schema unchanged."
    )

    round2: Dict[str, Any] = {}
    peers_comp = {k: compress_peer(v) for k, v in round1.items()}
    for j in JUDGES:
        me = round1[j["name"]]
        peers = [peers_comp[k] for k in peers_comp if k != j["name"]]
        user = (
            f"{context}\nJudge focus: {j['focus']}\n\n"
            f"Global summary (JSON): {global_s}\n\n"
            f"Symbolic excerpt (JSON): {sym_s}\n\n"
            f"Your round1 (JSON): {json.dumps(compress_peer(me), ensure_ascii=False)}\n\n"
            f"Peer summaries (JSON): {json.dumps(peers, ensure_ascii=False)}\n\n"
            "Return strict JSON in the schema."
        )
        try:
            out = call_llm_json(client, model_judge, revise_system, user, max_tokens=950, max_try=3)
            out["judge_name"] = j["name"]
            out["focus"] = j["focus"]
            out["score"] = clamp_int(out.get("score", me.get("score", 12)), 0, 25, 12)
            out.setdefault("pros", [])
            out.setdefault("cons", [])
            out.setdefault("suggestions", [])
            out.setdefault("rationale", "")
            round2[j["name"]] = out
        except Exception as e:
            keep = dict(me)
            keep["score"] = clamp_int(me.get("score", 12), 0, 25, 12)
            keep["cons"] = list(keep.get("cons", [])) + [f"Round2 fallback used: {repr(e)}"]
            round2[j["name"]] = keep

    scores = {
        "harmony": clamp_int(round2["harmony_judge"].get("score", 12), 0, 25, 12),
        "rhythm": clamp_int(round2["rhythm_judge"].get("score", 12), 0, 25, 12),
        "structure": clamp_int(round2["structure_judge"].get("score", 12), 0, 25, 12),
        "style": clamp_int(round2["style_judge"].get("score", 12), 0, 25, 12),
    }

    chair_system = (
        "You are the chair judge. Return one JSON object only.\n"
        "Schema: {"
        "score_total:int,"
        "scores:{harmony:int,rhythm:int,structure:int,style:int},"
        "pros:[str],cons:[str],suggestions:[str],overall_comment:str"
        "}\n"
        "Use 0-25 for each score and 0-100 total."
    )
    chair_user = (
        f"{context}\n"
        f"Global summary (JSON): {global_s}\n\n"
        f"Symbolic excerpt (JSON): {sym_s}\n\n"
        f"Round2 outputs (JSON): {json.dumps({k: compress_peer(v) for k, v in round2.items()}, ensure_ascii=False)}\n\n"
        "Return strict JSON. Keep suggestions concrete."
    )

    try:
        chair = call_llm_json(client, model_chair, chair_system, chair_user, max_tokens=950, max_try=4)
        chair.setdefault("scores", scores)
        chair["scores"]["harmony"] = clamp_int(chair["scores"].get("harmony", scores["harmony"]), 0, 25, scores["harmony"])
        chair["scores"]["rhythm"] = clamp_int(chair["scores"].get("rhythm", scores["rhythm"]), 0, 25, scores["rhythm"])
        chair["scores"]["structure"] = clamp_int(chair["scores"].get("structure", scores["structure"]), 0, 25, scores["structure"])
        chair["scores"]["style"] = clamp_int(chair["scores"].get("style", scores["style"]), 0, 25, scores["style"])
        score_default = sum(chair["scores"].values())
        chair["score_total"] = clamp_int(chair.get("score_total", score_default), 0, 100, score_default)
        chair.setdefault("pros", [])
        chair.setdefault("cons", [])
        chair.setdefault("suggestions", [])
        chair.setdefault("overall_comment", "")
    except Exception as e:
        chair = fallback_chair(scores, repr(e))

    return {
        "round1": round1,
        "round2": round2,
        "chair": chair,
    }


def ensure_text_fields(report: Dict[str, Any]) -> Dict[str, Any]:
    report["pros"] = report.get("pros") if isinstance(report.get("pros"), list) else []
    report["cons"] = report.get("cons") if isinstance(report.get("cons"), list) else []
    report["suggestions"] = report.get("suggestions") if isinstance(report.get("suggestions"), list) else []
    overall = report.get("overall_comment")
    if not isinstance(overall, str) or not overall.strip():
        p1 = report["pros"][0] if report["pros"] else "No major strengths identified."
        p2 = report["cons"][0] if report["cons"] else "No major weaknesses identified."
        report["overall_comment"] = f"Summary: {p1} {p2}"
    return report


def evaluate_one(
    midi_path: Path,
    *,
    out_path: Path,
    model_judge: str,
    model_chair: str,
    max_measures: int,
    target_style: str,
    intended_use: str,
) -> Dict[str, Any]:
    if not midi_path.exists() or not midi_path.is_file():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    global_meta = midi_global_summary(pm, str(midi_path.resolve()))
    symbolic = midi_symbolic_excerpt(pm, max_measures=max(1, max_measures))

    llm_pack = run_multi_agent(
        global_meta,
        symbolic,
        target_style=target_style,
        intended_use=intended_use,
        model_judge=model_judge,
        model_chair=model_chair,
    )
    chair = llm_pack["chair"]

    report: Dict[str, Any] = {
        "version": "music_eval_competition_light_v1",
        "ts": int(time.time()),
        "file": str(midi_path.resolve()),
        "run_cfg": {
            "model_judge": model_judge,
            "model_chair": model_chair,
            "max_measures": max(1, max_measures),
            "target_style": target_style,
            "intended_use": intended_use,
        },
        "midi_meta": global_meta,
        "symbolic_excerpt": symbolic,
        "scores": chair.get("scores", {}),
        "score_total": clamp_int(chair.get("score_total", 0), 0, 100, 0),
        "pros": chair.get("pros", []),
        "cons": chair.get("cons", []),
        "suggestions": chair.get("suggestions", []),
        "overall_comment": chair.get("overall_comment", ""),
        "judge_traces_round1": llm_pack["round1"],
        "judge_traces": llm_pack["round2"],
    }
    report = ensure_text_fields(report)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Lightweight multi-agent MIDI evaluation for competition use.")
    ap.add_argument("--midi", required=True, help="Path to MIDI file")
    ap.add_argument("--out", default="report_competition_light.json", help="Output JSON path")
    ap.add_argument("--model_judge", default=DEFAULT_MODEL)
    ap.add_argument("--model_chair", default=DEFAULT_MODEL)
    ap.add_argument("--max_measures", type=int, default=8)
    ap.add_argument("--target_style", default="general")
    ap.add_argument("--intended_use", default="competition")
    args = ap.parse_args()

    report = evaluate_one(
        Path(args.midi),
        out_path=Path(args.out),
        model_judge=args.model_judge,
        model_chair=args.model_chair,
        max_measures=max(1, int(args.max_measures)),
        target_style=args.target_style,
        intended_use=args.intended_use,
    )
    print("OK ->", Path(args.out).resolve())
    print("score_total =", report.get("score_total"))


if __name__ == "__main__":
    main()
