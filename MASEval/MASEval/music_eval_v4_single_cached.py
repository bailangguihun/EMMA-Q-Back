# -*- coding: utf-8 -*-
"""
music_eval_v4_single_cached.py
Single MIDI evaluation with:
- symbolic evidence (measure-level events)
- rule metrics
- multi-agent (Round1 -> moderator -> Round2 -> chair)
- structured edit_plan
- stage-level caching (each LLM call cached)
- after-loop also stage-cached (edited midi hash)
No batch/compare. Designed for stability + speed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pretty_midi
from openai import OpenAI, APIConnectionError

# -----------------------------
# Basic config
# -----------------------------

DEFAULT_MODEL = "deepseek-chat"

JUDGES = [
    {"name": "harmony_judge", "focus": "和声/乐理（基于音高集合、和弦/调性线索、声部进行）"},
    {"name": "rhythm_judge", "focus": "节奏/律动（基于拍网格、起音分布、切分、密度与稳定性）"},
    {"name": "structure_judge", "focus": "结构/段落（基于重复/对比、动机、密度与音域变化）"},
    {"name": "style_judge", "focus": "风格一致性/可听性（基于音域、跳进、织体、稳定性与目标风格）"},
]

NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
MAJOR_SCALE = {0,2,4,5,7,9,11}
MINOR_SCALE = {0,2,3,5,7,8,10}


def now_ts() -> int:
    return int(time.time())


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def clamp_int(x: Any, lo: int, hi: int, default: int = 0) -> int:
    try:
        v = int(x)
    except Exception:
        v = default
    return max(lo, min(hi, v))


def safe_json_loads(text: str) -> Dict[str, Any]:
    # strict first
    try:
        return json.loads(text)
    except Exception:
        # fallback: extract first {...}
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise ValueError("No JSON object found in model output.")
        return json.loads(m.group(0))


def llm_client() -> OpenAI:
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    api_key = os.environ["DEEPSEEK_API_KEY"]
    return OpenAI(api_key=api_key, base_url=base_url)


# -----------------------------
# Stage cache (per call)
# -----------------------------

def stage_cache_path(cache_dir: Path, midi_hash: str, cfg_hash: str, stage: str) -> Path:
    # stage should be filesystem-safe
    stage = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", stage)
    d = cache_dir / midi_hash / cfg_hash
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{stage}.json"


def load_stage(cache_path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_stage(cache_path: Path, obj: Dict[str, Any]) -> None:
    cache_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def call_llm_json_stage(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    cache_path: Path,
    max_tokens: int = 1000,
    max_try: int = 5,
) -> Dict[str, Any]:
    """
    Stage cache hit => return immediately
    Otherwise: JSON mode + retries + (optional) JSON repair pass
    """
    cached = load_stage(cache_path)
    if cached is not None:
        return cached

    # DeepSeek JSON Output requires prompt contains 'json'
    if "json" not in (system or "").lower() and "json" not in (user or "").lower():
        system = (system or "") + "\\njson"

    last_err: Exception | None = None
    for attempt in range(max_try):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content or ""

            try:
                obj = safe_json_loads(text)
            except Exception:
                # second-pass JSON repair
                fix = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "?? JSON ????????? JSON?json???????????"},
                        {"role": "user", "content": "?????????? JSON???? JSON?\\n\\n" + text},
                    ],
                    temperature=0.0,
                    max_tokens=max(300, min(max_tokens, 900)),
                    response_format={"type": "json_object"},
                )
                text2 = fix.choices[0].message.content or ""
                obj = safe_json_loads(text2)

            save_stage(cache_path, obj)
            return obj

        except (APIConnectionError, httpx.RemoteProtocolError, httpx.ReadTimeout, httpx.ConnectError) as e:
            last_err = e
            time.sleep(min(2 ** attempt, 8) + random.random())
            continue
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 6) + random.random())
            continue

    raise last_err or RuntimeError("LLM call failed")


# -----------------------------
# MIDI parsing -> evidence
# -----------------------------

def time_signature(pm: pretty_midi.PrettyMIDI) -> Tuple[int, int]:
    if pm.time_signature_changes:
        ts = pm.time_signature_changes[0]
        return int(ts.numerator), int(ts.denominator)
    return 4, 4


def tempo_at(pm: pretty_midi.PrettyMIDI, t: float) -> float:
    times, tempi = pm.get_tempo_changes()
    if len(tempi) == 0:
        return 120.0
    i = bisect_right(list(times), t) - 1
    if i < 0:
        return float(tempi[0])
    return float(tempi[i])


def pick_main_instruments(pm: pretty_midi.PrettyMIDI, keep: int = 2, include_drums: bool = False) -> List[pretty_midi.Instrument]:
    insts = []
    for inst in pm.instruments:
        if inst.is_drum and not include_drums:
            continue
        insts.append((len(inst.notes), inst))
    insts.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in insts[:keep]]


def infer_triad(pitch_classes: List[int]) -> Optional[str]:
    pcs = set(pitch_classes)
    if not pcs:
        return None
    best = None
    best_score = 0
    for root in range(12):
        maj = {root, (root + 4) % 12, (root + 7) % 12}
        minr = {root, (root + 3) % 12, (root + 7) % 12}
        maj_score = len(pcs & maj)
        min_score = len(pcs & minr)
        if maj_score > best_score and maj_score >= 2:
            best_score = maj_score
            best = (root, "maj")
        if min_score > best_score and min_score >= 2:
            best_score = min_score
            best = (root, "min")
    if not best:
        return None
    root, q = best
    return f"{NOTE_NAMES[root]}:{q}"


def guess_key_from_pitch_classes(pc_counts: List[int]) -> Dict[str, Any]:
    total = sum(pc_counts) or 1
    best = None
    best_score = -1.0
    for root in range(12):
        maj_scale = {(root + x) % 12 for x in MAJOR_SCALE}
        min_scale = {(root + x) % 12 for x in MINOR_SCALE}
        maj_cov = sum(pc_counts[i] for i in maj_scale) / total
        min_cov = sum(pc_counts[i] for i in min_scale) / total
        if maj_cov > best_score:
            best_score = maj_cov
            best = (root, "maj", maj_cov)
        if min_cov > best_score:
            best_score = min_cov
            best = (root, "min", min_cov)
    root, mode, cov = best
    return {
        "key_guess": f"{NOTE_NAMES[root]}:{mode}",
        "in_scale_ratio": float(cov),
        "out_of_scale_ratio": float(1.0 - float(cov)),
    }


def midi_global_summary(pm: pretty_midi.PrettyMIDI, midi_path: str) -> Dict[str, Any]:
    end_t = float(pm.get_end_time())
    times, tempi = pm.get_tempo_changes()
    tempo0 = float(tempi[0]) if len(tempi) else None
    num, den = time_signature(pm)

    notes = []
    inst_count = 0
    drum_count = 0
    for inst in pm.instruments:
        inst_count += 1
        if inst.is_drum:
            drum_count += 1
        notes.extend(inst.notes)

    note_count = len(notes)
    if note_count:
        pitch_min = int(min(n.pitch for n in notes))
        pitch_max = int(max(n.pitch for n in notes))
        vel_avg = float(sum(n.velocity for n in notes) / note_count)
    else:
        pitch_min = pitch_max = None
        vel_avg = None

    density = float(note_count / end_t) if end_t > 0 else None

    return {
        "midi_path": midi_path,
        "duration_s": end_t,
        "tempo_bpm_first": tempo0,
        "time_signature": f"{num}/{den}",
        "instrument_count": inst_count,
        "drum_instrument_count": drum_count,
        "note_count": note_count,
        "note_density": density,
        "pitch_min": pitch_min,
        "pitch_max": pitch_max,
        "velocity_avg": vel_avg,
    }


def midi_symbolic_excerpt(
    pm: pretty_midi.PrettyMIDI,
    max_measures: int = 12,
    steps_per_beat: int = 4,
    keep_top_instruments: int = 2,
    include_drums: bool = False,
    max_events_total: int = 400,
) -> Dict[str, Any]:
    beats = list(pm.get_beats())
    end_t = float(pm.get_end_time())
    num, den = time_signature(pm)
    beats_per_measure = num

    measure_starts = [beats[i] for i in range(0, len(beats), beats_per_measure)]
    if not measure_starts:
        measure_starts = [0.0]

    mcount = min(max_measures, len(measure_starts))
    last_end = measure_starts[mcount] if mcount < len(measure_starts) else end_t

    insts = pick_main_instruments(pm, keep=keep_top_instruments, include_drums=include_drums)

    measures: Dict[int, Dict[str, Any]] = {m: {"m": m, "events": []} for m in range(1, mcount + 1)}
    events_total = 0

    for inst in insts:
        for n in inst.notes:
            if n.start >= last_end:
                continue
            if n.end <= 0:
                continue

            mi = bisect_right(measure_starts[:mcount], n.start) - 1
            mi = max(0, min(mi, mcount - 1))
            m = mi + 1

            bi = bisect_right(beats, n.start) - 1
            if bi < 0:
                bi = 0

            beat_in_measure = max(0, bi - mi * beats_per_measure)
            frac = 0.0
            if bi + 1 < len(beats):
                span = beats[bi + 1] - beats[bi]
                if span > 1e-9:
                    frac = (n.start - beats[bi]) / span
            beat_pos = float(beat_in_measure) + frac

            tempo = tempo_at(pm, n.start)
            sec_per_beat = 60.0 / max(tempo, 1e-6)
            dur_beats = max(0.0, (n.end - n.start) / sec_per_beat)

            s = int(round(beat_pos * steps_per_beat))
            d = max(1, int(round(dur_beats * steps_per_beat)))
            p = int(n.pitch)
            name = pretty_midi.note_number_to_name(p)

            measures[m]["events"].append(
                {"s": s, "d": d, "p": p, "n": name, "v": int(n.velocity), "prog": int(inst.program), "drum": bool(inst.is_drum)}
            )

            events_total += 1
            if events_total >= max_events_total:
                break
        if events_total >= max_events_total:
            break

    out_measures = []
    pc_counts = [0] * 12

    for m in range(1, mcount + 1):
        evs = measures[m]["events"]
        evs.sort(key=lambda x: (x["s"], x["p"]))
        evs = evs[:80]

        pcs = []
        onset_bins: Dict[str, int] = {}
        for e in evs:
            onset_bins[str(e["s"])] = onset_bins.get(str(e["s"]), 0) + 1
            if not e.get("drum"):
                pc = int(e["p"]) % 12
                pcs.append(pc)
                pc_counts[pc] += 1

        chord = infer_triad(pcs)
        pr = [min(e["p"] for e in evs), max(e["p"] for e in evs)] if evs else None

        out_measures.append(
            {"m": m, "chord": chord, "pitch_range": pr, "event_count": len(evs), "onset_bins": onset_bins, "events": evs}
        )

    key_hint = guess_key_from_pitch_classes(pc_counts)
    times, tempi = pm.get_tempo_changes()
    tempo0 = float(tempi[0]) if len(tempi) else None

    return {
        "time_signature": f"{num}/{den}",
        "tempo_bpm_first": tempo0,
        "steps_per_beat": steps_per_beat,
        "max_measures": mcount,
        "key_hint": key_hint,
        "note": "events: s=start_step_in_measure, d=duration_steps, steps_per_beat=4 -> 1 step ~ 1/16 note (approx)",
        "measures": out_measures,
    }


# -----------------------------
# Rule metrics + rule edit plan
# -----------------------------

def cosine_sim(a: List[float], b: List[float]) -> float:
    import math
    da = sum(x*x for x in a) ** 0.5
    db = sum(x*x for x in b) ** 0.5
    if da == 0 or db == 0:
        return 0.0
    return sum(x*y for x, y in zip(a, b)) / (da * db)


def compute_rule_metrics(pm: pretty_midi.PrettyMIDI, symbolic: Dict[str, Any]) -> Dict[str, Any]:
    insts = [(len(i.notes), i) for i in pm.instruments if not i.is_drum]
    insts.sort(key=lambda x: x[0], reverse=True)
    melody_notes = sorted(insts[0][1].notes, key=lambda n: (n.start, n.pitch)) if insts else []

    intervals = []
    for i in range(1, len(melody_notes)):
        intervals.append(abs(int(melody_notes[i].pitch) - int(melody_notes[i-1].pitch)))
    leap_rate = float(sum(1 for x in intervals if x >= 8) / max(1, len(intervals)))

    measures = symbolic.get("measures", [])
    event_counts = [m.get("event_count", 0) for m in measures]
    avg_events = float(sum(event_counts) / max(1, len(event_counts)))
    max_events = int(max(event_counts)) if event_counts else 0

    chords = [m.get("chord") for m in measures if m.get("chord")]
    chord_consistency = 0.0
    chord_mode = None
    if chords:
        from collections import Counter
        c = Counter(chords)
        chord_mode, cnt = c.most_common(1)[0]
        chord_consistency = float(cnt / max(1, len(chords)))

    key_hint = symbolic.get("key_hint") or {}
    out_of_scale_ratio = float(key_hint.get("out_of_scale_ratio", 0.0))

    # repetition: measure-level pitch-class histogram similarity
    hists: List[List[float]] = []
    for m in measures:
        pc = [0.0] * 12
        for e in m.get("events", []):
            if not e.get("drum"):
                pc[int(e["p"]) % 12] += 1.0
        hists.append(pc)

    rep_pairs = []
    for i in range(len(hists)):
        for j in range(i+1, len(hists)):
            rep_pairs.append(cosine_sim(hists[i], hists[j]))
    rep_max = float(max(rep_pairs)) if rep_pairs else 0.0
    rep_high = float(sum(1 for x in rep_pairs if x >= 0.92) / max(1, len(rep_pairs))) if rep_pairs else 0.0

    rigidity = 0.0
    if measures:
        conc = []
        for m in measures:
            bins = m.get("onset_bins", {})
            total = sum(bins.values()) or 1
            top = max(bins.values()) if bins else 0
            conc.append(top / total)
        rigidity = float(sum(conc) / max(1, len(conc)))

    score = 100.0
    penalties: List[Dict[str, Any]] = []

    if leap_rate > 0.45:
        p = 18.0
        score -= p
        penalties.append({"metric": "leap_rate", "value": leap_rate, "penalty": p, "hint": "大跳偏多；可在关键处做级进填充或改变旋律走向。"})
    elif leap_rate > 0.30:
        p = 10.0
        score -= p
        penalties.append({"metric": "leap_rate", "value": leap_rate, "penalty": p, "hint": "大跳偏多；可增加级进连接。"})

    if avg_events > 45:
        p = 15.0
        score -= p
        penalties.append({"metric": "avg_events_per_measure", "value": avg_events, "penalty": p, "hint": "单位小节事件过密；可删弱拍装饰音或降低和弦堆叠。"})
    elif avg_events < 6:
        p = 10.0
        score -= p
        penalties.append({"metric": "avg_events_per_measure", "value": avg_events, "penalty": p, "hint": "单位小节事件过稀；可增加节奏型或伴奏层。"})

    if chord_consistency < 0.35 and len(chords) >= 4:
        p = 10.0
        score -= p
        penalties.append({"metric": "chord_consistency", "value": chord_consistency, "penalty": p, "hint": "和弦线索不稳定；可先确定主进行，再做变化。"})

    if out_of_scale_ratio > 0.45:
        p = 15.0
        score -= p
        penalties.append({"metric": "out_of_scale_ratio", "value": out_of_scale_ratio, "penalty": p, "hint": "离调音比例高；建议明确调性或让离调音有功能性。"})
    elif out_of_scale_ratio > 0.30:
        p = 8.0
        score -= p
        penalties.append({"metric": "out_of_scale_ratio", "value": out_of_scale_ratio, "penalty": p, "hint": "离调音偏多；建议限制在过渡处或做和声解释。"})

    if rep_max > 0.97 and rep_high > 0.25:
        p = 10.0
        score -= p
        penalties.append({"metric": "repetition_similarity", "value": {"rep_max": rep_max, "rep_high_ratio": rep_high}, "penalty": p, "hint": "重复过强；可做节奏/转位/移调变体。"})

    if rigidity > 0.70:
        p = 8.0
        score -= p
        penalties.append({"metric": "rhythm_rigidity", "value": rigidity, "penalty": p, "hint": "起音过度集中；可加入切分/弱拍变化。"})

    score = float(max(0.0, min(100.0, score)))
    return {
        "rule_score_total": score,
        "leap_rate": leap_rate,
        "avg_events_per_measure": avg_events,
        "max_events_per_measure": max_events,
        "key_hint": key_hint,
        "chord_mode": chord_mode,
        "chord_consistency": chord_consistency,
        "repetition": {"rep_max": rep_max, "rep_high_ratio": rep_high},
        "rhythm_rigidity": rigidity,
        "penalties": penalties,
    }


def rule_based_edit_plan(rule_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []
    for p in rule_metrics.get("penalties", []):
        metric = p.get("metric")
        if metric == "leap_rate":
            plan.append({"measure": None, "action": "smooth_leaps", "details": "降低连续大跳；在大跳前后增加级进连接。", "expected_effect": "旋律更连贯。"})
        elif metric == "avg_events_per_measure":
            v = p.get("value", 0)
            if isinstance(v, (int, float)) and v > 20:
                plan.append({"measure": None, "action": "reduce_density", "details": "删弱拍装饰音/重复音；限制堆叠。", "expected_effect": "织体更清晰。"})
            else:
                plan.append({"measure": None, "action": "add_rhythm_layer", "details": "补充简单伴奏/节奏型。", "expected_effect": "更有推动感。"})
        elif metric == "out_of_scale_ratio":
            plan.append({"measure": None, "action": "clarify_key", "details": "明确调性；离调音用经过/倚音或用和弦解释。", "expected_effect": "减少突兀感。"})
        elif metric == "chord_consistency":
            plan.append({"measure": None, "action": "stabilize_progression", "details": "先确定主和声进行（4-8小节），再做变化。", "expected_effect": "段落更统一。"})
        elif metric == "repetition_similarity":
            plan.append({"measure": None, "action": "vary_motifs", "details": "对重复处做变奏：节奏变体、转位、移调。", "expected_effect": "避免机械重复。"})
        elif metric == "rhythm_rigidity":
            plan.append({"measure": None, "action": "add_syncopation", "details": "加入切分/弱拍强调。", "expected_effect": "律动更灵活。"})
    # de-dup by action
    seen = set()
    out = []
    for it in plan:
        a = it.get("action")
        if a in seen:
            continue
        seen.add(a)
        out.append(it)
    return out[:10]


# -----------------------------
# Auto-edit (safe)
# -----------------------------

@dataclass
class EditOptions:
    quantize: bool = True
    denoise: bool = True
    vel_smooth: bool = True
    density: bool = True
    steps_per_beat: int = 4
    min_note_dur_s: float = 0.04
    min_velocity: int = 12
    max_notes_per_step: int = 4


def quantize_time_to_grid(beats: List[float], t: float, steps_per_beat: int) -> float:
    if not beats:
        return t
    i = bisect_right(beats, t) - 1
    i = max(0, min(i, len(beats) - 2))
    span = beats[i+1] - beats[i]
    if span <= 1e-9:
        return t
    frac = (t - beats[i]) / span
    beat_pos = i + frac
    step_pos = round(beat_pos * steps_per_beat)
    beat_pos_q = step_pos / steps_per_beat
    i2 = int(beat_pos_q)
    frac2 = beat_pos_q - i2
    i2 = max(0, min(i2, len(beats) - 2))
    return beats[i2] + frac2 * (beats[i2+1] - beats[i2])


def apply_auto_edit(pm: pretty_midi.PrettyMIDI, opts: EditOptions) -> pretty_midi.PrettyMIDI:
    beats = list(pm.get_beats())
    edited = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    edited.time_signature_changes = pm.time_signature_changes
    edited.key_signature_changes = pm.key_signature_changes

    for inst in pm.instruments:
        new_inst = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
        notes = sorted(inst.notes, key=lambda n: (n.start, n.pitch))
        q_notes: List[pretty_midi.Note] = []

        for n in notes:
            st, en = float(n.start), float(n.end)
            v = int(n.velocity)

            if opts.denoise:
                if (en - st) < opts.min_note_dur_s:
                    continue
                if v < opts.min_velocity:
                    continue

            if opts.quantize and beats:
                stq = quantize_time_to_grid(beats, st, opts.steps_per_beat)
                enq = quantize_time_to_grid(beats, en, opts.steps_per_beat)
                if enq <= stq:
                    enq = stq + max(0.03, (en - st))
                st, en = stq, enq

            q_notes.append(pretty_midi.Note(velocity=v, pitch=int(n.pitch), start=st, end=en))

        if opts.vel_smooth and q_notes:
            w = 3
            vels = [nn.velocity for nn in q_notes]
            sm = []
            for i in range(len(vels)):
                lo = max(0, i - w)
                hi = min(len(vels), i + w + 1)
                avg = sum(vels[lo:hi]) / (hi - lo)
                sm.append(int(round(avg)))
            for nn, nv in zip(q_notes, sm):
                nn.velocity = max(25, min(110, nv))

        if opts.density and beats and q_notes:
            bucket: Dict[Tuple[int, int], List[pretty_midi.Note]] = {}
            for nn in q_notes:
                st = nn.start
                i = bisect_right(beats, st) - 1
                i = max(0, min(i, len(beats) - 2))
                span = beats[i+1] - beats[i]
                frac = (st - beats[i]) / span if span > 1e-9 else 0.0
                step = int(round(frac * opts.steps_per_beat))
                step = max(0, min(opts.steps_per_beat - 1, step))
                key = (i, step)
                bucket.setdefault(key, []).append(nn)

            kept = []
            for _, lst in bucket.items():
                lst.sort(key=lambda n: n.velocity, reverse=True)
                kept.extend(lst[:opts.max_notes_per_step])
            q_notes = sorted(kept, key=lambda n: (n.start, n.pitch))

        new_inst.notes = q_notes
        edited.instruments.append(new_inst)

    return edited


# -----------------------------
# Multi-agent (Round1 -> Moderator -> Round2 -> Chair)
# -----------------------------

def compress_peer(j: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "judge_name": j.get("judge_name"),
        "focus": j.get("focus"),
        "score": j.get("score"),
        "pros": (j.get("pros") or [])[:4],
        "cons": (j.get("cons") or [])[:4],
        "suggestions": (j.get("suggestions") or [])[:4],
        "rationale": (j.get("rationale") or "")[:240],
    }


def run_deliberation(
    cache_dir: Path,
    midi_hash: str,
    cfg_hash: str,
    global_meta: Dict[str, Any],
    symbolic: Dict[str, Any],
    rule_metrics: Dict[str, Any],
    target_style: str,
    intended_use: str,
    model_judge: str,
    model_chair: str,
) -> Dict[str, Any]:
    client = llm_client()

    ctx = (
        f"目标风格：{target_style}\n"
        f"使用场景：{intended_use}\n"
        "请按目标风格/场景调整尺度：背景音乐更重视一致性与可听性，练习曲更重视规范与结构。"
    )

    global_s = json.dumps(global_meta, ensure_ascii=False)
    sym_s = json.dumps(symbolic, ensure_ascii=False)
    rule_s = json.dumps(rule_metrics, ensure_ascii=False)

    # ---- Round 1 (4 judges) ----
    judge_system = (
        "你是音乐评审。你必须只输出一个 JSON 对象（json）。\n"
        "评分范围 0-25（整数）。请尽量引用具体小节编号（第X小节）。\n"
        "schema: {"
        "\"judge_name\": str, \"focus\": str, \"score\": int, "
        "\"pros\": [str], \"cons\": [str], \"suggestions\": [str], \"rationale\": str"
        "}"
    )

    round1: Dict[str, Any] = {}
    for j in JUDGES:
        stage = f"r1_{j['name']}"
        cp = stage_cache_path(cache_dir, midi_hash, cfg_hash, stage)
        user = (
            f"{ctx}\n\n"
            f"评审维度：{j['focus']}\n\n"
            f"全局摘要(JSON)：{global_s}\n\n"
            f"规则指标(JSON)：{rule_s}\n\n"
            f"符号事件摘要(JSON)：{sym_s}\n\n"
            "按 schema 输出严格 JSON。"
        )
        out = call_llm_json_stage(client, model_judge, judge_system, user, cp, max_tokens=900, max_try=5)
        out["judge_name"] = j["name"]
        out["focus"] = j["focus"]
        out["score"] = clamp_int(out.get("score", 0), 0, 25)
        round1[j["name"]] = out

    # ---- Moderator ----
    mod_stage = "moderator"
    cp_mod = stage_cache_path(cache_dir, midi_hash, cfg_hash, mod_stage)
    mod_system = (
        "你是评审团主持人（moderator）。你必须只输出一个 JSON 对象（json）。\n"
        "目标：找出评委之间的主要分歧点，并提出需要在复议中解决的问题。\n"
        "schema: {"
        "\"disagreements\": [{\"topic\": str, \"why\": str, \"measures\": [int], \"questions\": [str]}],"
        "\"consensus_points\": [str]"
        "}"
    )
    scores = [int(v.get("score", 0)) for v in round1.values()]
    var = 0.0
    if scores:
        mean = sum(scores) / len(scores)
        var = sum((x - mean) ** 2 for x in scores) / len(scores)

    mod_user = (
        f"评委初评(JSON)：{json.dumps({k: compress_peer(v) for k,v in round1.items()}, ensure_ascii=False)}\n\n"
        f"分数方差提示：{var:.3f}\n\n"
        "输出 disagreements 至少 2 条；优先引用具体小节（无法确定则 measures=[]）。"
    )
    moderator = call_llm_json_stage(client, model_judge, mod_system, mod_user, cp_mod, max_tokens=800, max_try=5)
    moderator.setdefault("disagreements", [])
    moderator.setdefault("consensus_points", [])

    # ---- Round 2 (revise) ----
    revise_system = (
        "你是音乐评审。你已完成初评，现在看到了其他评委摘要 + 主持人的分歧清单。\n"
        "请复议：必要时修正分数与建议，使其更一致、更可执行。\n"
        "你必须只输出一个 JSON 对象（json）。schema 同初评。"
    )

    round2: Dict[str, Any] = {}
    peers_comp = {k: compress_peer(v) for k, v in round1.items()}
    for j in JUDGES:
        stage = f"r2_{j['name']}"
        cp = stage_cache_path(cache_dir, midi_hash, cfg_hash, stage)
        me = round1[j["name"]]
        others = [peers_comp[k] for k in peers_comp if k != j["name"]]
        user = (
            f"{ctx}\n\n"
            f"你的评审维度：{j['focus']}\n\n"
            f"全局摘要(JSON)：{global_s}\n\n"
            f"规则指标(JSON)：{rule_s}\n\n"
            f"符号事件摘要(JSON)：{sym_s}\n\n"
            f"你的初评(JSON)：{json.dumps(compress_peer(me), ensure_ascii=False)}\n\n"
            f"其他评委摘要(JSON 数组)：{json.dumps(others, ensure_ascii=False)}\n\n"
            f"主持人分歧清单(JSON)：{json.dumps(moderator, ensure_ascii=False)}\n\n"
            "按 schema 输出严格 JSON。"
        )
        out = call_llm_json_stage(client, model_judge, revise_system, user, cp, max_tokens=950, max_try=5)
        out["judge_name"] = j["name"]
        out["focus"] = j["focus"]
        out["score"] = clamp_int(out.get("score", me.get("score", 0)), 0, 25)
        round2[j["name"]] = out

    # ---- Chair ----
    chair_stage = "chair"
    cp_chair = stage_cache_path(cache_dir, midi_hash, cfg_hash, chair_stage)

    harmony = int(round2["harmony_judge"]["score"])
    rhythm = int(round2["rhythm_judge"]["score"])
    structure = int(round2["structure_judge"]["score"])
    style = int(round2["style_judge"]["score"])

    chair_system = (
        "你是总评/仲裁。你必须只输出一个 JSON 对象（json），不能输出其他文本。\n"
        "分数范围：各维度 0-25，总分 0-100（整数）。请尽量引用具体小节编号。\n"
        "schema: {"
        "\"score_total\": int,"
        "\"scores\": {\"harmony\": int, \"rhythm\": int, \"structure\": int, \"style\": int},"
        "\"pros\": [str], \"cons\": [str], \"suggestions\": [str],"
        "\"disagreements\": [{\"topic\": str, \"why\": str, \"measures\": [int], \"resolution\": str}],"
        "\"edit_plan\": [{\"measure\": int|null, \"action\": str, \"details\": str, \"expected_effect\": str}]"
        "}\n"
        "输出要求：pros/cons/suggestions 各**至少**5条（不能为空）；edit_plan **至少**6条、最多8条；disagreements 至少1条（若无显著分歧，写一条 topic='无显著分歧'）。"
    )

    chair_user = (
        f"{ctx}\n\n"
        f"全局摘要(JSON)：{global_s}\n\n"
        f"规则指标(JSON)：{rule_s}\n\n"
        f"符号事件摘要(JSON)：{sym_s}\n\n"
        f"主持人分歧清单(JSON)：{json.dumps(moderator, ensure_ascii=False)}\n\n"
        f"评委最终输出（Round2）：{json.dumps({k: compress_peer(v) for k,v in round2.items()}, ensure_ascii=False)}\n\n"
        "请给出最终结论（按 schema 输出严格 JSON）。"
    )

    chair = call_llm_json_stage(client, model_chair, chair_system, chair_user, cp_chair, max_tokens=950, max_try=6)
    chair.setdefault("scores", {"harmony": harmony, "rhythm": rhythm, "structure": structure, "style": style})
    chair["scores"]["harmony"] = clamp_int(chair["scores"].get("harmony", harmony), 0, 25, harmony)
    chair["scores"]["rhythm"] = clamp_int(chair["scores"].get("rhythm", rhythm), 0, 25, rhythm)
    chair["scores"]["structure"] = clamp_int(chair["scores"].get("structure", structure), 0, 25, structure)
    chair["scores"]["style"] = clamp_int(chair["scores"].get("style", style), 0, 25, style)
    chair["score_total"] = clamp_int(chair.get("score_total", harmony+rhythm+structure+style), 0, 100, harmony+rhythm+structure+style)
    chair.setdefault("pros", [])
    chair.setdefault("cons", [])
    chair.setdefault("suggestions", [])
    chair.setdefault("disagreements", [])
    chair.setdefault("edit_plan", [])

    return {
        "round1": round1,
        "moderator": moderator,
        "round2": round2,
        "chair": chair,
    }


# -----------------------------
# Main v4 single (with after stage cache)
# -----------------------------


def apply_text_fallback(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Guarantee non-empty pros/cons/suggestions/edit_plan/disagreements even if LLM returns empty lists.
    Only fills when missing/empty.
    """
    def _ensure_list(key: str, want_n: int, base: List[str]) -> None:
        lst = report.get(key) or []
        if not isinstance(lst, list):
            lst = []
        for x in base:
            if len(lst) >= want_n:
                break
            if x and x not in lst:
                lst.append(x)
        while len(lst) < want_n:
            lst.append(f"{key}:???????????????")
        report[key] = lst

    rm = report.get("rule_metrics") or {}
    pen = rm.get("penalties") or []
    pen_hints = [p.get("hint") for p in pen if isinstance(p, dict) and p.get("hint")]

    cons_base = list(pen_hints)
    sug_base = ["???" + h for h in pen_hints]

    # also use edit_plan as suggestion seeds
    ep0 = report.get("edit_plan") or []
    if isinstance(ep0, list):
        for it in ep0:
            if isinstance(it, dict):
                a = str(it.get("action","")).strip()
                d = str(it.get("details","")).strip()
                if a:
                    sug_base.append(f"?? {a}?{d}".strip())

    pros_base = []
    kh = rm.get("key_hint") or {}
    try:
        oos = float(kh.get("out_of_scale_ratio", 0.0) or 0.0)
    except Exception:
        oos = 0.0
    if oos <= 0.30:
        pros_base.append("??/????????????????")

    try:
        cc = float(rm.get("chord_consistency", 0.0) or 0.0)
    except Exception:
        cc = 0.0
    if cc >= 0.50:
        pros_base.append("????????????????????")

    try:
        ae = float(rm.get("avg_events_per_measure", 0.0) or 0.0)
    except Exception:
        ae = 0.0
    if 6 <= ae <= 45:
        pros_base.append("????????????????????????")

    try:
        lr = float(rm.get("leap_rate", 0.0) or 0.0)
    except Exception:
        lr = 0.0
    if lr <= 0.30:
        pros_base.append("???????????????????")

    pros_base.extend([
        "???????????????????????????",
        "????????????????????????",
    ])

    _ensure_list("pros", 5, pros_base)
    _ensure_list("cons", 5, cons_base)
    _ensure_list("suggestions", 5, sug_base)

    # ensure edit_plan 6~8
    ep = report.get("edit_plan") or []
    if not isinstance(ep, list):
        ep = []
    generic_plan = [
        {"measure": None, "action": "reduce_density", "details": "??????/?????????", "expected_effect": "??????"},
        {"measure": None, "action": "smooth_leaps", "details": "????????????????", "expected_effect": "??????"},
        {"measure": None, "action": "clarify_key", "details": "???????????????", "expected_effect": "??????"},
        {"measure": None, "action": "stabilize_progression", "details": "?????????4-8????", "expected_effect": "??????"},
        {"measure": None, "action": "add_syncopation", "details": "????/?????????", "expected_effect": "??????"},
        {"measure": None, "action": "vary_motifs", "details": "???????/??/?????", "expected_effect": "???????"},
    ]
    seen = set()
    for it in ep:
        if isinstance(it, dict):
            seen.add((str(it.get("action","")), str(it.get("measure",None))))
    for it in generic_plan:
        k = (str(it.get("action","")), str(it.get("measure",None)))
        if k not in seen and len(ep) < 8:
            ep.append(it)
            seen.add(k)
    report["edit_plan"] = ep[:8]

    # ensure disagreements at least 1
    dis = report.get("disagreements") or []
    if not isinstance(dis, list) or len(dis) == 0:
        report["disagreements"] = [{
            "topic": "?????",
            "why": "????????????????????",
            "measures": [],
            "resolution": "?????????????????"
        }]

    return report

def evaluate_v4_single(
    midi_path: Path,
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
    midi_bytes = midi_path.read_bytes()
    midi_hash = sha256_bytes(midi_bytes)

    run_cfg = {
        "model_judge": model_judge,
        "model_chair": model_chair,
        "max_measures": max_measures,
        "target_style": target_style,
        "intended_use": intended_use,
        "rule_weight": rule_weight,
        "auto_edit": auto_edit,
        "do_after": do_after,
        "version": "v4_single_segmented_cache",
    }
    cfg_hash = sha256_bytes(stable_json(run_cfg).encode("utf-8"))[:16]

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    global_meta = midi_global_summary(pm, str(midi_path))
    symbolic = midi_symbolic_excerpt(pm, max_measures=max_measures)
    rule_metrics = compute_rule_metrics(pm, symbolic)

    llm_pack = run_deliberation(
        cache_dir=cache_dir,
        midi_hash=midi_hash,
        cfg_hash=cfg_hash,
        global_meta=global_meta,
        symbolic=symbolic,
        rule_metrics=rule_metrics,
        target_style=target_style,
        intended_use=intended_use,
        model_judge=model_judge,
        model_chair=model_chair,
    )

    chair = llm_pack["chair"]
    llm_total = int(chair.get("score_total", 0))
    rule_total = float(rule_metrics.get("rule_score_total", 0.0))
    rule_weight = max(0.0, min(1.0, float(rule_weight)))
    final_total = float((1.0 - rule_weight) * llm_total + rule_weight * rule_total)
    final_total_i = clamp_int(round(final_total), 0, 100, llm_total)

    chair_plan = chair.get("edit_plan") if isinstance(chair.get("edit_plan"), list) else []
    merged_plan = chair_plan + rule_based_edit_plan(rule_metrics)
    # de-dup by (measure, action)
    seen = set()
    plan_out = []
    for it in merged_plan:
        if not isinstance(it, dict):
            continue
        m = it.get("measure", None)
        a = str(it.get("action", "")).strip()
        if not a:
            continue
        key = (m, a)
        if key in seen:
            continue
        seen.add(key)
        it.setdefault("details", "")
        it.setdefault("expected_effect", "")
        plan_out.append(it)
        if len(plan_out) >= 16:
            break

    report = {
        "version": "music_eval_v4_single_segmented_cache",
        "ts": now_ts(),
        "file": str(midi_path.resolve()),
        "run_cfg": run_cfg,
        "cache_key": f"{midi_hash}_{cfg_hash}",

        "midi_meta": global_meta,
        "symbolic_excerpt": symbolic,
        "rule_metrics": rule_metrics,

        "scores": chair.get("scores", {}),
        "score_total": llm_total,
        "rule_score_total": rule_total,
        "final_score_total": final_total_i,
        "rule_weight": rule_weight,

        "pros": chair.get("pros", []),
        "cons": chair.get("cons", []),
        "suggestions": chair.get("suggestions", []),
        "disagreements": chair.get("disagreements", []),
        "edit_plan": plan_out,

        "judge_traces_round1": llm_pack["round1"],
        "moderator": llm_pack["moderator"],
        "judge_traces": llm_pack["round2"],
    }

    # AFTER loop (stage cached under edited midi hash)
    if do_after:
        opts = EditOptions(
            quantize=("Q" in auto_edit),
            denoise=("N" in auto_edit),
            vel_smooth=("V" in auto_edit),
            density=("D" in auto_edit),
        )
        edited_pm = apply_auto_edit(pretty_midi.PrettyMIDI(str(midi_path)), opts)

        edited_path = out_path.parent / f"{midi_path.stem}.edited.mid"
        edited_pm.write(str(edited_path))

        edited_bytes = edited_path.read_bytes()
        edited_hash = sha256_bytes(edited_bytes)

        after_cfg = dict(run_cfg)
        after_cfg["do_after"] = False
        after_cfg_hash = sha256_bytes(stable_json(after_cfg).encode("utf-8"))[:16]

        pm_a = pretty_midi.PrettyMIDI(str(edited_path))
        global_a = midi_global_summary(pm_a, str(edited_path))
        symbolic_a = midi_symbolic_excerpt(pm_a, max_measures=max_measures)
        rule_a = compute_rule_metrics(pm_a, symbolic_a)

        llm_after = run_deliberation(
            cache_dir=cache_dir,
            midi_hash=edited_hash,
            cfg_hash=after_cfg_hash,
            global_meta=global_a,
            symbolic=symbolic_a,
            rule_metrics=rule_a,
            target_style=target_style,
            intended_use=intended_use,
            model_judge=model_judge,
            model_chair=model_chair,
        )

        chair_a = llm_after["chair"]
        llm_total_a = int(chair_a.get("score_total", 0))
        rule_total_a = float(rule_a.get("rule_score_total", 0.0))
        final_a = float((1.0 - rule_weight) * llm_total_a + rule_weight * rule_total_a)
        final_a_i = clamp_int(round(final_a), 0, 100, llm_total_a)

        after_report = {
            "version": "music_eval_v4_after_segmented_cache",
            "ts": now_ts(),
            "file": str(edited_path.resolve()),
            "cache_key": f"{edited_hash}_{after_cfg_hash}",
            "midi_meta": global_a,
            "symbolic_excerpt": symbolic_a,
            "rule_metrics": rule_a,
            "scores": chair_a.get("scores", {}),
            "score_total": llm_total_a,
            "rule_score_total": rule_total_a,
            "final_score_total": final_a_i,
            "pros": chair_a.get("pros", []),
            "cons": chair_a.get("cons", []),
            "suggestions": chair_a.get("suggestions", []),
            "disagreements": chair_a.get("disagreements", []),
            "edit_plan": chair_a.get("edit_plan", []),
        }

        report["after"] = {
            "enabled": True,
            "auto_edit": auto_edit,
            "edited_midi": str(edited_path.resolve()),
            "after_report": after_report,
            "delta": {
                "final_score_total": final_a_i - report["final_score_total"],
                "llm_score_total": llm_total_a - report["score_total"],
                "rule_score_total": rule_total_a - report["rule_score_total"],
            },
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    
report = apply_text_fallback(report)
out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi", required=True)
    ap.add_argument("--out", default="report_v4_single.json")
    ap.add_argument("--cache_dir", default=".cache_music_eval_v4_stages")

    ap.add_argument("--model_judge", default=DEFAULT_MODEL)
    ap.add_argument("--model_chair", default=DEFAULT_MODEL)
    ap.add_argument("--max_measures", type=int, default=12)
    ap.add_argument("--target_style", default="general")
    ap.add_argument("--intended_use", default="demo")
    ap.add_argument("--rule_weight", type=float, default=0.25)
    ap.add_argument("--auto_edit", default="QNV")
    ap.add_argument("--after", action="store_true")
    args = ap.parse_args()

    if not os.getenv("DEEPSEEK_API_KEY"):
        raise SystemExit("Missing env var: DEEPSEEK_API_KEY")
    midi_path = Path(args.midi)
    out_path = Path(args.out)
    cache_dir = Path(args.cache_dir)

    rep = evaluate_v4_single(
        midi_path=midi_path,
        out_path=out_path,
        cache_dir=cache_dir,
        model_judge=args.model_judge,
        model_chair=args.model_chair,
        max_measures=args.max_measures,
        target_style=args.target_style,
        intended_use=args.intended_use,
        rule_weight=args.rule_weight,
        auto_edit=args.auto_edit,
        do_after=bool(args.after),
    )

    print("OK ->", out_path.resolve())
    print("final_score_total =", rep.get("final_score_total"))

if __name__ == "__main__":
    main()
