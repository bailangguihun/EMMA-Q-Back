import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from bisect import bisect_right

import pretty_midi
from openai import OpenAI


JUDGES = [
    {"name": "harmony_judge", "focus": "和声/乐理（基于小节内音高集合、和弦推断、声部进行）"},
    {"name": "rhythm_judge", "focus": "节奏/律动（基于拍网格、起音分布、密度与稳定性）"},
    {"name": "structure_judge", "focus": "结构/段落（基于小节间重复/对比、动机、密度变化）"},
    {"name": "style_judge", "focus": "风格一致性/可听性（基于音域、跳进、织体与一致性）"},
]


def _safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise ValueError("No JSON object found in model output.")
        return json.loads(m.group(0))


def _tempo_at(pm: pretty_midi.PrettyMIDI, t: float) -> float:
    # get_tempo_changes -> (times, tempi)
    tempo_times, tempi = pm.get_tempo_changes()
    if len(tempi) == 0:
        return 120.0
    i = bisect_right(list(tempo_times), t) - 1
    if i < 0:
        return float(tempi[0])
    return float(tempi[i])


def _time_signature(pm: pretty_midi.PrettyMIDI) -> Tuple[int, int]:
    # 只取第一段拍号（够用作评审输入；后续可扩展为拍号变化）
    if pm.time_signature_changes:
        ts = pm.time_signature_changes[0]
        return int(ts.numerator), int(ts.denominator)
    return 4, 4


def _infer_triad(pitch_classes: List[int]) -> Optional[str]:
    # 粗略三和弦推断：返回 "C:maj"/"A:min"/None
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
    name = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"][root]
    return f"{name}:{q}"


def midi_global_summary(midi_path: str) -> Dict[str, Any]:
    pm = pretty_midi.PrettyMIDI(midi_path)
    end_t = float(pm.get_end_time())
    tempo_times, tempi = pm.get_tempo_changes()
    tempo0 = float(tempi[0]) if len(tempi) else None

    notes = []
    inst_count = 0
    drum_inst_count = 0
    for inst in pm.instruments:
        inst_count += 1
        if inst.is_drum:
            drum_inst_count += 1
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

    num, den = _time_signature(pm)

    return {
        "midi_path": midi_path,
        "duration_s": end_t,
        "tempo_bpm_first": tempo0,
        "time_signature": f"{num}/{den}",
        "instrument_count": inst_count,
        "drum_instrument_count": drum_inst_count,
        "note_count": note_count,
        "note_density": density,
        "pitch_min": pitch_min,
        "pitch_max": pitch_max,
        "velocity_avg": vel_avg,
    }


def midi_symbolic_excerpt(
    midi_path: str,
    max_measures: int = 12,
    steps_per_beat: int = 4,
    keep_top_instruments: int = 2,
    include_drums: bool = False,
    max_events_total: int = 300,
) -> Dict[str, Any]:
    pm = pretty_midi.PrettyMIDI(midi_path)
    beats = list(pm.get_beats())
    end_t = float(pm.get_end_time())
    num, den = _time_signature(pm)
    beats_per_measure = num  # 以“拍”为单位的每小节长度（足够用于 4/4、3/4 等常见拍号）

    # 小节起点（按 beats 切）
    measure_starts = [beats[i] for i in range(0, len(beats), beats_per_measure)]
    if not measure_starts:
        measure_starts = [0.0]
    # 选前 N 小节
    mcount = min(max_measures, len(measure_starts))
    last_end = measure_starts[mcount] if mcount < len(measure_starts) else end_t

    # 选“最主要的音轨”（按 note 数）
    insts = []
    for inst in pm.instruments:
        if inst.is_drum and not include_drums:
            continue
        insts.append((len(inst.notes), inst))
    insts.sort(key=lambda x: x[0], reverse=True)
    insts = [x[1] for x in insts[:keep_top_instruments]] if insts else []

    measures: Dict[int, Dict[str, Any]] = {m: {"m": m, "events": []} for m in range(1, mcount + 1)}
    events_total = 0

    for inst in insts:
        for n in inst.notes:
            if n.start >= last_end:
                continue
            if n.end <= 0:
                continue

            # 找到所在小节
            mi = bisect_right(measure_starts[:mcount], n.start) - 1
            if mi < 0:
                mi = 0
            if mi >= mcount:
                continue
            m = mi + 1

            # beat index（用于小节内位置）
            bi = bisect_right(beats, n.start) - 1
            if bi < 0:
                bi = 0
            # 小节内第几拍（整数）+ 拍内小数
            beat_in_measure = (bi - mi * beats_per_measure)
            beat_in_measure = max(0, beat_in_measure)

            frac = 0.0
            if bi + 1 < len(beats):
                span = beats[bi + 1] - beats[bi]
                if span > 1e-9:
                    frac = (n.start - beats[bi]) / span
            beat_pos = float(beat_in_measure) + frac

            tempo = _tempo_at(pm, n.start)
            sec_per_beat = 60.0 / max(tempo, 1e-6)
            dur_beats = (n.end - n.start) / sec_per_beat
            dur_beats = max(0.0, dur_beats)

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

    # 每小节补充：和弦猜测、音域、事件数（并对事件数做限幅）
    out_measures = []
    for m in range(1, mcount + 1):
        evs = measures[m]["events"]
        evs.sort(key=lambda x: (x["s"], x["p"]))
        evs = evs[:60]  # 每小节最多 60 个事件，避免 token 爆炸
        pcs = [(e["p"] % 12) for e in evs if not e.get("drum")]
        chord = _infer_triad(pcs)
        if evs:
            pr = [min(e["p"] for e in evs), max(e["p"] for e in evs)]
        else:
            pr = None
        out_measures.append(
            {"m": m, "chord": chord, "pitch_range": pr, "event_count": len(evs), "events": evs}
        )

    tempo_times, tempi = pm.get_tempo_changes()
    tempo0 = float(tempi[0]) if len(tempi) else None

    return {
        "time_signature": f"{num}/{den}",
        "tempo_bpm_first": tempo0,
        "steps_per_beat": steps_per_beat,
        "max_measures": mcount,
        "note": "events: s=start_step_in_measure, d=duration_steps, steps_per_beat=4 means 1 step=1/16 note in 4/4",
        "measures": out_measures,
    }


def _call_llm_json(client: OpenAI, model: str, system: str, user: str, max_try: int = 2) -> Dict[str, Any]:
    last_err = None
    for _ in range(max_try):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=1600,
        )
        txt = resp.choices[0].message.content or ""
        try:
            return _safe_json_loads(txt)
        except Exception as e:
            last_err = e
            fix = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是 JSON 修复器。只输出可被 json.loads 解析的 JSON。"},
                    {"role": "user", "content": "把下面内容改写成严格 JSON（只输出 JSON）。\n\n" + txt},
                ],
                temperature=0.0,
                max_tokens=1600,
            )
            txt2 = fix.choices[0].message.content or ""
            try:
                return _safe_json_loads(txt2)
            except Exception as e2:
                last_err = e2
    raise last_err or RuntimeError("LLM JSON parse failed.")


def _compress_peer(j: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "judge_name": j.get("judge_name"),
        "focus": j.get("focus"),
        "score": j.get("score"),
        "pros": (j.get("pros") or [])[:4],
        "cons": (j.get("cons") or [])[:4],
        "suggestions": (j.get("suggestions") or [])[:4],
        "rationale": (j.get("rationale") or "")[:220],
    }


def evaluate_midi_deliberation_symbolic(midi_path: str, model_judge: str, model_chair: str, max_measures: int) -> Dict[str, Any]:
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    api_key = os.environ["DEEPSEEK_API_KEY"]
    client = OpenAI(api_key=api_key, base_url=base_url)

    global_meta = midi_global_summary(midi_path)
    symbolic = midi_symbolic_excerpt(midi_path, max_measures=max_measures)

    global_s = json.dumps(global_meta, ensure_ascii=False)
    sym_s = json.dumps(symbolic, ensure_ascii=False)

    judge_system = (
        "你是音乐评审。你必须只输出一个 JSON 对象，不能输出任何额外文本。\n"
        "评分范围 0-25（整数）。\n"
        "请在 rationale / suggestions 中尽量引用具体小节编号（例如“第3小节…”）。\n"
        "schema: {"
        "\"judge_name\": str, \"focus\": str, \"score\": int, "
        "\"pros\": [str], \"cons\": [str], \"suggestions\": [str], \"rationale\": str"
        "}"
    )

    # Round1
    round1: Dict[str, Any] = {}
    for j in JUDGES:
        user = (
            f"评审维度：{j['focus']}\n\n"
            f"全局摘要(JSON)：{global_s}\n\n"
            f"符号事件摘要(JSON, 前{max_measures}小节)：{sym_s}\n\n"
            "按 schema 输出严格 JSON。"
        )
        out = _call_llm_json(client, model_judge, judge_system, user, max_try=2)
        out["judge_name"] = j["name"]
        out["focus"] = j["focus"]
        out["score"] = max(0, min(25, int(out.get("score", 0))))
        round1[j["name"]] = out

    # Round2
    revise_system = (
        "你是音乐评审。你已完成初评，现在看到了其他评委观点摘要。\n"
        "请复议修正，使建议更一致、可执行，并尽量引用具体小节。\n"
        "你必须只输出一个 JSON 对象，不能输出任何额外文本。\n"
        "schema 同初评。"
    )

    round2: Dict[str, Any] = {}
    for j in JUDGES:
        me = round1[j["name"]]
        peers = []
        for oname, other in round1.items():
            if oname == j["name"]:
                continue
            peers.append(_compress_peer(other))

        user = (
            f"你的评审维度：{j['focus']}\n\n"
            f"全局摘要(JSON)：{global_s}\n\n"
            f"符号事件摘要(JSON, 前{max_measures}小节)：{sym_s}\n\n"
            f"你的初评(JSON)：{json.dumps(_compress_peer(me), ensure_ascii=False)}\n\n"
            f"其他评委摘要(JSON 数组)：{json.dumps(peers, ensure_ascii=False)}\n\n"
            "请复议后按 schema 输出严格 JSON。"
        )
        out = _call_llm_json(client, model_judge, revise_system, user, max_try=2)
        out["judge_name"] = j["name"]
        out["focus"] = j["focus"]
        out["score"] = max(0, min(25, int(out.get('score', me.get('score', 0)))))
        round2[j["name"]] = out

    harmony = int(round2["harmony_judge"]["score"])
    rhythm = int(round2["rhythm_judge"]["score"])
    structure = int(round2["structure_judge"]["score"])
    style = int(round2["style_judge"]["score"])
    score_total_guess = harmony + rhythm + structure + style

    chair_system = (
        "你是总评/仲裁。你必须只输出一个 JSON 对象，不能输出任何额外文本。\n"
        "schema: {"
        "\"score_total\": int, "
        "\"scores\": {\"harmony\": int, \"rhythm\": int, \"structure\": int, \"style\": int}, "
        "\"pros\": [str], \"cons\": [str], \"suggestions\": [str]"
        "}\n分数范围：各维度 0-25，总分 0-100（整数）。\n"
        "请在 pros/cons/suggestions 中尽量引用具体小节编号。"
    )

    chair_user = (
        f"全局摘要(JSON)：{global_s}\n\n"
        f"符号事件摘要(JSON, 前{max_measures}小节)：{sym_s}\n\n"
        f"评委最终输出（Round2）：{json.dumps(round2, ensure_ascii=False)}\n\n"
        "按 schema 输出严格 JSON。"
    )
    chair = _call_llm_json(client, model_chair, chair_system, chair_user, max_try=2)

    report = {
        "version": "music_eval_v3_symbolic_deliberation",
        "ts": int(time.time()),
        "midi_meta": global_meta,
        "symbolic_excerpt": symbolic,
        "scores": {"harmony": harmony, "rhythm": rhythm, "structure": structure, "style": style},
        "score_total": int(chair.get("score_total", score_total_guess)),
        "pros": chair.get("pros", []),
        "cons": chair.get("cons", []),
        "suggestions": chair.get("suggestions", []),
        "judge_traces_round1": round1,
        "judge_traces": round2,
    }
    report["score_total"] = max(0, min(100, int(report["score_total"])))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi", required=True)
    ap.add_argument("--out", default="report_symbolic.json")
    ap.add_argument("--model_judge", default="deepseek-chat")
    ap.add_argument("--model_chair", default="deepseek-chat")
    ap.add_argument("--max_measures", type=int, default=12)
    args = ap.parse_args()

    midi_path = str(Path(args.midi).resolve())
    out_path = Path(args.out)

    report = evaluate_midi_deliberation_symbolic(midi_path, args.model_judge, args.model_chair, args.max_measures)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OK ->", out_path.resolve())
    print("score_total =", report.get("score_total"))


if __name__ == "__main__":
    main()
