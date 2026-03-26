import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict

import pretty_midi
from openai import OpenAI

JUDGES = [
    {"name": "harmony_judge", "focus": "和声/乐理"},
    {"name": "rhythm_judge", "focus": "节奏/律动"},
    {"name": "structure_judge", "focus": "结构/段落"},
    {"name": "style_judge", "focus": "风格一致性/可听性"},
]

def _safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise ValueError("No JSON object found in model output.")
        return json.loads(m.group(0))

def _midi_summary(midi_path: str) -> Dict[str, Any]:
    pm = pretty_midi.PrettyMIDI(midi_path)
    end_t = float(pm.get_end_time())
    tempi, _ = pm.get_tempo_changes()
    tempo = float(tempi[0]) if len(tempi) else None

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
        onset = sorted(float(n.start) for n in notes)
        ioi = [onset[i + 1] - onset[i] for i in range(len(onset) - 1)]
        ioi_avg = float(sum(ioi) / len(ioi)) if ioi else None
    else:
        pitch_min = pitch_max = None
        vel_avg = None
        ioi_avg = None

    density = float(note_count / end_t) if end_t > 0 else None

    return {
        "midi_path": midi_path,
        "duration_s": end_t,
        "tempo_bpm": tempo,
        "instrument_count": inst_count,
        "drum_instrument_count": drum_inst_count,
        "note_count": note_count,
        "note_density": density,
        "pitch_min": pitch_min,
        "pitch_max": pitch_max,
        "velocity_avg": vel_avg,
        "ioi_avg_s": ioi_avg,
    }

def _call_llm_json(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.2,
    max_tokens: int = 1100,
    max_try: int = 2,
) -> Dict[str, Any]:
    last_err = None
    for _ in range(max_try):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content or ""
        try:
            return _safe_json_loads(text)
        except Exception as e:
            last_err = e
            fix = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是 JSON 修复器。只输出可被 json.loads 解析的 JSON。"},
                    {"role": "user", "content": "把下面内容改写成严格 JSON（只输出 JSON）。\n\n" + text},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            text2 = fix.choices[0].message.content or ""
            try:
                return _safe_json_loads(text2)
            except Exception as e2:
                last_err = e2
    raise last_err or RuntimeError("LLM JSON parse failed.")

def evaluate_midi(midi_path: str, model_judge: str, model_chair: str) -> Dict[str, Any]:
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    api_key = os.environ["DEEPSEEK_API_KEY"]
    client = OpenAI(api_key=api_key, base_url=base_url)

    meta = _midi_summary(midi_path)
    meta_s = json.dumps(meta, ensure_ascii=False)

    judge_system = (
        "你是音乐评审。你必须只输出一个 JSON 对象，不能输出任何额外文本。\n"
        "评分范围 0-25（整数）。\n"
        "schema: {"
        "\"judge_name\": str, \"focus\": str, \"score\": int, "
        "\"pros\": [str], \"cons\": [str], \"suggestions\": [str], \"rationale\": str"
        "}"
    )

    traces: Dict[str, Any] = {}
    score_map: Dict[str, int] = {}

    for j in JUDGES:
        user = f"评审维度：{j['focus']}\nMIDI 摘要(JSON)：{meta_s}\n\n按 schema 输出严格 JSON。"
        out = _call_llm_json(client, model_judge, judge_system, user, max_try=2)
        out["judge_name"] = j["name"]
        out["focus"] = j["focus"]
        sc = int(out.get("score", 0))
        sc = max(0, min(25, sc))
        out["score"] = sc
        traces[j["name"]] = out
        score_map[j["name"]] = sc

    harmony = score_map.get("harmony_judge", 0)
    rhythm = score_map.get("rhythm_judge", 0)
    structure = score_map.get("structure_judge", 0)
    style = score_map.get("style_judge", 0)
    score_total = harmony + rhythm + structure + style

    chair_system = (
        "你是总评/仲裁。你必须只输出一个 JSON 对象，不能输出任何额外文本。\n"
        "schema: {"
        "\"score_total\": int, "
        "\"scores\": {\"harmony\": int, \"rhythm\": int, \"structure\": int, \"style\": int}, "
        "\"pros\": [str], \"cons\": [str], \"suggestions\": [str]"
        "}\n分数范围：各维度 0-25，总分 0-100（整数）。"
    )

    chair_user = (
        f"MIDI 摘要：{meta_s}\n\n评委输出：{json.dumps(traces, ensure_ascii=False)}\n\n"
        "请汇总最终结论，pros/cons/suggestions 各给 3-8 条，尽量具体可执行。"
    )

    chair = _call_llm_json(client, model_chair, chair_system, chair_user, max_try=2)

    report = {
        "version": "music_eval_v1",
        "ts": int(time.time()),
        "midi_meta": meta,
        "scores": {"harmony": harmony, "rhythm": rhythm, "structure": structure, "style": style},
        "score_total": int(chair.get("score_total", score_total)),
        "pros": chair.get("pros", []),
        "cons": chair.get("cons", []),
        "suggestions": chair.get("suggestions", []),
        "judge_traces": traces,
    }
    report["score_total"] = max(0, min(100, int(report["score_total"])))
    return report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi", required=True, help="path to .mid/.midi")
    ap.add_argument("--out", default="report.json")
    ap.add_argument("--model_judge", default="deepseek-chat")
    ap.add_argument("--model_chair", default="deepseek-chat")
    args = ap.parse_args()

    midi_path = str(Path(args.midi).resolve())
    out_path = Path(args.out)

    report = evaluate_midi(midi_path, args.model_judge, args.model_chair)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OK ->", out_path.resolve())
    print("score_total =", report.get("score_total"))

if __name__ == "__main__":
    main()
