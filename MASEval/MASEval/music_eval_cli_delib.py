import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

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
    max_tokens: int = 1200,
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


def _compress_peer(j: Dict[str, Any]) -> Dict[str, Any]:
    # 给评委看的“他人观点摘要”，控制长度
    return {
        "judge_name": j.get("judge_name"),
        "focus": j.get("focus"),
        "score": j.get("score"),
        "pros": (j.get("pros") or [])[:4],
        "cons": (j.get("cons") or [])[:4],
        "suggestions": (j.get("suggestions") or [])[:4],
    }


def evaluate_midi_deliberation(midi_path: str, model_judge: str, model_chair: str) -> Dict[str, Any]:
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

    # ---- Round 1：独立评审 ----
    round1: Dict[str, Any] = {}
    for j in JUDGES:
        user = (
            f"评审维度：{j['focus']}\n"
            f"MIDI 摘要(JSON)：{meta_s}\n\n"
            "请按 schema 输出严格 JSON。"
        )
        out = _call_llm_json(client, model_judge, judge_system, user, max_try=2)
        out["judge_name"] = j["name"]
        out["focus"] = j["focus"]
        sc = int(out.get("score", 0))
        out["score"] = max(0, min(25, sc))
        round1[j["name"]] = out

    # ---- Round 2：看同伴观点后复议修正（协作）----
    revise_system = (
        "你是音乐评审。你已经完成初评，现在看到了其他评委的观点摘要。\n"
        "请你进行复议：必要时修正你的分数与建议，使整体更一致、更有可执行性。\n"
        "你必须只输出一个 JSON 对象，不能输出任何额外文本。\n"
        "评分范围 0-25（整数），schema 同初评："
        "{"
        "\"judge_name\": str, \"focus\": str, \"score\": int, "
        "\"pros\": [str], \"cons\": [str], \"suggestions\": [str], \"rationale\": str"
        "}"
    )

    round2: Dict[str, Any] = {}
    for j in JUDGES:
        me = round1[j["name"]]
        peers = []
        for other_name, other in round1.items():
            if other_name == j["name"]:
                continue
            peers.append(_compress_peer(other))

        user = (
            f"你的评审维度：{j['focus']}\n"
            f"MIDI 摘要(JSON)：{meta_s}\n\n"
            f"你的初评(JSON)：{json.dumps(_compress_peer(me), ensure_ascii=False)}\n\n"
            f"其他评委观点摘要(JSON 数组)：{json.dumps(peers, ensure_ascii=False)}\n\n"
            "请复议后按 schema 输出严格 JSON。"
        )
        out = _call_llm_json(client, model_judge, revise_system, user, max_try=2)
        out["judge_name"] = j["name"]
        out["focus"] = j["focus"]
        sc = int(out.get("score", me.get("score", 0)))
        out["score"] = max(0, min(25, sc))
        round2[j["name"]] = out

    # 从 Round2 取最终分
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
        "}\n分数范围：各维度 0-25，总分 0-100（整数）。"
    )

    chair_user = (
        f"MIDI 摘要：{meta_s}\n\n"
        f"评委最终输出（Round2）：{json.dumps(round2, ensure_ascii=False)}\n\n"
        "要求：pros/cons/suggestions 各给 3-8 条，尽量具体可执行。"
    )

    chair = _call_llm_json(client, model_chair, chair_system, chair_user, max_try=2)

    report = {
        "version": "music_eval_v2_deliberation",
        "ts": int(time.time()),
        "midi_meta": meta,
        "deliberation": {"rounds": 2, "note": "Round1 independent, Round2 revised after peer summaries"},
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
    ap.add_argument("--midi", required=True, help="path to .mid/.midi")
    ap.add_argument("--out", default="report.json")
    ap.add_argument("--model_judge", default="deepseek-chat")
    ap.add_argument("--model_chair", default="deepseek-chat")
    args = ap.parse_args()

    midi_path = str(Path(args.midi).resolve())
    out_path = Path(args.out)

    report = evaluate_midi_deliberation(midi_path, args.model_judge, args.model_chair)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OK ->", out_path.resolve())
    print("score_total =", report.get("score_total"))


if __name__ == "__main__":
    main()
