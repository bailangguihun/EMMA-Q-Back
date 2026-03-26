import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
import pretty_midi
from openai import OpenAI

from maseval.core.benchmark import Benchmark
from maseval.core.environment import Environment
from maseval.core.evaluator import Evaluator
from maseval.core.agent import AgentAdapter

# ---------- 1) MIDI 摘要（轻量、稳定） ----------
def midi_summary(midi_path: str) -> Dict[str, Any]:
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
        ioi = [onset[i+1] - onset[i] for i in range(len(onset)-1)]
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


def safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


def call_llm_json(client: OpenAI, model: str, system: str, user: str, max_try: int = 2) -> Dict[str, Any]:
    last_err = None
    for _ in range(max_try):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=1200,
        )
        txt = resp.choices[0].message.content or ""
        try:
            return safe_json_loads(txt)
        except Exception as e:
            last_err = e
            # 让模型“修 JSON”
            fix = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是 JSON 修复器。只输出可被 json.loads 解析的 JSON。"},
                    {"role": "user", "content": "把下面内容改写成严格 JSON（只输出 JSON）。\n\n" + txt},
                ],
                temperature=0.0,
                max_tokens=1200,
            )
            txt2 = fix.choices[0].message.content or ""
            try:
                return safe_json_loads(txt2)
            except Exception as e2:
                last_err = e2
    raise last_err or RuntimeError("LLM JSON parse failed")


# ---------- 2) 评审 PoC：evaluate_midi(midi_path)->report(dict) ----------
JUDGES = [
    {"name": "harmony_judge", "focus": "和声/乐理"},
    {"name": "rhythm_judge", "focus": "节奏/律动"},
    {"name": "structure_judge", "focus": "结构/段落"},
    {"name": "style_judge", "focus": "风格一致性/可听性"},
]

def evaluate_midi(midi_path: str, model_judge: str, model_chair: str) -> Dict[str, Any]:
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    api_key = os.environ["DEEPSEEK_API_KEY"]
    client = OpenAI(api_key=api_key, base_url=base_url)

    meta = midi_summary(midi_path)
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
        out = call_llm_json(client, model_judge, judge_system, user, max_try=2)
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

    chair = call_llm_json(client, model_chair, chair_system, chair_user, max_try=2)

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


# ---------- 3) MASEval 适配层（Benchmark / Environment / AgentAdapter / Evaluator） ----------
class NullMusicEnv(Environment):
    def setup_state(self, task_data: dict) -> Any:
        return {"task_data": task_data}

    def create_tools(self) -> Dict[str, Any]:
        return {}  # 音乐评审不需要工具


class MusicEvalAgent(AgentAdapter):
    def __init__(self, name: str, model_judge: str, model_chair: str, out_dir: Path):
        self.name = name
        self._messages: List[Dict[str, Any]] = []
        self.model_judge = model_judge
        self.model_chair = model_chair
        self.out_dir = out_dir

    def get_messages(self):
        # MASEval 里通常是 MessageHistory，这里用 list[dict] 也能被 traces 收集
        return self._messages

    def run(self, query: str) -> Any:
        # query = midi_path
        self._messages.append({"role": "user", "content": query})
        report = evaluate_midi(query, self.model_judge, self.model_chair)
        self._messages.append({"role": "assistant", "content": json.dumps(report, ensure_ascii=False)})
        # 落盘 report.json（固定 schema）
        p = Path(query)
        out_path = self.out_dir / f"{p.stem}.report.json"
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report


class MusicEvalEvaluator(Evaluator):
    def filter_traces(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        return traces

    def __call__(self, traces: Dict[str, Any], final_answer: Any = None) -> Dict[str, Any]:
        # final_answer 是 dict(report)
        report = final_answer if isinstance(final_answer, dict) else {}
        scores = report.get("scores", {}) if isinstance(report, dict) else {}
        return {
            "score_total": report.get("score_total"),
            "harmony": scores.get("harmony"),
            "rhythm": scores.get("rhythm"),
            "structure": scores.get("structure"),
            "style": scores.get("style"),
        }


class MusicEvalBenchmark(Benchmark):
    def __init__(self, out_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def setup_environment(self, agent_data: Dict[str, Any], task):
        return NullMusicEnv(task.environment_data if hasattr(task, "environment_data") else {})

    def setup_user(self, agent_data: Dict[str, Any], environment, task):
        return None

    def setup_agents(self, agent_data: Dict[str, Any], environment, task, user):
        agent = MusicEvalAgent(
            name="music_eval_agent",
            model_judge=agent_data.get("model_judge", "deepseek-chat"),
            model_chair=agent_data.get("model_chair", "deepseek-chat"),
            out_dir=self.out_dir,
        )
        return [agent], {"music_eval_agent": agent}

    def setup_evaluators(self, environment, task, agents: Sequence[AgentAdapter], user):
        return [MusicEvalEvaluator(task, environment, user)]

    def run_agents(self, agents: Sequence[AgentAdapter], task, environment, query: str):
        return agents[0].run(query)

    def evaluate(self, evaluators: Sequence[Evaluator], agents: Dict[str, AgentAdapter], final_answer: Any, traces: Dict[str, Any]):
        results = []
        for ev in evaluators:
            filtered = ev.filter_traces(traces)
            results.append(ev(filtered, final_answer=final_answer))
        return results

    def get_model_adapter(self, model_id: str, **kwargs):
        # 本 benchmark 直接用 OpenAI SDK 调 DeepSeek，不走 MASEval ModelAdapter
        raise NotImplementedError("MusicEvalBenchmark does not use get_model_adapter().")


# ---------- 4) tasks 构造（尽量兼容不同版本 Task 定义） ----------
def build_tasks(midi_dir: str):
    from maseval.core.task import Task  # Task 页面抓不到，但模块存在于包内（Benchmark 文档也引用）
    import inspect

    midis = sorted([p for p in Path(midi_dir).glob("**/*") if p.suffix.lower() in {".mid", ".midi"}])
    if not midis:
        raise SystemExit(f"No MIDI under: {midi_dir}")

    sig = inspect.signature(Task)
    tasks = []
    for p in midis:
        kwargs = {}
        if "task_id" in sig.parameters:
            kwargs["task_id"] = p.stem
        if "query" in sig.parameters:
            kwargs["query"] = str(p)
        if "environment_data" in sig.parameters:
            kwargs["environment_data"] = {}
        if "evaluation_data" in sig.parameters:
            kwargs["evaluation_data"] = {}
        if "metadata" in sig.parameters:
            kwargs["metadata"] = {"midi_path": str(p)}
        tasks.append(Task(**kwargs))
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi_dir", required=True)
    ap.add_argument("--out_dir", default="music_eval_out")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--model_judge", default="deepseek-chat")
    ap.add_argument("--model_chair", default="deepseek-chat")
    args = ap.parse_args()

    tasks = build_tasks(args.midi_dir)

    bench = MusicEvalBenchmark(
        out_dir=args.out_dir,
        num_workers=args.num_workers,
        n_task_repeats=1,
        progress_bar=True,
        fail_on_task_error=False,
        fail_on_evaluation_error=False,
        fail_on_setup_error=False,
    )

    reports = bench.run(tasks=tasks, agent_data={"model_judge": args.model_judge, "model_chair": args.model_chair})

    # 汇总：从 out_dir 读 report.json（不依赖 reports 内部结构）
    out_dir = Path(args.out_dir)
    rows = []
    for rp in sorted(out_dir.glob("*.report.json")):
        obj = json.loads(rp.read_text(encoding="utf-8"))
        sc = obj.get("scores", {})
        rows.append({
            "file": rp.name.replace(".report.json", ".mid"),
            "score_total": obj.get("score_total"),
            "harmony": sc.get("harmony"),
            "rhythm": sc.get("rhythm"),
            "structure": sc.get("structure"),
            "style": sc.get("style"),
            "report": str(rp),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    agg = df[["score_total", "harmony", "rhythm", "structure", "style"]].mean(numeric_only=True).to_dict()
    (out_dir / "aggregate.json").write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OK ->", out_dir.resolve())
    print(json.dumps(agg, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
