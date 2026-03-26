from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from tool_generate import EEG_MIDI_p2p_r2v, EEG_MIDI_r2p_p2v


def _to_numeric_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s


def _sanitize_instrument_names(music_obj) -> None:
    for i, inst in enumerate(getattr(music_obj, "instruments", [])):
        name = getattr(inst, "name", "")
        if name is None:
            inst.name = f"track_{i}"
        elif not isinstance(name, str):
            inst.name = str(name)
        elif name.strip() == "":
            inst.name = f"track_{i}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--channel", type=int, default=1, help="无表头CSV：0列=时间，1~为通道")
    ap.add_argument("--rule", choices=["p2p_r2v", "r2p_p2v"], default="p2p_r2v")
    ap.add_argument("--seconds", type=float, default=60.0, help="只处理前N秒，避免太长")
    ap.add_argument("--out", default="./files/out_60s.mid")
    ap.add_argument("--numerator", type=int, default=4)
    ap.add_argument("--denominator", type=int, default=4)
    ap.add_argument("--magnet", type=int, default=2)
    ap.add_argument("--pitches", default="84,60,36")
    ap.add_argument("--velocities", default="120,80,40")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"找不到CSV：{csv_path}")

    # 你的数据是无表头纯矩阵：强制 header=None
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] < 2:
        raise SystemExit("CSV列数不足：至少需要 时间列(第0列) + 1个通道列")

    if args.channel <= 0 or args.channel >= df.shape[1]:
        raise SystemExit(f"--channel 必须在 1~{df.shape[1]-1}（0列是时间）")

    # time / track 保持为 pandas Series（关键：不要转 numpy）
    time = _to_numeric_series(df.iloc[:, 0])
    track = _to_numeric_series(df.iloc[:, args.channel])

    if time.isna().all():
        raise SystemExit("第0列时间轴无法解析为数值（全是NaN）")
    if track.isna().all():
        raise SystemExit(f"通道列({args.channel})无法解析为数值（全是NaN）")

    # 归一到从0开始
    time = time - time.min()

    # 过滤无效点 + 截取前N秒
    mask = time.notna() & track.notna()
    if args.seconds and args.seconds > 0:
        mask = mask & (time <= float(args.seconds))

    time = time[mask]
    track = track[mask]

    if len(time) < 10:
        raise SystemExit("有效数据点太少（<10），请检查 seconds/channel 是否正确")

    # 关键：给 track 一个“字符串名字”，避免 mido 写 track_name 报错
    track.name = f"ch_{args.channel}"

    pitches = tuple(int(x.strip()) for x in args.pitches.split(","))
    velocities = tuple(int(x.strip()) for x in args.velocities.split(","))
    if len(pitches) != 3 or len(velocities) != 3:
        raise SystemExit("pitches/velocities 必须是 3 个整数，例如 84,60,36")

    if args.rule == "p2p_r2v":
        m = EEG_MIDI_p2p_r2v(
            time, track,
            numerator=args.numerator,
            denomimator=args.denominator,
            magnet=args.magnet,
            pithces=pitches,
            velocities=velocities
        )
    else:
        m = EEG_MIDI_r2p_p2v(
            time, track,
            numerator=args.numerator,
            denomimator=args.denominator,
            magnet=args.magnet,
            pithces=pitches,
            velocities=velocities
        )

    # 再兜底一次：所有轨道名强制变成 str
    _sanitize_instrument_names(m)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.dump(str(out))

    print("OK:", out.resolve())


if __name__ == "__main__":
    main()