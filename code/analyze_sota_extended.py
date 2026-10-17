# -*- coding: utf-8 -*-
"""
analyze_sota_extended.py — 汇总「反向域对」与「healthy/disease 正向」两组 SOTA 结果，
计算 7 种子均值±std，并用配对 Wilcoxon 检验验证两条结论：
  (1) 方向不对称：反向 field->lab 上 SOTA 方法是否同样无益/有害；
  (2) 任务难度梯度：healthy/disease 正向的 UDA 增益是否同样大于 healthy/rust 正向。
"""
import json
import statistics as st
from pathlib import Path

import scipy.stats as ss

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "outputs" / "results"

SEEDS = ["42", "123", "2024", "7", "99", "13", "777"]

# 方法：tag 前缀 -> 显示名
METHODS = [
    ("P2", "P2(ImageNet微调)"),
    ("P7_dann", "DANN"),
    ("CW-DANN", "CW-DANN"),
    ("DeepCORAL", "DeepCORAL"),
    ("DAN", "DAN"),
    ("CDAN", "CDAN"),
    ("SHOT", "SHOT"),
]

# 正向主任务 healthy/rust（已算好的 best/fixed，用于难度梯度对比）
FORWARD_HR = {
    "P2": (70.4, 46.2),
    "P7_dann": (79.0, 61.8),
    "CW-DANN": (81.8, 65.2),
    "DeepCORAL": (79.2, 65.3),
    "DAN": (83.6, 73.8),
    "CDAN": (78.5, 51.8),
    "SHOT": (75.1, 74.1),
}

TASKS = [
    ("_rev", "反向 field->lab (healthy/rust)"),
    ("_hd", "正向 lab->field (healthy/disease)"),
]


def load(prefix: str, suffix: str, seeds) -> list:
    """返回 [(best, fixed)] 按种子顺序；缺失种子 None 占位。文件名 {prefix}{suffix}_s{seed}。"""
    out = []
    for s in seeds:
        p = RES / f"{prefix}{suffix}_s{s}_metrics.json"
        if not p.exists():
            out.append(None)
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        best = d["best"]["test_acc"] if isinstance(d.get("best"), dict) else d.get("best")
        fixed = d.get("test_acc")
        out.append((best, fixed))
    return out


def stats(pairs):
    bests = [x[0] * 100 for x in pairs if x and x[0] is not None]
    fixeds = [x[1] * 100 for x in pairs if x and x[1] is not None]
    if not bests:
        return "N/A", "N/A"
    b = f"{st.mean(bests):.1f} ± {st.stdev(bests):.1f}" if len(bests) > 1 else f"{bests[0]:.1f}"
    f = f"{st.mean(fixeds):.1f} ± {st.stdev(fixeds):.1f}" if len(fixeds) > 1 else f"{fixeds[0]:.1f}"
    return b, f


def wilcoxon(a_pairs, b_pairs, mode="best"):
    """配对 Wilcoxon（两方法都有值的种子），返回 (b-a 均值差 pp, p)。"""
    idx = 0 if mode == "best" else 1
    xs, ys = [], []
    for a, b in zip(a_pairs, b_pairs):
        if a and b and a[idx] is not None and b[idx] is not None:
            xs.append(a[idx] * 100)
            ys.append(b[idx] * 100)
    if len(xs) < 5:
        return None
    diff = st.mean([y - x for x, y in zip(xs, ys)])
    try:
        p = ss.wilcoxon(xs, ys, alternative="two-sided").pvalue
    except Exception:
        p = float("nan")
    return diff, p


def fmt_w(a_pairs, b_pairs, mode="best"):
    """b vs a：+xx.xpp (p=0.xxx)。"""
    r = wilcoxon(a_pairs, b_pairs, mode)
    if r is None:
        return "—"
    d, p = r
    sign = "+" if d >= 0 else "−"
    return f"{sign}{abs(d):.1f}pp (p={p:.3f})"


def main():
    lines = []
    for suffix, tname in TASKS:
        lines.append(f"\n{'='*72}\n## {tname}\n{'='*72}")
        data = {p: load(p, suffix, SEEDS) for p, _ in METHODS}

        lines.append("\n| 方法 | Best-epoch | Fixed-epoch |")
        lines.append("|---|---|---|")
        for p, disp in METHODS:
            b, f = stats(data[p])
            lines.append(f"| {disp} | {b}% | {f}% |")

        lines.append("\n**各方法 vs P2（是否有效，best / fixed）**")
        for p, disp in METHODS[1:]:
            b = fmt_w(data["P2"], data[p], "best")
            f = fmt_w(data["P2"], data[p], "fixed")
            lines.append(f"- {disp}：best {b} / fixed {f}")

        lines.append("\n**SOTA vs DANN / CW-DANN（best / fixed）**")
        for p, disp in METHODS[3:]:
            bd = fmt_w(data["P7_dann"], data[p], "best")
            fd = fmt_w(data["P7_dann"], data[p], "fixed")
            bc = fmt_w(data["CW-DANN"], data[p], "best")
            fc = fmt_w(data["CW-DANN"], data[p], "fixed")
            lines.append(f"- {disp}：vs DANN best {bd} / fixed {fd}；vs CW-DANN best {bc} / fixed {fc}")

    # 任务难度梯度（best-epoch 增益对比）
    lines.append(f"\n{'='*72}\n## 任务难度梯度：healthy/rust 正向 vs healthy/disease 正向\n{'='*72}")
    lines.append("\n| 方法 | healthy/rust 增益(best) | healthy/disease 增益(best) | 梯度 |")
    lines.append("|---|---|---|---|")
    hd = {p: load(p, "_hd", SEEDS) for p, _ in METHODS}
    p2_hd_best = st.mean([x[0] * 100 for x in hd["P2"] if x and x[0] is not None])
    for p, disp in METHODS[1:]:
        gain_hr = FORWARD_HR[p][0] - FORWARD_HR["P2"][0]
        bests = [x[0] * 100 for x in hd[p] if x and x[0] is not None]
        gain_hd = st.mean(bests) - p2_hd_best if bests else float("nan")
        grad = f"{'↑' if gain_hd > gain_hr else '↓'} ({gain_hd - gain_hr:+.1f}pp)"
        lines.append(f"| {disp} | +{gain_hr:.1f}pp | +{gain_hd:.1f}pp | {grad} |")

    report = "\n".join(lines)
    print(report)
    out = ROOT / "SOTA扩展实验结果_反向与hd.md"
    out.write_text("# SOTA 扩展实验结果（反向 + healthy/disease）\n\n" + report, encoding="utf-8")
    print(f"\n[SAVED] {out}")


if __name__ == "__main__":
    main()
