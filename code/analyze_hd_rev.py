# -*- coding: utf-8 -*-
"""
analyze_hd_rev.py — healthy/disease 反向（field->lab）的完整统计：
复用 evaluate_balanced_sota 的模型重建/评估，收集 7 种子 per-seed 的
accuracy 与 balanced accuracy，做配对 Wilcoxon，输出结论。
"""
import statistics as st
from pathlib import Path

import torch
import scipy.stats as ss
from torch.utils.data import DataLoader

import evaluate_balanced_sota as ev

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "outputs" / "results"
SEEDS = ev.SEEDS
METHODS = ev.METHODS  # [(prefix, kind, disp)]


def collect(prefix, kind, device, loader):
    accs, bals = [], []
    for s in SEEDS:
        ckpt = RES / f"{prefix}_s{s}_best.pth"
        if not ckpt.exists():
            accs.append(None)
            bals.append(None)
            continue
        model = ev.build(kind, ckpt).to(device)
        acc, bal, _ = ev.evaluate(model, loader, device, kind)
        accs.append(acc * 100)
        bals.append(bal * 100)
    return accs, bals


def wilcoxon(a, b):
    """a, b 为等长列表（含 None），返回 (b-a 均值差 pp, p)。"""
    xs, ys = [], []
    for x, y in zip(a, b):
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
    if len(xs) < 5:
        return None
    diff = st.mean([y - x for x, y in zip(xs, ys)])
    try:
        p = ss.wilcoxon(xs, ys, alternative="two-sided").pvalue
    except Exception:
        p = float("nan")
    return diff, p


def fmt(a, b):
    r = wilcoxon(a, b)
    if r is None:
        return "—"
    d, p = r
    sign = "+" if d >= 0 else "−"
    return f"{sign}{abs(d):.1f}pp (p={p:.3f})"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = ev.TestDS(ROOT / "data" / "processed" / "test_T_hd_rev.csv")
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    data = {}
    for prefix, kind, disp in METHODS:
        accs, bals = collect(prefix, kind, device, loader)
        data[disp] = {"acc": accs, "bal": bals}

    p2 = data["P2"]
    dann = data["DANN"]
    cwdann = data["CW-DANN"]

    lines = []
    lines.append("# healthy/disease 反向（field->lab）SOTA 全方法结果\n")
    lines.append("目标域 BRACOL 测试集 healthy 79 / disease 426（失衡 5.4:1）。7 种子配对 Wilcoxon。\n")
    lines.append("## 一、best-epoch 汇总（accuracy / balanced accuracy）\n")
    lines.append("| 方法 | acc(best) | balanced(best) | vs P2(bal) |")
    lines.append("|---|---|---|---|")
    for disp in ["P2", "DANN", "CW-DANN", "DeepCORAL", "DAN", "CDAN", "SHOT"]:
        a = data[disp]["acc"]
        b = data[disp]["bal"]
        a_s = f"{st.mean(a):.1f} ± {st.stdev(a):.1f}"
        b_s = f"{st.mean(b):.1f} ± {st.stdev(b):.1f}"
        vs = "—" if disp == "P2" else fmt(p2["bal"], b)
        lines.append(f"| {disp} | {a_s}% | {b_s}% | {vs} |")

    lines.append("\n## 二、Wilcoxon（balanced accuracy，7 种子配对）\n")
    lines.append("**各 UDA vs P2**")
    for disp in ["DANN", "CW-DANN", "DeepCORAL", "DAN", "CDAN", "SHOT"]:
        lines.append(f"- {disp}：balanced {fmt(p2['bal'], data[disp]['bal'])}")
    lines.append("\n**SOTA vs DANN / CW-DANN**")
    for disp in ["DeepCORAL", "DAN", "CDAN", "SHOT"]:
        lines.append(f"- {disp}：vs DANN {fmt(dann['bal'], data[disp]['bal'])}；vs CW-DANN {fmt(cwdann['bal'], data[disp]['bal'])}")

    report = "\n".join(lines)
    print(report)
    out = ROOT / "hd_rev_SOTA结果.md"
    out.write_text(report, encoding="utf-8")
    print(f"\n[SAVED] {out}")


if __name__ == "__main__":
    main()
