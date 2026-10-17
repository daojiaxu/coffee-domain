# -*- coding: utf-8 -*-
"""
analyze_results.py — 汇总所有实验结果 + Wilcoxon 符号秩检验。

1. 二分类跨域任务：读 P2 / DANN / CB-DANN(focal) / conditional / focal+cond 的多种子
   best_acc，计算均值±标准差，并做配对 Wilcoxon 符号秩检验（精确分布）。
2. 5 类任务：读 b5_ce / b5_focal 的 balanced_acc + macro_f1。
3. 输出一份论文级汇总表（Markdown）与 JSON。

用法：python analyze_results.py
"""
import json
import os
from pathlib import Path
from itertools import product
import statistics

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "outputs" / "results"


def load(tag):
    with open(RES / f"{tag}_metrics.json", "r", encoding="utf-8") as f:
        d = json.load(f)
    return d


def best_acc(tag):
    d = load(tag)
    return d["best"]["test_acc"]


def wilcoxon_signed_rank(a, b):
    """配对 Wilcoxon 符号秩检验（精确双尾 p 值，枚举 2^n 符号）。"""
    d = [a[i] - b[i] for i in range(len(a))]
    d = [x for x in d if abs(x) > 1e-9]
    n = len(d)
    if n == 0:
        return 1.0, None
    # 按 |d| 排序赋秩（并列用平均秩）
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1

    W_plus = sum(ranks[k] for k in range(n) if d[k] > 0)
    W_minus = sum(ranks[k] for k in range(n) if d[k] < 0)
    W_obs = min(W_plus, W_minus)
    rank_sum = sum(ranks)

    # 精确分布
    total = 0
    count = 0
    for signs in product([-1, 1], repeat=n):
        s = sum(ranks[k] * signs[k] for k in range(n))
        wm = (rank_sum - s) / 2.0
        w = min(wm, rank_sum - wm)
        total += 1
        if w <= W_obs + 1e-9:
            count += 1
    p = count / total
    return p, (W_plus, W_minus, n)


def mean_std(vals):
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, s


def fmt(vals):
    m, s = mean_std(vals)
    return f"{m * 100:.1f}% ± {s * 100:.1f}%"


def main():
    lines = []

    # ===== 二分类跨域 =====
    p2 = [best_acc(t) for t in
          ["P2_imagenet", "P2_imagenet_s123", "P2_imagenet_s2024",
           "P2_imagenet_s7", "P2_imagenet_s99", "P2_imagenet_s13", "P2_imagenet_s777"]
          if os.path.exists(RES / f"{t}_metrics.json")]
    dann = [best_acc(t) for t in
            ["P7_dann_s42", "P7_dann_s123", "P7_dann_s2024",
             "P7_dann_s7", "P7_dann_s99", "P7_dann_s13", "P7_dann_s777"]
            if os.path.exists(RES / f"{t}_metrics.json")]
    cbdann = [best_acc(t) for t in
              ["CW-DANN_focal_s42", "CW-DANN_focal_s123", "CW-DANN_focal_s2024",
               "CW-DANN_focal_s7", "CW-DANN_focal_s99", "CW-DANN_focal_s13", "CW-DANN_focal_s777"]
              if os.path.exists(RES / f"{t}_metrics.json")]
    cond = [best_acc(t) for t in
            ["CW-DANN_cond_s42", "CW-DANN_cond_s123", "CW-DANN_cond_s2024"]
            if os.path.exists(RES / f"{t}_metrics.json")]
    both = [best_acc(t) for t in
            ["CW-DANN_s42", "CW-DANN_s123", "CW-DANN_s2024"]
            if os.path.exists(RES / f"{t}_metrics.json")]

    p0 = best_acc("P0_domain")
    p1 = best_acc("P1_scratch")
    p4 = best_acc("P4_simclr")

    lines.append("# 咖啡叶病跨域泛化 — 最终实验结果汇总")
    lines.append("")
    lines.append("## 一、二分类跨域任务（BRACOL 白底 → RoCoLe 田间）")
    lines.append("")
    lines.append("| 方法 | 目标域标注 | 种子数 | test_acc（均值±标准差） |")
    lines.append("|---|---|---|---|")
    lines.append(f"| 域内上界 P0 | 有 | 1 | {p0 * 100:.1f}% |")
    lines.append(f"| ImageNet 微调 P2 | 无 | {len(p2)} | {fmt(p2)} |")
    lines.append(f"| 从头训练 P1 | 无 | 1 | {p1 * 100:.1f}% |")
    lines.append(f"| SimCLR 自监督 P4 | 无 | 1 | {p4 * 100:.1f}% |")
    lines.append(f"| DANN P7 | 无 | {len(dann)} | {fmt(dann)} |")
    lines.append(f"| **CB-DANN（本文方法）** | 无 | {len(cbdann)} | **{fmt(cbdann)}** |")
    lines.append("")
    lines.append("## 二、消融分析")
    lines.append("")
    lines.append("| 配置 | focal | conditional | 种子数 | test_acc |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| DANN | ✗ | ✗ | {len(dann)} | {fmt(dann)} |")
    lines.append(f"| **CB-DANN** | ✓ | ✗ | {len(cbdann)} | **{fmt(cbdann)}** |")
    lines.append(f"| DANN + conditional | ✗ | ✓ | {len(cond)} | {fmt(cond)} |")
    lines.append(f"| focal + conditional | ✓ | ✓ | {len(both)} | {fmt(both)} |")
    lines.append("")
    lines.append("## 三、显著性检验（Wilcoxon 符号秩，配对）")
    lines.append("")
    for name_a, A, name_b, B in [("CB-DANN", cbdann, "DANN", dann),
                                 ("CB-DANN", cbdann, "P2", p2),
                                 ("DANN", dann, "P2", p2)]:
        n = min(len(A), len(B))
        a, b = A[:n], B[:n]
        p, stat = wilcoxon_signed_rank(a, b)
        d = [a[i] - b[i] for i in range(n)]
        sign = "+" if statistics.mean(d) >= 0 else "-"
        lines.append(f"- {name_a} vs {name_b}：n={n}，平均差 {sign}{abs(statistics.mean(d)) * 100:.1f}pp，"
                     f"Wilcoxon p = {p:.4f} {'（显著）' if p < 0.05 else '（不显著）'}")
    lines.append("")

    # ===== 5 类任务 =====
    lines.append("## 四、5 类细粒度任务（BRACOL 域内，CE vs focal）")
    lines.append("")
    lines.append("| 方法 | 种子数 | balanced_acc | macro-F1 |")
    lines.append("|---|---|---|---|")
    for tag in ["b5_ce", "b5_focal"]:
        tags = [f"{tag}_s42", f"{tag}_s123", f"{tag}_s2024"]
        tags = [t for t in tags if os.path.exists(RES / f"{t}_metrics.json")]
        if not tags:
            continue
        bal = [load(t)["best"]["balanced_acc"] for t in tags]
        f1 = [load(t)["best"]["macro_f1"] for t in tags]
        name = "CE（基线）" if tag == "b5_ce" else "Focal（类别平衡）"
        lines.append(f"| {name} | {len(tags)} | {fmt(bal)} | {fmt(f1)} |")
    lines.append("")

    text = "\n".join(lines)
    out = ROOT / "实验结果汇总.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print("\n[OK] 已写入:", out)


if __name__ == "__main__":
    main()
