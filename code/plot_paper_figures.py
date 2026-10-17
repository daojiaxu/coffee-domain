# -*- coding: utf-8 -*-
"""
plot_paper_figures.py — 生成 SCI 期刊论文所需的全部图表。
输出到 outputs/paper/。
"""
import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "font.family": "DejaVu Sans",
})

# ---------- 已验证数据（7 种子均值±std） ----------
METHODS = ["P2", "DANN", "CW-DANN", "DeepCORAL", "DAN", "CDAN", "SHOT"]
MAIN_BEST = [70.4, 79.0, 81.8, 79.2, 83.6, 78.5, 75.1]
MAIN_BEST_S = [4.1, 3.3, 3.8, 4.1, 3.5, 6.1, 2.1]
MAIN_FIXED = [46.2, 61.8, 65.2, 65.3, 73.8, 51.8, 74.1]
MAIN_FIXED_S = [3.5, 8.3, 10.0, 8.0, 5.5, 7.3, 2.1]

REV_BEST = [96.3, 95.8, 96.2, 96.3, 95.4, 96.2, 90.4]
REV_BEST_S = [0.8, 0.8, 0.6, 0.4, 0.9, 0.7, 2.4]
REV_FIXED = [93.6, 88.0, 92.6, 91.0, 91.3, 91.4, 82.6]
REV_FIXED_S = [3.3, 8.1, 1.3, 1.7, 1.3, 3.3, 5.1]

HD_BEST = [51.5, 64.4, 70.1, 58.2, 67.5, 64.5, 66.9]
HD_BEST_S = [2.6, 4.4, 4.2, 3.5, 6.2, 6.3, 4.4]
HD_FIXED = [48.4, 52.4, 57.3, 50.3, 57.8, 52.2, 66.0]
HD_FIXED_S = [0.4, 2.3, 4.1, 1.6, 7.5, 3.8, 4.4]

HD_REV_BAL = [84.3, 78.3, 76.9, 84.6, 85.6, 74.4, 80.7]
HD_REV_BAL_S = [8.3, 8.9, 14.9, 3.0, 4.9, 14.8, 1.7]

C_P2 = "#8c8c8c"; C_OURS = "#c0392b"; C_UDA = "#3b78c3"; C_ORACLE = "#555555"
COLORS = [C_P2, C_UDA, C_OURS, C_UDA, C_UDA, C_UDA, C_UDA]


# ---------- Fig: 正向主任务 7 方法 best/fixed ----------
def fig_main():
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(METHODS)); w = 0.38
    b1 = ax.bar(x - w/2, MAIN_BEST, w, yerr=MAIN_BEST_S, capsize=3,
                color=[c + "" for c in COLORS], alpha=0.95, label="Best-epoch (oracle)")
    b2 = ax.bar(x + w/2, MAIN_FIXED, w, yerr=MAIN_FIXED_S, capsize=3,
                color=[c for c in COLORS], alpha=0.45, hatch="//", edgecolor="white",
                label="Fixed-epoch (label-free)")
    for xi, v in zip(x, MAIN_BEST):
        ax.text(xi - w/2, v + 1.2, f"{v:.1f}", ha="center", fontsize=8.5)
    for xi, v in zip(x, MAIN_FIXED):
        ax.text(xi + w/2, v + 1.2, f"{v:.1f}", ha="center", fontsize=8.5)
    ax.set_ylabel("Target-domain accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(x); ax.set_xticklabels(METHODS)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_main_results.png", dpi=300)
    plt.close(fig)


# ---------- Fig: 方向不对称 ----------
def fig_direction():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True)
    tasks = [("healthy vs rust", MAIN_BEST, MAIN_BEST_S, REV_BEST, REV_BEST_S),
             ("healthy vs disease (balanced)", HD_BEST, HD_BEST_S, HD_REV_BAL, HD_REV_BAL_S)]
    for ax, (tname, fw, fws, bw, bws) in zip(axes, tasks):
        x = np.arange(len(METHODS)); w = 0.38
        ax.bar(x - w/2, fw, w, yerr=fws, capsize=3, color=[c for c in COLORS], alpha=0.9,
               label="lab $\\rightarrow$ field")
        ax.bar(x + w/2, bw, w, yerr=bws, capsize=3, color=[c for c in COLORS], alpha=0.45,
               hatch="//", edgecolor="white", label="field $\\rightarrow$ lab")
        ax.set_xticks(x); ax.set_xticklabels(METHODS, rotation=30, ha="right")
        ax.set_title(tname)
        ax.grid(axis="y", ls=":", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Accuracy / balanced accuracy (%)")
    axes[0].set_ylim(0, 108)
    axes[0].legend(loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_direction.png", dpi=300)
    plt.close(fig)


# ---------- Fig: 任务难度梯度 ----------
def fig_gradient():
    # UDA 增益（best-epoch, vs P2）；severity 只有 DANN/CW-DANN
    tasks = ["healthy/rust\n(+8.6 DANN)", "healthy/disease\n(+12.9 DANN)", "severity 4-level\n(+16.3 DANN)"]
    gains = {
        "DANN":      [8.6, 12.9, 16.3],
        "CW-DANN":   [11.3, 18.7, 12.1],
        "DeepCORAL": [8.7, 6.7, None],
        "DAN":       [13.2, 16.0, None],
        "CDAN":      [8.0, 13.0, None],
        "SHOT":      [4.7, 15.4, None],
    }
    colors = {"DANN": "#3b78c3", "CW-DANN": "#c0392b", "DeepCORAL": "#2e8b57",
              "DAN": "#8e44ad", "CDAN": "#e67e22", "SHOT": "#16a085"}
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(3)
    for m, g in gains.items():
        xs = [i for i, v in enumerate(g) if v is not None]
        ys = [v for v in g if v is not None]
        ax.plot(xs, ys, "o-", label=m, color=colors[m], lw=1.8, ms=6)
        ax.annotate(f"+{ys[-1]:.1f}", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(6, -2), fontsize=8.5, color=colors[m])
    ax.set_xticks(x); ax.set_xticklabels(tasks)
    ax.set_ylabel("UDA gain over P2 (pp, best-epoch)")
    ax.set_xlabel("Task (ordered by source-class imbalance / difficulty)")
    ax.grid(ls=":", alpha=0.4)
    ax.legend(ncol=3, loc="upper left", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_gradient.png", dpi=300)
    plt.close(fig)


# ---------- Fig: CW-DANN 架构图 ----------
def fig_architecture():
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.axis("off")

    def box(xy, w, h, text, fc, ec="black", fs=10, tc="white"):
        r = plt.Rectangle(xy, w, h, fc=fc, ec=ec, lw=1.2, zorder=2)
        ax.add_patch(r)
        ax.text(xy[0] + w/2, xy[1] + h/2, text, ha="center", va="center",
                fontsize=fs, color=tc, zorder=3, fontweight="bold")

    def arrow(p1, p2, color="black", style="-|>", lw=1.6):
        ax.annotate("", xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw))

    # 输入
    box((0.2, 3.2), 1.5, 1.1, "Source $x^s$\n(labeled)", "#34495e")
    box((0.2, 1.0), 1.5, 1.1, "Target $x^t$\n(unlabeled)", "#7f8c8d")
    # 骨干
    box((2.6, 2.2), 2.0, 1.2, "ResNet-50\nfeature $f$\n(ImageNet init)", "#2e6da4")
    arrow((1.7, 3.75), (2.6, 3.0))
    arrow((1.7, 1.55), (2.6, 2.5))
    # 分类器
    box((5.5, 3.7), 1.9, 0.95, "Classifier $g$\n(2 classes)", "#c0392b")
    arrow((4.6, 3.1), (5.5, 4.1))
    box((8.1, 3.7), 2.3, 0.95, "Class-balanced\nfocal loss $L_{cls}$", "#8e44ad")
    arrow((7.4, 4.17), (8.1, 4.17))
    # 条件域判别器
    box((5.5, 1.9), 1.9, 0.85, "concat$(f, g)$\n(multilinear)", "#16a085", fs=9)
    arrow((4.6, 2.5), (5.5, 2.35))
    box((8.1, 1.9), 2.3, 0.85, "GRL($\\lambda$) + $D$\n$L_{dom}$", "#27ae60")
    arrow((7.4, 2.32), (8.1, 2.32))
    # 汇总
    box((6.2, 0.2), 2.6, 0.8, "$L = L_{cls} + L_{dom}$", "#2c3e50")
    arrow((9.2, 3.7), (8.6, 1.0), color="#8e44ad")
    arrow((9.2, 1.9), (8.6, 1.0), color="#27ae60")
    ax.set_xlim(0, 11.2); ax.set_ylim(0, 5.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_architecture.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------- Fig: 两个域的示例图片 ----------
def fig_domain_examples():
    random.seed(7)
    def pick(csv_path, label, n):
        rows = []
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r.get("label") == label:
                    rows.append(r["image_path"])
        random.shuffle(rows)
        return rows[:n]

    br = ROOT / "data" / "processed" / "labeled_S.csv"
    ro = ROOT / "data" / "processed" / "test_T.csv"
    br_h = pick(br, "healthy", 2); br_r = pick(br, "rust", 2)
    ro_h = pick(ro, "healthy", 2); ro_r = pick(ro, "rust", 2)

    paths = br_h + br_r + ro_h + ro_r
    titles = ["healthy", "healthy", "rust", "rust"] * 2

    fig, axes = plt.subplots(2, 4, figsize=(9, 5))
    for i, (ax, p, t) in enumerate(zip(axes.flat, paths, titles)):
        img = Image.open(p).convert("RGB")
        img.thumbnail((360, 360))
        ax.imshow(img); ax.axis("off")
        ax.set_title(t, fontsize=10)
    for ax, lab in zip(axes[:, 0], ["BRACOL (lab, source)", "RoCoLe (field, target)"]):
        ax.set_ylabel(lab, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig_domain_examples.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_main(); print("[OK] fig_main_results.png")
    fig_direction(); print("[OK] fig_direction.png")
    fig_gradient(); print("[OK] fig_gradient.png")
    fig_architecture(); print("[OK] fig_architecture.png")
    fig_domain_examples(); print("[OK] fig_domain_examples.png")
    print("ALL FIGURES DONE ->", OUT)
