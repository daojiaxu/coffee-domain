# -*- coding: utf-8 -*-
"""Plot the main results comparison bar chart (Fig. 2) in English.

Uses the final 7-seed results (best-epoch mean +/- std).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams["axes.unicode_minus"] = False

labels = [
    "Oracle\n(P0, in-domain)",
    "CW-DANN\n(ours, unlabeled)",
    "DANN\n(unlabeled)",
    "ImageNet\nfine-tune (P2)",
    "From scratch\n(P1)",
    "SimCLR\n(P4)",
]
means = [96.2, 81.8, 79.0, 70.4, 65.5, 48.0]
stds = [0.0, 3.8, 3.3, 4.1, 0.0, 0.0]  # P0/P1/P4 single seed -> no error bar
colors = ["#8c8c8c", "#c0392b", "#5b8ff9", "#5b8ff9", "#5b8ff9", "#95a5a6"]

x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=200)
ax.bar(x, means, yerr=stds, capsize=5, width=0.62, color=colors,
       edgecolor="black", linewidth=0.8,
       error_kw=dict(elinewidth=1.5, capthick=1.5, ecolor="#333333"))

for i, (m, s) in enumerate(zip(means, stds)):
    txt = f"{m:.1f}%" if s == 0 else f"{m:.1f}±{s:.1f}%"
    ax.text(i, m + (s or 0) + 1.8, txt, ha="center", va="bottom",
            fontsize=10, fontweight="bold")

ax.annotate("+8.6 pp\n(p = 0.016)", xy=(2.0, 80.5), xytext=(2.7, 88.0),
            fontsize=10, fontweight="bold", color="#c0392b", ha="center",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2))

ax.set_ylabel("best test accuracy (%)", fontsize=12)
ax.set_title("Cross-domain coffee leaf disease recognition (BRACOL white background -> RoCoLe field)",
             fontsize=12.5, fontweight="bold")
ax.set_ylim(0, 105)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "outputs", "results_comparison.png")
plt.savefig(out, bbox_inches="tight", facecolor="white")
print("[OK]", os.path.abspath(out))
