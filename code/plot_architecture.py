# -*- coding: utf-8 -*-
"""Plot the CB-DANN architecture diagram (Fig. 1) in English."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")


def box(x, y, w, h, text, fc, fs=9, ec="black", tc="white", lw=1.2):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                       facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight="bold", linespacing=1.4)


def arrow(x1, y1, x2, y2, color="black", lw=1.6, style="-|>"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                        mutation_scale=14, color=color, linewidth=lw)
    ax.add_patch(a)


c_in = "#34495e"
c_feat = "#2471a3"
c_cls = "#c0392b"
c_dom = "#1e8449"
c_loss = "#8e44ad"

box(0.3, 2.4, 1.5, 1.2, "Input image x\n(224x224)", c_in, fs=9)
box(2.5, 2.2, 2.2, 1.6, "ResNet-50\nfeature extractor f\n(ImageNet pretrained)", c_feat, fs=8.5)
box(5.6, 3.7, 1.9, 1.0, "Classifier g\n(Linear -> 2 classes)", c_cls, fs=8.5)
box(8.0, 3.7, 1.7, 1.0, "Class-Balanced\nFocal Loss\nL_cls", c_loss, fs=8.5)
box(5.6, 0.9, 1.9, 1.0, "GRL(lambda)\ngradient reversal", c_dom, fs=9)
box(8.0, 0.9, 1.7, 1.0, "Domain\ndiscriminator D\nL_dom", c_dom, fs=8.5)

arrow(1.8, 3.0, 2.5, 3.0)
arrow(4.7, 3.0, 5.6, 4.2)
arrow(7.5, 4.2, 8.0, 4.2)
arrow(4.7, 3.0, 5.6, 1.4)
arrow(7.5, 1.4, 8.0, 1.4)

box(3.0, 0.5, 1.4, 0.7, "Target x_t\n(unlabeled)", "#7f8c8d", fs=7.5, ec="#5d6d7e", tc="white")
arrow(3.7, 1.2, 5.6, 1.4, color="#7f8c8d", lw=1.4)

box(8.6, 2.4, 1.2, 1.2, "L =\nL_cls\n+L_dom", "#212f3d", fs=9)
arrow(8.85, 3.7, 9.0, 3.6, color=c_loss, lw=1.4)
arrow(8.85, 1.4, 9.0, 2.4, color=c_dom, lw=1.4)

ax.text(0.3, 5.7, "Source domain (white-background BRACOL, labeled)", fontsize=8.5, color="#5d6d7e")
ax.text(5.6, 0.15, "Core component: Class-Balanced Focal Loss (reweights healthy 272 vs. rust 531)",
        fontsize=8.5, color=c_cls, fontweight="bold")

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "outputs", "cbdann_architecture.png")
plt.savefig(out, bbox_inches="tight", facecolor="white")
print("[OK]", os.path.abspath(out))
