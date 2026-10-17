# -*- coding: utf-8 -*-
"""生成 Agriculture 投稿 Graphical Abstract (2:1, 2400x1200)"""
import csv, random
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = str(ROOT / "graphical_abstract.png")

def pick(csv_path, label, n, seed):
    rng = random.Random(seed)
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("label") == label:
                rows.append(r["image_path"])
    rng.shuffle(rows)
    return rows[:n]

def load_square(p, size=320):
    img = Image.open(p).convert("RGB")
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    return img.resize((size, size))

br = ROOT / "data" / "processed" / "labeled_S.csv"
ro = ROOT / "data" / "processed" / "test_T.csv"
br_h = pick(br, "healthy", 1, 7)[0]
br_r = pick(br, "rust", 1, 8)[0]
ro_h = pick(ro, "healthy", 1, 9)[0]
ro_r = pick(ro, "rust", 1, 10)[0]
# 手动选定更具代表性的示例图（锈斑/叶片更典型清晰）
br_r = str(ROOT / "data" / "raw" / "bracol" / "images" / "899.jpg")
ro_h = str(ROOT / "data" / "raw" / "rocole" / "Photos" / "C11P14H1.jpg")

fig = plt.figure(figsize=(12, 6))

# ---------- 顶部标题 ----------
fig.text(0.5, 0.945,
         "Cross-Domain Coffee Leaf Disease Recognition: Lab \u2192 Field",
         ha="center", va="center", fontsize=19, fontweight="bold", color="#1a1a1a")

# ---------- 底部结论条 ----------
fig.patches.append(FancyBboxPatch((0.02, 0.015), 0.96, 0.135,
    transform=fig.transFigure, boxstyle="round,pad=0.005,rounding_size=0.012",
    linewidth=1.5, edgecolor="#8a8a8a", facecolor="#f6f6f6"))
fig.text(0.5, 0.105, "Key findings", ha="center", va="center",
         fontsize=11, fontweight="bold", color="#555555")
fig.text(0.185, 0.055,
         "\u2460 UDA works lab\u2192field\n(all 6 methods, p \u2264 0.047)",
         ha="center", va="center", fontsize=10, color="#222222")
fig.text(0.5, 0.055,
         "\u2461 Direction-asymmetric:\nno gain field\u2192lab",
         ha="center", va="center", fontsize=10, color="#222222")
fig.text(0.815, 0.055,
         "\u2462 Gain grows with task difficulty;\nSHOT most stable label-free",
         ha="center", va="center", fontsize=10, color="#222222")

# ---------- 左：Source (BRACOL) ----------
fig.patches.append(FancyBboxPatch((0.02, 0.22), 0.30, 0.60,
    transform=fig.transFigure, boxstyle="round,pad=0.008,rounding_size=0.02",
    linewidth=2.5, edgecolor="#2c6fbb", facecolor="#eef4fb"))
fig.text(0.17, 0.775, "Source domain: BRACOL", ha="center", va="center",
         fontsize=14, fontweight="bold", color="#1f4e79")
fig.text(0.17, 0.728, "labeled \u00b7 white background (lab)", ha="center", va="center",
         fontsize=11, color="#3a6ea5")

ax1 = fig.add_axes([0.055, 0.42, 0.10, 0.20], facecolor="none"); ax1.imshow(load_square(br_h)); ax1.axis("off"); ax1.set_zorder(3)
ax2 = fig.add_axes([0.175, 0.42, 0.10, 0.20], facecolor="none"); ax2.imshow(load_square(br_r)); ax2.axis("off"); ax2.set_zorder(3)
fig.text(0.105, 0.385, "Healthy", ha="center", va="center", fontsize=10.5, color="#1f4e79")
fig.text(0.225, 0.385, "Rust", ha="center", va="center", fontsize=10.5, color="#1f4e79")

# ---------- 右：Target (RoCoLe) ----------
fig.patches.append(FancyBboxPatch((0.68, 0.22), 0.30, 0.60,
    transform=fig.transFigure, boxstyle="round,pad=0.008,rounding_size=0.02",
    linewidth=2.5, edgecolor="#2e7d32", facecolor="#eef7ef"))
fig.text(0.83, 0.775, "Target domain: RoCoLe", ha="center", va="center",
         fontsize=14, fontweight="bold", color="#1b5e20")
fig.text(0.83, 0.728, "unlabeled \u00b7 natural field background", ha="center", va="center",
         fontsize=11, color="#2e7d32")

ax3 = fig.add_axes([0.725, 0.42, 0.10, 0.20], facecolor="none"); ax3.imshow(load_square(ro_h)); ax3.axis("off"); ax3.set_zorder(3)
ax4 = fig.add_axes([0.845, 0.42, 0.10, 0.20], facecolor="none"); ax4.imshow(load_square(ro_r)); ax4.axis("off"); ax4.set_zorder(3)
fig.text(0.775, 0.385, "Healthy", ha="center", va="center", fontsize=10.5, color="#1b5e20")
fig.text(0.895, 0.385, "Rust", ha="center", va="center", fontsize=10.5, color="#1b5e20")

# ---------- 中：UDA ----------
fig.text(0.5, 0.665, "Unsupervised Domain", ha="center", va="center",
         fontsize=14, fontweight="bold", color="#6a1b9a")
fig.text(0.5, 0.618, "Adaptation", ha="center", va="center",
         fontsize=14, fontweight="bold", color="#6a1b9a")
fig.patches.append(FancyArrowPatch((0.335, 0.50), (0.665, 0.50),
    transform=fig.transFigure, arrowstyle="-|>", mutation_scale=34,
    linewidth=3.5, color="#6a1b9a"))
fig.text(0.5, 0.455, "6 methods benchmarked:", ha="center", va="center",
         fontsize=10.5, color="#555555")
fig.text(0.5, 0.405, "DANN  \u00b7  CW-DANN  \u00b7  Deep CORAL", ha="center", va="center",
         fontsize=10.5, color="#333333")
fig.text(0.5, 0.355, "DAN  \u00b7  CDAN  \u00b7  SHOT", ha="center", va="center",
         fontsize=10.5, color="#333333")

fig.savefig(OUT, dpi=200, facecolor="white")
print("Saved:", OUT)
