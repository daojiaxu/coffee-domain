# -*- coding: utf-8 -*-
"""Redraw Figure 2: architecture of adversarial methods (clean, publication style)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patheffects as pe
from pathlib import Path

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
})

W, H = 12.4, 6.6
fig, ax = plt.subplots(figsize=(W, H), dpi=300)
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

C = dict(
    src_face="#DCE9F8", src_edge="#2E5A88",
    tgt_face="#EFEFEF", tgt_edge="#7A7A7A",
    bb_face="#C9DCEF", bb_edge="#1F4E79",
    cls_face="#DFF0DC", cls_edge="#3E7A3A",
    ce_face="#FFF2CC", ce_edge="#B08A1E",
    cb_face="#FFE0B3", cb_edge="#B5651D",
    map_face="#EDE7F6", map_edge="#5E35B1",
    grl_face="#FFFFFF", grl_edge="#5E35B1",
    dis_face="#F8D7DA", dis_edge="#A03038",
    dom_face="#FFF2CC", dom_edge="#B08A1E",
    tot_face="#E8E8E8", tot_edge="#444444",
)

def box(x, y, w, h, text, fc, ec, fs=11, weight="normal", dashed=False, lw=1.6):
    b = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.06,rounding_size=0.14",
                       linewidth=lw, edgecolor=ec, facecolor=fc,
                       linestyle="--" if dashed else "-")
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight, color="#1A1A1A", linespacing=1.35)
    return (x, y, w, h)

def arrow(p0, p1, color="#444444", lw=1.8, dashed=False, curved=0.0, grad=False):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14,
                        linewidth=lw, color=color,
                        linestyle=(0, (5, 3)) if dashed else "-",
                        connectionstyle=f"arc3,rad={curved}",
                        shrinkA=2, shrinkB=2, zorder=3)
    if grad:
        a.set_path_effects([pe.withStroke(linewidth=lw + 2.2, foreground="white")])
    ax.add_patch(a)

# ---- boxes -------------------------------------------------------------
src  = box(0.35, 4.55, 2.05, 1.05, "Source $x^s$\n(labeled)",   C["src_face"], C["src_edge"], fs=11.5, weight="bold")
tgt  = box(0.35, 1.15, 2.05, 1.05, "Target $x^t$\n(unlabeled)", C["tgt_face"], C["tgt_edge"], fs=11.5, weight="bold")
bb   = box(3.15, 2.85, 2.55, 1.35, "Feature extractor $f$\nResNet-50\n(ImageNet init)", C["bb_face"], C["bb_edge"], fs=11.5, weight="bold")

cls  = box(6.55, 4.85, 2.30, 0.95, "Classifier $g$\n(linear head)", C["cls_face"], C["cls_edge"], fs=11.5, weight="bold")
ce   = box(9.75, 5.05, 2.25, 0.95, "Cross-entropy\n(DANN, CDAN)", C["ce_face"], C["ce_edge"], fs=10.5)
cb   = box(9.75, 3.85, 2.25, 0.95, "Class-balanced\nfocal loss (CW-DANN)", C["cb_face"], C["cb_edge"], fs=10.5)

mp   = box(6.55, 1.55, 2.30, 0.95, "Multilinear map\n$g \\otimes f$", C["map_face"], C["map_edge"], fs=11)
grl_c = (9.30, 2.02)
dis  = box(9.95, 1.55, 2.05, 0.95, "Discriminator $D$", C["dis_face"], C["dis_edge"], fs=11, weight="bold")
dom  = box(9.95, 0.25, 2.05, 0.85, "$L_{\\mathrm{dom}}$", C["dom_face"], C["dom_edge"], fs=11.5, weight="bold")

tot  = box(4.30, 0.15, 3.60, 0.85, "$\\mathcal{L} \\;=\\; \\mathcal{L}_{\\mathrm{cls}} \\;+\\; \\lambda\\,\\mathcal{L}_{\\mathrm{dom}}$",
           C["tot_face"], C["tot_edge"], fs=13, weight="bold")

# ---- arrows ------------------------------------------------------------
arrow((2.40, 5.05), (3.20, 3.95), curved=0.18)              # src -> f
arrow((2.40, 1.75), (3.20, 3.10), curved=-0.18, dashed=True) # tgt -> f
arrow((5.70, 3.75), (6.60, 5.15), curved=-0.15)              # f -> g
arrow((8.85, 5.32), (9.75, 5.52))                            # g -> CE
arrow((8.85, 5.05), (9.75, 4.50), curved=-0.12)              # g -> focal
arrow((5.70, 3.20), (6.60, 2.15), curved=0.15)               # f -> map
arrow((8.85, 2.02), (8.98, 2.02), lw=0)                      # spacer (none)
arrow((8.85, 2.02), (9.02, 2.02), lw=1.8)                    # map -> GRL
arrow((9.58, 2.02), (9.95, 2.02), lw=1.8)                    # GRL -> D
arrow((10.97, 1.55), (10.97, 1.10))                          # D -> L_dom
arrow((9.95, 0.65), (7.90, 0.57), curved=0.0)                # L_dom -> total

# GRL annotation (gradient reversal, dashed backward arrow)
circ = Circle(grl_c, 0.30, facecolor=C["grl_face"], edgecolor=C["grl_edge"],
              linewidth=1.8, linestyle="--", zorder=4)
ax.add_patch(circ)
ax.text(grl_c[0], grl_c[1], "GRL", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#3E2A80", zorder=5)
ax.annotate("$\\lambda(p)$ schedule, $-\\lambda$ on backward pass",
            xy=(9.55, 2.36), xytext=(10.45, 3.35),
            fontsize=9.5, color="#5E35B1", ha="center",
            arrowprops=dict(arrowstyle="->", color="#5E35B1", lw=1.2,
                            linestyle="--", connectionstyle="arc3,rad=-0.25"))

# classifier loss arrow down to total
arrow((7.70, 4.85), (6.55, 1.00), curved=0.18, color="#3E7A3A")
ax.text(7.35, 3.75, "$\\mathcal{L}_{\\mathrm{cls}}$", fontsize=12,
        color="#3E7A3A", ha="left", va="center")

# legend note
ax.text(0.35, 0.62, "dashed arrow: unlabeled\ntarget stream", 
        fontsize=9, color="#666666", ha="left", va="center", style="italic")

fig.savefig(str(Path(__file__).resolve().parent.parent / "figures" / "fig_architecture.png"),
            bbox_inches="tight", facecolor="white")
print("saved")
