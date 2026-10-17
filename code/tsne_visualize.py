# -*- coding: utf-8 -*-
"""
tsne_visualize.py — 可视化跨域特征分布（域对齐对比）。

对比 P2（ImageNet 微调，无域适应）/ DANN / CW-DANN 三个模型在
「源域(BRACOL 白底) + 目标域(RoCoLe 田间)」上的特征 t-SNE 分布。

可视化约定：
  颜色 = 类别（蓝=healthy，红=rust）
  形状 = 域（圆 o = 源域，三角 ^ = 目标域）
预期：无域适应时同类跨域分离（同色圆与三角分开）；
     域适应后同类跨域聚集（同色圆与三角重叠）。

同时输出 proxy A-distance：线性域分类器 5 折准确率，
50% = 完全对齐（域不可分），越高 = 域差距越大。

用法：
  python code/tsne_visualize.py [--cpu]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from train_cwdann import (load_image, eval_transform, LabeledDataset,
                          UnlabeledDataset, CW_DANN)

# 中文字体
for c in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]:
    if os.path.exists(c):
        font_manager.fontManager.addfont(c)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=c).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_p2(ckpt_path, device):
    """加载 P2 模型，fc 换成 Identity 以提取 2048 维特征。"""
    model = torchvision.models.resnet50(weights=None)
    model.fc = nn.Identity()
    state = torch.load(ckpt_path, map_location="cpu")
    state = {k: v for k, v in state.items() if not k.startswith("fc.")}
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def load_dann(ckpt_path, conditional, device):
    model = CW_DANN(num_classes=2, conditional=conditional)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model.to(device).eval()


@torch.no_grad()
def extract(model, ds, device):
    """提取特征与标签。返回 feats [N,2048], labels [N]。"""
    feats, labels = [], []
    for i in range(len(ds)):
        img, y = ds[i]
        img = img.unsqueeze(0).to(device)
        if hasattr(model, "feature"):
            f = model.feature(img)
        else:
            f = model(img)  # P2: fc=Identity，直接输出特征
        feats.append(f.squeeze(0).cpu().numpy())
        labels.append(y)
    return np.stack(feats), np.array(labels)


def proxy_a_distance(f_src, f_tgt, seed=42):
    """proxy A-distance：线性域分类器 5 折准确率。
    2*(1-2*err) 为经典 A-distance 的 proxy，这里直接报告域分类准确率。
    50% = 域不可分（对齐好），越高 = 域差距越大。"""
    X = np.concatenate([f_src, f_tgt], axis=0)
    y = np.concatenate([np.zeros(len(f_src)), np.ones(len(f_tgt))])
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    acc = cross_val_score(clf, X, y, cv=5, scoring="accuracy", n_jobs=1).mean()
    return acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cpu", action="store_true", help="用 CPU 提取特征（避免与训练抢 GPU）")
    args = p.parse_args()
    device = torch.device("cpu") if args.cpu else DEFAULT_DEVICE

    src_ds = LabeledDataset(ROOT / "data/processed/labeled_S.csv", eval_transform())
    tgt_ds = LabeledDataset(ROOT / "data/processed/test_T.csv", eval_transform())
    src_items = [src_ds[i] for i in range(len(src_ds))]
    tgt_items = [tgt_ds[i] for i in range(len(tgt_ds))]

    models = {
        "P2 (ImageNet FT)": load_p2(ROOT / "outputs/results/P2_imagenet_best.pth", device),
        "DANN": load_dann(ROOT / "outputs/results/P7_dann_s42_best.pth", False, device),
        "CW-DANN (ours)": load_dann(ROOT / "outputs/results/CW-DANN_s42_best.pth", True, device),
    }

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), dpi=150)
    report = {}

    for ax, (name, model) in zip(axes, models.items()):
        f_src, y_src = extract(model, src_items, device)
        f_tgt, y_tgt = extract(model, tgt_items, device)

        # proxy A-distance（域可分性）
        adist = proxy_a_distance(f_src, f_tgt)
        report[name] = {"proxy_a_distance_acc": round(float(adist), 4)}

        # t-SNE
        X = np.concatenate([f_src, f_tgt], axis=0)
        emb = TSNE(n_components=2, perplexity=30, random_state=42,
                   init="pca", learning_rate="auto").fit_transform(X)
        e_src, e_tgt = emb[:len(f_src)], emb[len(f_src):]

        # 画 4 组：类别=颜色，域=形状
        for cls_id, cls_name, color in [(0, "healthy", "#4c72b0"), (1, "rust", "#c0392b")]:
            m_src = y_src == cls_id
            m_tgt = y_tgt == cls_id
            ax.scatter(e_src[m_src, 0], e_src[m_src, 1], s=12, c=color,
                       marker="o", alpha=0.6, label=f"Source {cls_name}", rasterized=True)
            ax.scatter(e_tgt[m_tgt, 0], e_tgt[m_tgt, 1], s=18, c=color,
                       marker="^", alpha=0.75, label=f"Target {cls_name}",
                       edgecolors="black", linewidths=0.3, rasterized=True)

        ax.set_title(f"{name}\nDomain cls. acc. = {adist*100:.1f}% (50% = aligned)",
                     fontsize=12.5, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=8, loc="best", ncol=2, markerscale=1.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("t-SNE of cross-domain features (color = class, circle = source, triangle = target)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = ROOT / "outputs/tsne_comparison.png"
    plt.savefig(out, bbox_inches="tight")
    print("已保存:", out)

    import json
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
