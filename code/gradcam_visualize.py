# -*- coding: utf-8 -*-
"""
gradcam_visualize.py — Grad-CAM 可视化（P2 vs DANN vs CB-DANN 的注意力对比）。

对目标域 RoCoLe 测试集选定的 healthy/rust 图，分别用三个模型的分类头做 Grad-CAM，
展示：纯监督微调（P2）与域对抗方法（DANN / CB-DANN）关注的区域差异，
验证「域对抗让模型关注病害区域而非背景」的可解释性假设。

用法：python gradcam_visualize.py --cpu    # CPU 模式不抢训练显存
"""
import os
import argparse
import csv
import random
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "outputs" / "results"
OUT = ROOT / "outputs"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 中文字体
for _c in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]:
    if os.path.exists(_c):
        font_manager.fontManager.addfont(_c)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=_c).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False


class CW_DANN(nn.Module):
    """与 train_cwdann.py 一致的模型结构（用于加载 checkpoint）。"""

    def __init__(self, num_classes: int = 2, conditional: bool = False):
        super().__init__()
        self.conditional = conditional
        self.feature = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = self.feature.fc.in_features
        self.feature.fc = nn.Identity()
        self.classifier = nn.Linear(in_features, num_classes)
        dom_in = in_features + num_classes if conditional else in_features
        self.domain_classifier = nn.Sequential(
            nn.Linear(dom_in, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x, lambda_=0.0):
        f = self.feature(x)
        logits = self.classifier(f)
        return logits, None


def load_p2(ckpt):
    m = torchvision.models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return m


def load_dann(ckpt, conditional=False):
    m = CW_DANN(num_classes=2, conditional=conditional)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return m


def eval_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_image(path):
    p = Path(path)
    parts = p.parts
    if "raw" in parts:
        ds = parts[parts.index("raw") + 1]
        cached = ROOT / "data" / "processed" / "imgs256" / ds / (p.stem + ".jpg")
        if cached.exists():
            p = cached
    return Image.open(p).convert("RGB")


def get_target_layer(model):
    """返回 resnet50 的最后一个卷积层 + 其输入特征获取方式。"""
    if hasattr(model, "feature"):  # CW_DANN
        return model.feature.layer4[-1].conv3
    return model.layer4[-1].conv3


def gradcam(model, img_tensor, target_layer, device):
    """返回 (cam, logits) 其中 cam 是 7x7 numpy。"""
    model.eval()
    feats = {}
    grads = {}

    def fwd_hook(m, i, o):
        feats["v"] = o

    def bwd_hook(m, gi, go):
        grads["v"] = go[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    x = img_tensor.unsqueeze(0).to(device)
    if hasattr(model, "feature"):
        logits, _ = model(x, 0.0)
    else:
        logits = model(x)
    c = logits.argmax(1).item()
    model.zero_grad()
    logits[0, c].backward()

    h1.remove(); h2.remove()
    w = grads["v"].mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
    cam = F.relu((w * feats["v"]).sum(dim=1, keepdim=True))  # [1,1,7,7]
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
    return cam, logits[0].detach().cpu()


def load_test_items():
    """选 healthy / rust 各 3 张目标域测试图。"""
    items = []
    with open(ROOT / "data" / "processed" / "test_T.csv", "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for label in ["healthy", "rust"]:
        sub = [r for r in rows if r["label"] == label]
        random.seed(42)
        random.shuffle(sub)
        for r in sub[:3]:
            items.append(r)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    models = {
        "P2 (ImageNet FT)": load_p2(RES / "P2_imagenet_best.pth"),
        "DANN": load_dann(RES / "P7_dann_s42_best.pth", conditional=False),
        "CW-DANN (ours)": load_dann(RES / "CW-DANN_s42_best.pth", conditional=True),
    }
    for name, m in models.items():
        m.to(device).eval()

    items = load_test_items()
    tf = eval_transform()

    fig, axes = plt.subplots(len(items), len(models) + 1,
                             figsize=(4.2 * (len(models) + 1), 2.6 * len(items)), dpi=150)
    for i, item in enumerate(items):
        raw = load_image(item["image_path"])
        # 原图
        axes[i][0].imshow(raw)
        axes[i][0].set_title(f"Original\n{item['label']}", fontsize=9)
        axes[i][0].axis("off")
        # 各模型 Grad-CAM
        img = tf(raw)
        for j, (name, m) in enumerate(models.items(), start=1):
            layer = get_target_layer(m)
            cam, logits = gradcam(m, img, layer, device)
            axes[i][j].imshow(raw)
            axes[i][j].imshow(cam, cmap="jet", alpha=0.45)
            pred = "rust" if logits.argmax().item() == 1 else "healthy"
            axes[i][j].set_title(f"{name}\npred={pred}", fontsize=8)
            axes[i][j].axis("off")

    fig.suptitle("Grad-CAM: does domain adaptation focus more on disease regions than background?",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = OUT / "gradcam_comparison.png"
    plt.savefig(out, bbox_inches="tight")
    print("[OK] 已保存:", out)


if __name__ == "__main__":
    main()
