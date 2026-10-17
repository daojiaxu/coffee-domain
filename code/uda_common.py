# -*- coding: utf-8 -*-
"""
uda_common.py — UDA 基线方法（DeepCORAL / DAN / CDAN / SHOT）共享的数据加载、
模型骨架、评估工具与结果保存。

与 train_dann.py / train_cwdann.py 保持完全一致的数据流水线、评估协议
（best-epoch / fixed-epoch）与 metrics.json 字段格式，确保公平对比：
  - 相同的 ImageNet 预训练 ResNet50 特征提取器（2048 维）
  - 相同的 256 缓存小图读取、训练/评估增强
  - 相同的 best-epoch（目标域测试集选最大准确率）+ 末轮 fixed-epoch 双协议
"""
import csv
import json
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "outputs" / "results"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_PIL_TO_TENSOR = transforms.ToTensor()


def load_image(path: str) -> torch.Tensor:
    """PIL 读图，优先读 256 缓存小图。"""
    p = Path(path)
    parts = p.parts
    if "raw" in parts:
        ds = parts[parts.index("raw") + 1]
        cached = ROOT / "data" / "processed" / "imgs256" / ds / (p.stem + ".jpg")
        if cached.exists():
            p = cached
    return _PIL_TO_TENSOR(Image.open(p).convert("RGB"))


def train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class LabeledDataset(Dataset):
    """有标签数据集（源域 / 目标测试集）。labels 决定类别过滤与索引映射。"""

    def __init__(self, csv_path: Path, transform, labels: List[str]):
        self.l2i = {v: i for i, v in enumerate(labels)}
        self.items: List[Dict] = []
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("label") in self.l2i:
                    self.items.append(row)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img = self.transform(load_image(self.items[i]["image_path"]))
        return img, self.l2i[self.items[i]["label"]]


class UnlabeledDataset(Dataset):
    """无标签数据集（目标域无标注池）。"""

    def __init__(self, csv_path: Path, transform):
        self.paths: List[str] = []
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                self.paths.append(row["image_path"])
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(load_image(self.paths[i]))


def make_backbone():
    """ImageNet 预训练 ResNet50 特征提取器（fc 替换为 Identity，输出 2048 维）。"""
    model = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Identity()
    return model, in_features


def evaluate(model, loader, device) -> float:
    """评估目标域测试集准确率。model 需实现 predict(x) -> logits。"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model.predict(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def save_metrics(tag, method, src, tgt, test, test_acc, best, extra=None):
    """统一保存 metrics.json，字段与 DANN/CW-DANN 一致。"""
    result = {"method": method, "src": src, "tgt": tgt, "test": test,
              "test_acc": round(test_acc, 4), "best": best}
    if extra:
        result.update(extra)
    RES.mkdir(parents=True, exist_ok=True)
    with open(RES / f"{tag}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("[DONE] 结果已保存:", RES / f"{tag}_metrics.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))
