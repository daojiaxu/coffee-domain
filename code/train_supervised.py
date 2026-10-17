# -*- coding: utf-8 -*-
"""
train_supervised.py — 监督训练/微调 + 在目标域测试集上评估。

三种初始化模式（--init）：
  scratch    : 随机初始化（对应协议 P1）
  imagenet   : ImageNet 预训练（对应协议 P2/P3，默认）
  ssl        : 加载自监督预训练 backbone（对应协议 P4/P5/P6，需 --ssl-weights）

用法：
  python train_supervised.py --train ../data/processed/labeled_S.csv \
         --test ../data/processed/test_T.csv --init imagenet
  python train_supervised.py --train ../data/processed/labeled_S.csv \
         --test ../data/processed/test_T.csv --init ssl \
         --ssl-weights ../outputs/checkpoints/simclr_backbone.pth
"""
import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "outputs" / "results"

LABELS = ["healthy", "rust"]
L2I = {v: i for i, v in enumerate(LABELS)}

# ImageNet 归一化（ImageNet 预训练模型的标准输入预处理，对 scratch/ssl 也统一使用）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_PIL_TO_TENSOR = transforms.ToTensor()


def load_image(path: str) -> torch.Tensor:
    """用 PIL 读图并转为 [C,H,W] float [0,1] tensor。

    优先读取 256 缓存小图（短边 256，解码快 ~50 倍），缓存缺失时回退读原图。
    """
    p = Path(path)
    parts = p.parts
    if "raw" in parts:
        ds = parts[parts.index("raw") + 1]  # bracol / rocole
        cached = ROOT / "data" / "processed" / "imgs256" / ds / (p.stem + ".jpg")
        if cached.exists():
            p = cached
    return _PIL_TO_TENSOR(Image.open(p).convert("RGB"))


class LabeledDataset(Dataset):
    def __init__(self, csv_path: Path, transform, train: bool):
        self.items: List[Dict] = []
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("label") not in L2I:
                    continue
                self.items.append(row)
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img = load_image(self.items[i]["image_path"])
        if self.train:
            img = self.transform(img)
        else:
            img = transforms.Resize(256)(img)
            img = transforms.CenterCrop(224)(img)
            img = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(img)
        return img, L2I[self.items[i]["label"]]


def train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_model(init: str, ssl_weights: Optional[Path], num_classes: int):
    if init == "imagenet":
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        model = torchvision.models.resnet50(weights=None)
    if init == "ssl":
        if ssl_weights is None or not Path(ssl_weights).exists():
            raise FileNotFoundError("请用 --ssl-weights 指定预训练权重")
        state = torch.load(ssl_weights, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print("[INFO] 已加载自监督 backbone:", ssl_weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    all_pred, all_true = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
            all_pred += pred.cpu().tolist()
            all_true += y.cpu().tolist()
    acc = correct / max(total, 1)
    return acc, all_pred, all_true


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--init", choices=["scratch", "imagenet", "ssl"], default="imagenet")
    p.add_argument("--ssl-weights", default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default="run")
    p.add_argument("--labels", type=str, default="healthy,rust",
                   help="逗号分隔的类别标签（如 healthy,disease）")
    args = p.parse_args()

    global LABELS, L2I
    LABELS = [x.strip() for x in args.labels.split(",")]
    L2I = {v: i for i, v in enumerate(LABELS)}

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = LabeledDataset(Path(args.train), train_transform(), train=True)
    test_ds = LabeledDataset(Path(args.test), None, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    print(f"[INFO] train={len(train_ds)} test={len(test_ds)}")

    model = build_model(args.init, Path(args.ssl_weights) if args.ssl_weights else None,
                        len(LABELS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best = {"test_acc": 0.0, "epoch": -1}
    RES.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
        sched.step()
        acc, _, _ = evaluate(model, test_loader, device)
        print(f"[EPOCH {epoch+1}/{args.epochs}] test_acc={acc:.4f}")
        if acc > best["test_acc"]:
            best = {"test_acc": acc, "epoch": epoch + 1}
            torch.save(model.state_dict(), RES / f"{args.tag}_best.pth")

    # 最终评估
    acc, pred, true = evaluate(model, test_loader, device)
    result = {"init": args.init, "train": args.train, "test": args.test,
              "test_acc": round(acc, 4), "best": best}
    with open(RES / f"{args.tag}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("[DONE] 结果已保存:", RES / f"{args.tag}_metrics.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
