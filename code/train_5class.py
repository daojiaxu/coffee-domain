# -*- coding: utf-8 -*-
"""
train_5class.py — BRACOL 5 类细粒度分类（域内多分类）。

用于验证 CB-DANN 的 focal loss 组件（类别平衡）在多类、更严重不平衡场景下是否依然有效。
对比：标准 CE vs 类别平衡 Focal Loss（--focal），各 3 种子。

指标：accuracy + balanced accuracy（各类 recall 均值）+ macro-F1（5 类不平衡，后两者更可靠）。

用法：
  python train_5class.py --train data/processed/bracol_5class_train.csv \
         --test data/processed/bracol_5class_test.csv --tag b5_ce_s42
  python train_5class.py --train data/processed/bracol_5class_train.csv \
         --test data/processed/bracol_5class_test.csv --focal --tag b5_focal_s42
"""
import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
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
    def __init__(self, csv_path: Path, transform, l2i: Dict[str, int]):
        self.items: List[Dict] = []
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("label") in l2i:
                    self.items.append(row)
        self.transform = transform
        self.l2i = l2i

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img = self.transform(load_image(self.items[i]["image_path"]))
        return img, self.l2i[self.items[i]["label"]]


def focal_loss(logits, targets, gamma: float = 2.0, alpha=None):
    ce = F.cross_entropy(logits, targets, weight=alpha, reduction="none")
    pt = torch.exp(-ce)
    return (((1.0 - pt) ** gamma) * ce).mean()


def evaluate(model, loader, device, num_classes):
    model.eval()
    correct, total = 0, 0
    per_class_correct = [0] * num_classes
    per_class_total = [0] * num_classes
    all_pred, all_true = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
            for p, t in zip(pred.cpu().tolist(), y.cpu().tolist()):
                per_class_total[t] += 1
                if p == t:
                    per_class_correct[t] += 1
            all_pred += pred.cpu().tolist()
            all_true += y.cpu().tolist()

    acc = correct / max(total, 1)
    recalls = [per_class_correct[c] / max(per_class_total[c], 1) for c in range(num_classes)]
    balanced_acc = sum(recalls) / num_classes

    # macro-F1（手算，避免 sklearn 依赖）
    tp, fp, fn = [0] * num_classes, [0] * num_classes, [0] * num_classes
    for p, t in zip(all_pred, all_true):
        if p == t:
            tp[p] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    f1s = []
    for c in range(num_classes):
        prec = tp[c] / max(tp[c] + fp[c], 1)
        rec = tp[c] / max(tp[c] + fn[c], 1)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-9))
    macro_f1 = sum(f1s) / num_classes

    return {"acc": acc, "balanced_acc": balanced_acc, "macro_f1": macro_f1}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default="b5")
    p.add_argument("--focal", action="store_true", help="启用类别平衡 focal loss")
    p.add_argument("--gamma", type=float, default=2.0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 从 train CSV 动态构建类别映射（5 类：healthy/miner/rust/phoma/cercospora）
    labels = []
    with open(args.train, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["label"] not in labels:
                labels.append(row["label"])
    labels.sort()
    l2i = {v: i for i, v in enumerate(labels)}
    print(f"[INFO] 类别 {len(labels)} 个: {labels}")

    train_ds = LabeledDataset(Path(args.train), train_transform(), l2i)
    test_ds = LabeledDataset(Path(args.test), eval_transform(), l2i)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    # 类别权重（inverse frequency）
    counts = [0] * len(labels)
    for row in train_ds.items:
        counts[l2i[row["label"]]] += 1
    total = sum(counts)
    alpha = torch.tensor([total / (len(labels) * c) for c in counts],
                         dtype=torch.float32, device=device)
    print(f"[INFO] train={len(train_ds)} test={len(test_ds)}")
    print(f"[INFO] 类别计数={counts} alpha={alpha.cpu().tolist()}")

    model = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(labels))
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ce = nn.CrossEntropyLoss()

    RES.mkdir(parents=True, exist_ok=True)
    best = {"balanced_acc": 0.0, "epoch": -1}
    best_full = None

    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if args.focal:
                loss = focal_loss(logits, y, gamma=args.gamma, alpha=alpha)
            else:
                loss = ce(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        m = evaluate(model, test_loader, device, len(labels))
        print(f"[EPOCH {epoch+1}/{args.epochs}] acc={m['acc']:.4f} "
              f"bal_acc={m['balanced_acc']:.4f} macro_f1={m['macro_f1']:.4f}")
        if m["balanced_acc"] > best["balanced_acc"]:
            best = {"balanced_acc": m["balanced_acc"], "epoch": epoch + 1}
            best_full = m
            torch.save(model.state_dict(), RES / f"{args.tag}_best.pth")

    final_m = evaluate(model, test_loader, device, len(labels))
    result = {"task": "5class", "labels": labels, "train": args.train, "test": args.test,
              "focal": args.focal, "gamma": args.gamma,
              "best": {"balanced_acc": round(best["balanced_acc"], 4),
                       "epoch": best["epoch"], **{k: round(v, 4) for k, v in best_full.items()}},
              "final": {k: round(v, 4) for k, v in final_m.items()},
              "class_weights": alpha.cpu().tolist()}
    with open(RES / f"{args.tag}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("[DONE]", RES / f"{args.tag}_metrics.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
