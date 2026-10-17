# -*- coding: utf-8 -*-
"""
train_dann.py — DANN 域对抗适应（源域有标注 + 目标域无标注）。

用法：
  python train_dann.py --src data/processed/labeled_S.csv \
         --tgt data/processed/ssl_pool_T.csv \
         --test data/processed/test_T.csv --epochs 50 --tag P7_dann

说明：
  - 特征提取器用 ImageNet 预训练 ResNet50（与 P2 公平对比：P2 是 ImageNet 微调，
    DANN 是 ImageNet 微调 + 域对抗，看域对抗能否进一步缩小域差距）；
  - 域判别器 + 梯度反转层（GRL）使特征域不变；
  - 目标域完全无标注（只用于域判别，不用于标签分类）。
"""
import argparse
import csv
import json
import math
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

LABELS = ["healthy", "rust"]
L2I = {v: i for i, v in enumerate(LABELS)}

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
    """源域（有标签）。"""

    def __init__(self, csv_path: Path, transform):
        self.items: List[Dict] = []
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("label") in L2I:
                    self.items.append(row)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img = self.transform(load_image(self.items[i]["image_path"]))
        return img, L2I[self.items[i]["label"]]


class UnlabeledDataset(Dataset):
    """目标域（无标签，仅用于域判别）。"""

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


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_):
    return GradientReversalFunction.apply(x, lambda_)


class DANN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.feature = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = self.feature.fc.in_features
        self.feature.fc = nn.Identity()
        self.classifier = nn.Linear(in_features, num_classes)
        self.domain_classifier = nn.Sequential(
            nn.Linear(in_features, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x, lambda_):
        f = self.feature(x)
        class_logits = self.classifier(f)
        reversed_f = grad_reverse(f, lambda_)
        domain_logits = self.domain_classifier(reversed_f)
        return class_logits, domain_logits


def lambda_schedule(p: float) -> float:
    """DANN 论文的渐进式 λ：从 0 平滑升到 1。"""
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x, 0.0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="源域标注索引（labeled_S.csv）")
    p.add_argument("--tgt", required=True, help="目标域无标注索引（ssl_pool_T.csv）")
    p.add_argument("--test", required=True, help="目标域测试集（test_T.csv）")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default="P7_dann")
    p.add_argument("--labels", type=str, default="healthy,rust",
                   help="逗号分隔的类别标签（如 healthy,disease）")
    args = p.parse_args()

    global LABELS, L2I
    LABELS = [x.strip() for x in args.labels.split(",")]
    L2I = {v: i for i, v in enumerate(LABELS)}

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_ds = LabeledDataset(Path(args.src), train_transform())
    tgt_ds = UnlabeledDataset(Path(args.tgt), train_transform())
    test_ds = LabeledDataset(Path(args.test), eval_transform())

    src_loader = DataLoader(src_ds, batch_size=args.batch, shuffle=True,
                            num_workers=0, drop_last=False)
    tgt_loader = DataLoader(tgt_ds, batch_size=args.batch, shuffle=True,
                            num_workers=0, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    print(f"[INFO] src={len(src_ds)} tgt={len(tgt_ds)} test={len(test_ds)}")

    model = DANN(len(LABELS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()

    steps_per_epoch = min(len(src_loader), len(tgt_loader))
    total_steps = steps_per_epoch * args.epochs
    global_step = 0

    RES.mkdir(parents=True, exist_ok=True)
    best = {"test_acc": 0.0, "epoch": -1}

    for epoch in range(args.epochs):
        model.train()
        running_cls = 0.0
        running_dom = 0.0
        n = 0
        for (xs, ys), xt in zip(src_loader, tgt_loader):
            xs, ys = xs.to(device), ys.to(device)
            xt = xt.to(device)

            lam = lambda_schedule(global_step / max(total_steps - 1, 1))

            class_logits_s, domain_logits_s = model(xs, lam)
            _, domain_logits_t = model(xt, lam)

            loss_cls = ce(class_logits_s, ys)
            domain_labels_s = torch.zeros(domain_logits_s.shape[0], 1, device=device)
            domain_labels_t = torch.ones(domain_logits_t.shape[0], 1, device=device)
            loss_dom = bce(domain_logits_s, domain_labels_s) + bce(domain_logits_t, domain_labels_t)
            loss = loss_cls + loss_dom

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            running_cls += loss_cls.item()
            running_dom += loss_dom.item()
            n += 1

        acc = evaluate(model, test_loader, device)
        print(f"[EPOCH {epoch+1}/{args.epochs}] test_acc={acc:.4f} "
              f"cls={running_cls/n:.3f} dom={running_dom/n:.3f} lam={lam:.3f}")
        if acc > best["test_acc"]:
            best = {"test_acc": acc, "epoch": epoch + 1}
            torch.save(model.state_dict(), RES / f"{args.tag}_best.pth")

    acc = evaluate(model, test_loader, device)
    result = {"init": "imagenet+dann", "src": args.src, "tgt": args.tgt,
              "test": args.test, "test_acc": round(acc, 4), "best": best}
    with open(RES / f"{args.tag}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("[DONE] 结果已保存:", RES / f"{args.tag}_metrics.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
