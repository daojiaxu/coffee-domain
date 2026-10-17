# -*- coding: utf-8 -*-
"""
train_simclr.py — SimCLR 自监督预训练（无标注数据）。

用法：
  python train_simclr.py --csv ../data/processed/ssl_pool_T.csv --epochs 200 --batch 512

说明：
  - 只读取 image_path 列（不使用标签），符合"零标注"约束；
  - 输出预训练 backbone 权重到 outputs/checkpoints/，供 train_supervised.py 加载；
  - 显存不足时用 --accum 做梯度累积，等效增大 batch。
"""
import argparse
import csv
import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "outputs" / "checkpoints"

_PIL_TO_TENSOR = transforms.ToTensor()


def load_image(path: str) -> torch.Tensor:
    """用 PIL 读图并转为 [C,H,W] float [0,1] tensor。

    优先读取 256 缓存小图（快 ~50 倍），缓存缺失时回退读原图。
    """
    p = Path(path)
    parts = p.parts
    if "raw" in parts:
        ds = parts[parts.index("raw") + 1]  # bracol / rocole
        cached = ROOT / "data" / "processed" / "imgs256" / ds / (p.stem + ".jpg")
        if cached.exists():
            p = cached
    return _PIL_TO_TENSOR(Image.open(p).convert("RGB"))


# ---------------- 数据 ----------------

class UnlabeledImageDataset(Dataset):
    def __init__(self, csv_path: Path, transform):
        self.paths: List[str] = []
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                self.paths.append(row["image_path"])
        if not self.paths:
            raise ValueError("CSV 中没有图像路径: " + str(csv_path))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = load_image(self.paths[i])
        # 两张独立增强视图
        x1 = self.transform(img)
        x2 = self.transform(img)
        return x1, x2


def simclr_augment():
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
    ])


# ---------------- 模型 ----------------

class SimCLR(nn.Module):
    def __init__(self, proj_dim: int = 128):
        super().__init__()
        self.backbone = torchvision.models.resnet50(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # 去掉分类头
        self.proj = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, proj_dim),
        )

    def forward(self, x):
        h = self.backbone(x)
        z = F.normalize(self.proj(h), dim=-1)
        return z


def ntxent_loss(z1, z2, temperature=0.5):
    z1 = z1.float()
    z2 = z2.float()
    z = torch.cat([z1, z2], dim=0)                       # 2N x d
    sim = torch.mm(z, z.t()) / temperature               # 2N x 2N
    N = z1.shape[0]
    # 正样本对索引
    pos = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)], dim=0).to(z.device)
    mask = torch.eye(2 * N, device=z.device).bool()
    sim = sim.masked_fill(mask, -9e15)
    logits_max = sim.max(dim=1, keepdim=True)[0]
    logits = sim - logits_max.detach()
    labels = pos
    loss = F.cross_entropy(logits, labels)
    return loss


# ---------------- 训练 ----------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="无标注图像索引 CSV（ssl_pool_*.csv）")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--accum", type=int, default=1, help="梯度累积步数")
    p.add_argument("--lr", type=float, default=0.3)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="simclr_backbone.pth")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] 设备:", device)

    dataset = UnlabeledImageDataset(Path(args.csv), simclr_augment())
    loader = DataLoader(dataset, batch_size=args.batch // args.accum,
                        shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=False)
    print("[INFO] 样本数:", len(dataset))

    model = SimCLR(args.proj_dim).to(device)
    # 使用 SGD + 大学习率（SimCLR 标准）；也可换 LARS
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=1e-6)
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.epochs
    pct_start = min(max(args.warmup / args.epochs, 0.0), 0.5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=pct_start, anneal_strategy="cos")

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    CKPT.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        running = 0.0
        for i, (x1, x2) in enumerate(loader):
            x1, x2 = x1.to(device), x2.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                z1 = model(x1)
                z2 = model(x2)
            loss = ntxent_loss(z1, z2, args.temperature) / args.accum
            scaler.scale(loss).backward()
            running += loss.item() * args.accum

            if (i + 1) % args.accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        avg = running / max(steps_per_epoch, 1)
        print(f"[EPOCH {epoch+1}/{args.epochs}] loss={avg:.4f} lr={scheduler.get_last_lr()[0]:.4f}")
        if avg < best_loss:
            best_loss = avg
            torch.save(model.backbone.state_dict(), CKPT / args.out)
            print("[SAVE] 最佳 backbone 已保存:", CKPT / args.out)

    print("[DONE] 预训练完成，权重:", CKPT / args.out)


if __name__ == "__main__":
    main()
