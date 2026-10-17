# -*- coding: utf-8 -*-
"""
train_moco.py — MoCo v2 自监督预训练（无标注数据）。

用法：
  python train_moco.py --csv ../data/processed/ssl_pool_T.csv --epochs 200 --batch 128

说明：
  - 只读取 image_path 列（不使用标签），符合"零标注"约束；
  - 用动量编码器 + memory bank 队列提供大量负样本，绕开 SimCLR 对
    大 batch 的依赖（小 batch 也能拿到 K 个负样本）；
  - 输出 query encoder 的 backbone 权重到 outputs/checkpoints/，
    供 train_supervised.py --init ssl 加载。

已知简化：
  - 未做 MoCo v2 的 ShuffleBN（防 key encoder BN 在 batch 内信息泄漏）。
    数据量小、batch 128 场景下影响有限，如需严格对齐可后续补。
"""
import argparse
import csv
from pathlib import Path
from typing import List

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
        x1 = self.transform(img)
        x2 = self.transform(img)
        return x1, x2


def moco_augment():
    """MoCo v2 标准增强（略弱于 SimCLR 的强增强）。"""
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
    ])


class MoCo(nn.Module):
    """MoCo v2：query encoder + momentum key encoder + memory bank 队列。"""

    def __init__(self, dim: int = 128, K: int = 8192, m: float = 0.999, T: float = 0.2):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = torchvision.models.resnet50(weights=None)
        in_features = self.encoder_q.fc.in_features
        self.encoder_q.fc = nn.Identity()

        self.encoder_k = torchvision.models.resnet50(weights=None)
        self.encoder_k.fc = nn.Identity()

        self.head_q = nn.Sequential(
            nn.Linear(in_features, in_features), nn.ReLU(inplace=True), nn.Linear(in_features, dim))
        self.head_k = nn.Sequential(
            nn.Linear(in_features, in_features), nn.ReLU(inplace=True), nn.Linear(in_features, dim))

        # 初始化 key encoder 与 query 相同，且冻结（只通过动量更新）
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # memory bank 队列：dim x K，初始随机单位向量
        self.register_buffer("queue", F.normalize(torch.randn(dim, K), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """keys: [N, dim]，归一化后入队，最老的出队。"""
        batch_size = keys.shape[0]
        keys = keys.float()
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0, f"K({self.K}) 必须是 batch({batch_size}) 的整数倍"
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, xq, xk):
        q = self.head_q(self.encoder_q(xq))
        q = F.normalize(q, dim=-1)
        with torch.no_grad():
            self._momentum_update()
            k = self.head_k(self.encoder_k(xk))
            k = F.normalize(k, dim=-1)
        return q, k


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="无标注图像索引 CSV（ssl_pool_*.csv）")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--moco-dim", type=int, default=128)
    p.add_argument("--moco-k", type=int, default=8192)
    p.add_argument("--moco-m", type=float, default=0.999)
    p.add_argument("--moco-t", type=float, default=0.2)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="moco_backbone.pth")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] 设备:", device)

    dataset = UnlabeledImageDataset(Path(args.csv), moco_augment())
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, pin_memory=True, drop_last=True)
    print("[INFO] 样本数:", len(dataset), " 每 epoch step:", len(loader))

    model = MoCo(args.moco_dim, args.moco_k, args.moco_m, args.moco_t).to(device)
    # 只优化 query encoder + head_q（key 侧 requires_grad=False）
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # OneCycleLR：total_steps 必须是总 step 数（epochs * steps_per_epoch），不是 epoch 数
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.epochs
    pct_start = min(max(args.warmup / args.epochs, 0.0), 0.5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=pct_start, anneal_strategy="cos")
    criterion = nn.CrossEntropyLoss()

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    CKPT.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for i, (xq, xk) in enumerate(loader):
            xq, xk = xq.to(device), xk.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                q, k = model(xq, xk)
            # InfoNCE loss 在 fp32 计算，且用【旧 queue】（不含当前 batch 的 key，避免正样本泄漏）
            q = q.float()
            k = k.float()
            l_pos = torch.einsum("nc,nc->n", q, k).unsqueeze(-1)          # N x 1
            l_neg = torch.einsum("nc,ck->nk", q, model.queue.clone().detach())  # N x K
            logits = torch.cat([l_pos, l_neg], dim=1) / model.T
            labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()

            # loss 计算完成后，再把当前 batch 的 key 入队（供后续 step 使用）
            model._dequeue_and_enqueue(k.detach())

        avg = running / max(len(loader), 1)
        print(f"[EPOCH {epoch+1}/{args.epochs}] loss={avg:.4f} lr={scheduler.get_last_lr()[0]:.5f}")
        if avg < best_loss:
            best_loss = avg
            torch.save(model.encoder_q.state_dict(), CKPT / args.out)
            print("[SAVE] 最佳 backbone 已保存:", CKPT / args.out)

    print("[DONE] 预训练完成，权重:", CKPT / args.out)


if __name__ == "__main__":
    main()
