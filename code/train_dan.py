# -*- coding: utf-8 -*-
"""
train_dan.py — DAN 基线（多核 MMD 全局特征分布对齐）。
损失 = 交叉熵(源域) + λ * MK-MMD(源域特征, 目标域特征)。

用法：
  python train_dan.py --src data/processed/labeled_S.csv \
         --tgt data/processed/ssl_pool_T.csv --test data/processed/test_T.csv \
         --epochs 50 --tag DAN_s42
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from uda_common import (RES, LabeledDataset, UnlabeledDataset, evaluate,
                        make_backbone, save_metrics, train_transform, eval_transform)


class DAN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.feature, in_features = make_backbone()
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        f = self.feature(x)
        return self.classifier(f), f

    def predict(self, x):
        return self.classifier(self.feature(x))


def median_pairwise_sq(f: torch.Tensor) -> torch.Tensor:
    """采样估计 pairwise 平方距离中位数，用于确定高斯核带宽。"""
    n = f.shape[0]
    if n < 2:
        return torch.tensor(1.0, device=f.device)
    idx = torch.randperm(n)[: min(n, 200)].to(f.device)
    sub = f[idx]
    l2 = (sub ** 2).sum(1, keepdim=True) + (sub ** 2).sum(1, keepdim=True).t() \
         - 2.0 * (sub @ sub.t())
    triu = torch.triu(torch.ones_like(l2, dtype=torch.bool), diagonal=1)
    vals = l2[triu]
    if vals.numel() == 0:
        return torch.tensor(1.0, device=f.device)
    return vals.median().clamp(min=1e-3)


def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, bandwidths) -> torch.Tensor:
    l2 = (x ** 2).sum(1, keepdim=True) + (y ** 2).sum(1, keepdim=True).t() \
         - 2.0 * (x @ y.t())
    k = torch.zeros_like(l2)
    for b in bandwidths:
        k = k + torch.exp(-l2 / (2.0 * b))
    return k


def mk_mmd(f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
    """多核 MMD（4 个带宽，按 pairwise 距离中位数缩放）。"""
    n, m = f_s.shape[0], f_t.shape[0]
    with torch.no_grad():
        sigma2 = median_pairwise_sq(torch.cat([f_s, f_t], dim=0))
    bandwidths = [sigma2 * c for c in (0.5, 1.0, 2.0, 4.0)]
    k_ss = gaussian_kernel(f_s, f_s, bandwidths)
    k_tt = gaussian_kernel(f_t, f_t, bandwidths)
    k_st = gaussian_kernel(f_s, f_t, bandwidths)
    return k_ss.sum() / (n * n) + k_tt.sum() / (m * m) - 2.0 * k_st.sum() / (n * m)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--tgt", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lambda_mmd", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default="DAN")
    p.add_argument("--labels", type=str, default="healthy,rust")
    args = p.parse_args()

    labels = [x.strip() for x in args.labels.split(",")]
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_ds = LabeledDataset(Path(args.src), train_transform(), labels)
    tgt_ds = UnlabeledDataset(Path(args.tgt), train_transform())
    test_ds = LabeledDataset(Path(args.test), eval_transform(), labels)

    src_loader = DataLoader(src_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    tgt_loader = DataLoader(tgt_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    print(f"[INFO] src={len(src_ds)} tgt={len(tgt_ds)} test={len(test_ds)} "
          f"lambda_mmd={args.lambda_mmd}")

    model = DAN(len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ce = nn.CrossEntropyLoss()

    best = {"test_acc": 0.0, "epoch": -1}

    for epoch in range(args.epochs):
        model.train()
        running_cls = running_mmd = 0.0
        n = 0
        for (xs, ys), xt in zip(src_loader, tgt_loader):
            xs, ys = xs.to(device), ys.to(device)
            xt = xt.to(device)

            logits_s, f_s = model(xs)
            _, f_t = model(xt)

            loss_cls = ce(logits_s, ys)
            loss_mmd = args.lambda_mmd * mk_mmd(f_s, f_t)
            loss = loss_cls + loss_mmd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_cls += loss_cls.item()
            running_mmd += loss_mmd.item()
            n += 1

        acc = evaluate(model, test_loader, device)
        print(f"[EPOCH {epoch+1}/{args.epochs}] test_acc={acc:.4f} "
              f"cls={running_cls/n:.3f} mmd={running_mmd/n:.4f}")
        if acc > best["test_acc"]:
            best = {"test_acc": acc, "epoch": epoch + 1}
            torch.save(model.state_dict(), RES / f"{args.tag}_best.pth")

    acc = evaluate(model, test_loader, device)
    save_metrics(args.tag, "dan", args.src, args.tgt, args.test, acc, best,
                 extra={"lambda_mmd": args.lambda_mmd})


if __name__ == "__main__":
    main()
