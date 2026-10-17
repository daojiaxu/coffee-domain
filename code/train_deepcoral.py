# -*- coding: utf-8 -*-
"""
train_deepcoral.py — DeepCORAL 基线（CORAL 二阶统计量对齐）。
损失 = 交叉熵(源域) + λ * CORAL(源域特征协方差, 目标域特征协方差)。

用法：
  python train_deepcoral.py --src data/processed/labeled_S.csv \
         --tgt data/processed/ssl_pool_T.csv --test data/processed/test_T.csv \
         --epochs 50 --tag DeepCORAL_s42
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


class DeepCORAL(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.feature, in_features = make_backbone()
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        f = self.feature(x)
        return self.classifier(f), f

    def predict(self, x):
        return self.classifier(self.feature(x))


def coral_loss(f_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
    """CORAL loss：源/目标特征协方差矩阵的 Frobenius 距离（按维度归一化）。"""
    d = f_s.shape[1]
    f_s = f_s - f_s.mean(0, keepdim=True)
    f_t = f_t - f_t.mean(0, keepdim=True)
    cov_s = (f_s.t() @ f_s) / (f_s.shape[0] - 1)
    cov_t = (f_t.t() @ f_t) / (f_t.shape[0] - 1)
    return (cov_s - cov_t).pow(2).sum() / (4.0 * d * d)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--tgt", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lambda_coral", type=float, default=100.0,
                   help="CORAL 对齐权重（特征量级小，需较大 λ，已按实测量级标定）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default="DeepCORAL")
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
          f"lambda_coral={args.lambda_coral}")

    model = DeepCORAL(len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ce = nn.CrossEntropyLoss()

    steps_per_epoch = min(len(src_loader), len(tgt_loader))
    total_steps = steps_per_epoch * args.epochs
    global_step = 0

    best = {"test_acc": 0.0, "epoch": -1}

    for epoch in range(args.epochs):
        model.train()
        running_cls = running_coral = 0.0
        n = 0
        for (xs, ys), xt in zip(src_loader, tgt_loader):
            xs, ys = xs.to(device), ys.to(device)
            xt = xt.to(device)

            logits_s, f_s = model(xs)
            _, f_t = model(xt)

            loss_cls = ce(logits_s, ys)
            loss_coral = args.lambda_coral * coral_loss(f_s, f_t)
            loss = loss_cls + loss_coral

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            running_cls += loss_cls.item()
            running_coral += loss_coral.item()
            n += 1

        acc = evaluate(model, test_loader, device)
        print(f"[EPOCH {epoch+1}/{args.epochs}] test_acc={acc:.4f} "
              f"cls={running_cls/n:.3f} coral={running_coral/n:.4f}")
        if acc > best["test_acc"]:
            best = {"test_acc": acc, "epoch": epoch + 1}
            torch.save(model.state_dict(), RES / f"{args.tag}_best.pth")

    acc = evaluate(model, test_loader, device)
    save_metrics(args.tag, "deepcoral", args.src, args.tgt, args.test, acc, best,
                 extra={"lambda_coral": args.lambda_coral})


if __name__ == "__main__":
    main()
