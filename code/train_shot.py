# -*- coding: utf-8 -*-
"""
train_shot.py — SHOT 基线（源域预训练 + 目标域信息最大化自训练）。
阶段1：源域交叉熵监督预训练；
阶段2：固定特征提取器，用信息最大化（熵最小化 + 多样性）在目标域无标注
       数据上微调分类器。

用法：
  python train_shot.py --src data/processed/labeled_S.csv \
         --tgt data/processed/ssl_pool_T.csv --test data/processed/test_T.csv \
         --src_epochs 50 --adapt_epochs 15 --tag SHOT_s42
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from uda_common import (RES, LabeledDataset, UnlabeledDataset, evaluate,
                        make_backbone, save_metrics, train_transform, eval_transform)


class SHOT(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.feature, in_features = make_backbone()
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.classifier(self.feature(x))

    def predict(self, x):
        return self.classifier(self.feature(x))


def im_loss(logits: torch.Tensor) -> torch.Tensor:
    """信息最大化损失 = 熵最小化 + 多样性。"""
    p = F.softmax(logits, dim=1)
    ent = -(p * torch.log(p + 1e-6)).sum(1).mean()
    p_hat = p.mean(0)
    div = (p_hat * torch.log(p_hat + 1e-6)).sum()
    return ent + div


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--tgt", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--src_epochs", type=int, default=50)
    p.add_argument("--adapt_epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--adapt_lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default="SHOT")
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
          f"src_epochs={args.src_epochs} adapt_epochs={args.adapt_epochs}")

    model = SHOT(len(labels)).to(device)
    ce = nn.CrossEntropyLoss()

    # ---- 阶段1：源域监督预训练 ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.src_epochs)
    for epoch in range(args.src_epochs):
        model.train()
        running, n = 0.0, 0
        for xs, ys in src_loader:
            xs, ys = xs.to(device), ys.to(device)
            loss = ce(model(xs), ys)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item()
            n += 1
        print(f"[SRC EPOCH {epoch+1}/{args.src_epochs}] loss={running/n:.4f}")

    # ---- 阶段2：目标域信息最大化自训练（固定 backbone，仅微调分类器）----
    model.feature.requires_grad_(False)
    optimizer2 = torch.optim.AdamW(model.classifier.parameters(), lr=args.adapt_lr,
                                   weight_decay=1e-4)
    best = {"test_acc": 0.0, "epoch": -1}

    for epoch in range(args.adapt_epochs):
        model.train()
        running, n = 0.0, 0
        for xt in tgt_loader:
            xt = xt.to(device)
            loss = im_loss(model(xt))
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()
            running += loss.item()
            n += 1
        acc = evaluate(model, test_loader, device)
        print(f"[ADAPT EPOCH {epoch+1}/{args.adapt_epochs}] test_acc={acc:.4f} "
              f"im={running/n:.4f}")
        if acc > best["test_acc"]:
            best = {"test_acc": acc, "epoch": epoch + 1}
            torch.save(model.state_dict(), RES / f"{args.tag}_best.pth")

    acc = evaluate(model, test_loader, device)
    save_metrics(args.tag, "shot", args.src, args.tgt, args.test, acc, best,
                 extra={"src_epochs": args.src_epochs, "adapt_epochs": args.adapt_epochs})


if __name__ == "__main__":
    main()
