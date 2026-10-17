# -*- coding: utf-8 -*-
"""
train_cdan.py — CDAN 基线（条件域对抗 + 多线性条件）。
域判别器输入 = 特征与分类器 softmax 输出的外积（multilinear map），经随机
多线性投影（RMC）降维。损失 = 交叉熵(源域) + 域对抗损失。

用法：
  python train_cdan.py --src data/processed/labeled_S.csv \
         --tgt data/processed/ssl_pool_T.csv --test data/processed/test_T.csv \
         --epochs 50 --tag CDAN_s42
"""
import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from uda_common import (RES, LabeledDataset, UnlabeledDataset, evaluate,
                        make_backbone, save_metrics, train_transform, eval_transform)


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


def lambda_schedule(p: float) -> float:
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


class CDAN(nn.Module):
    def __init__(self, num_classes: int, proj_dim: int = 1024):
        super().__init__()
        self.feature, in_features = make_backbone()
        self.classifier = nn.Linear(in_features, num_classes)
        # 随机多线性投影（RMC），固定不训练
        self.proj_f = nn.Linear(in_features, proj_dim, bias=False)
        self.proj_g = nn.Linear(num_classes, num_classes, bias=False)
        for prm in list(self.proj_f.parameters()) + list(self.proj_g.parameters()):
            prm.requires_grad = False
        self.dom_in = proj_dim * num_classes
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.dom_in, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x, lambda_):
        f = self.feature(x)
        logits = self.classifier(f)
        g = F.softmax(logits, dim=1).detach()  # 条件信号，不反传（稳定训练）
        f_ = self.proj_f(f)
        g_ = self.proj_g(g)
        outer = (f_.unsqueeze(2) * g_.unsqueeze(1)).flatten(1)
        dom_input = grad_reverse(outer, lambda_)
        domain_logits = self.domain_classifier(dom_input)
        return logits, domain_logits

    def predict(self, x):
        return self.classifier(self.feature(x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--tgt", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default="CDAN")
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

    print(f"[INFO] src={len(src_ds)} tgt={len(tgt_ds)} test={len(test_ds)}")

    model = CDAN(len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()

    steps_per_epoch = min(len(src_loader), len(tgt_loader))
    total_steps = steps_per_epoch * args.epochs
    global_step = 0

    best = {"test_acc": 0.0, "epoch": -1}

    for epoch in range(args.epochs):
        model.train()
        running_cls = running_dom = 0.0
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
    save_metrics(args.tag, "cdan", args.src, args.tgt, args.test, acc, best)


if __name__ == "__main__":
    main()
