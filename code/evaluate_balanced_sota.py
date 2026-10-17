# -*- coding: utf-8 -*-
"""
evaluate_balanced_sota.py — 对 healthy/disease 反向（目标域 BRACOL 测试集失衡
healthy 79 / disease 426）的 best checkpoint 统一重评，输出 accuracy + balanced
accuracy + per-class recall。支持全部 7 个方法（P2/DANN/CW-DANN/DeepCORAL/DAN/CDAN/SHOT），
与既有 evaluate_balanced.py 保持协议一致（accuracy 训练+选 best，此处补 balanced 指标）。
"""
import csv
import json
import statistics as st
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "outputs" / "results"

L2I = {"healthy": 0, "disease": 1}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
_PIL_TO_TENSOR = transforms.ToTensor()

SEEDS = ["42", "123", "2024", "7", "99", "13", "777"]

# tag 前缀 -> (build_kind, 显示名)
METHODS = [
    ("P2_hd_rev", "supervised", "P2"),
    ("P7_dann_hd_rev", "dann", "DANN"),
    ("CW-DANN_hd_rev", "cwdann", "CW-DANN"),
    ("DeepCORAL_hd_rev", "deepcoral", "DeepCORAL"),
    ("DAN_hd_rev", "dan", "DAN"),
    ("CDAN_hd_rev", "cdan", "CDAN"),
    ("SHOT_hd_rev", "shot", "SHOT"),
]


def load_image(path: str) -> torch.Tensor:
    p = Path(path)
    parts = p.parts
    if "raw" in parts:
        ds = parts[parts.index("raw") + 1]
        cached = ROOT / "data" / "processed" / "imgs256" / ds / (p.stem + ".jpg")
        if cached.exists():
            p = cached
    return _PIL_TO_TENSOR(Image.open(p).convert("RGB"))


class TestDS(Dataset):
    def __init__(self, csv_path):
        self.items = []
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("label") in L2I:
                    self.items.append(row)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img = load_image(self.items[i]["image_path"])
        img = transforms.Resize(256)(img)
        img = transforms.CenterCrop(224)(img)
        img = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(img)
        return img, L2I[self.items[i]["label"]]


def build(kind, ckpt):
    if kind == "supervised":
        model = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif kind == "dann":
        import train_dann
        model = train_dann.DANN(2)
    elif kind == "cwdann":
        import train_cwdann
        model = train_cwdann.CW_DANN(2, conditional=True)
    elif kind == "deepcoral":
        import train_deepcoral
        model = train_deepcoral.DeepCORAL(2)
    elif kind == "dan":
        import train_dan
        model = train_dan.DAN(2)
    elif kind == "cdan":
        import train_cdan
        model = train_cdan.CDAN(2)
    elif kind == "shot":
        import train_shot
        model = train_shot.SHOT(2)
    else:
        raise ValueError(kind)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model


def evaluate(model, loader, device, kind):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            if kind == "supervised":
                logits = model(x)
            elif kind in ("dann", "cwdann"):
                logits, _ = model(x, 0.0)
            else:
                logits = model.predict(x)
            preds += logits.argmax(1).cpu().tolist()
            trues += y.tolist()
    acc = sum(p == t for p, t in zip(preds, trues)) / max(len(trues), 1)
    recall = {}
    for c in range(2):
        idx = [i for i, t in enumerate(trues) if t == c]
        recall[c] = (sum(preds[i] == c for i in idx) / len(idx)) if idx else 0.0
    bal = (recall[0] + recall[1]) / 2.0
    return acc, bal, recall


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = TestDS(ROOT / "data" / "processed" / "test_T_hd_rev.csv")
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    print(f"[INFO] test={len(test_ds)} (healthy/disease)")

    rows = []
    for prefix, kind, disp in METHODS:
        accs, bals, r0, r1 = [], [], [], []
        for s in SEEDS:
            ckpt = RES / f"{prefix}_s{s}_best.pth"
            if not ckpt.exists():
                print(f"  [MISS] {prefix}_s{s}")
                continue
            model = build(kind, ckpt).to(device)
            acc, bal, recall = evaluate(model, test_loader, device, kind)
            accs.append(acc * 100)
            bals.append(bal * 100)
            r0.append(recall[0] * 100)
            r1.append(recall[1] * 100)
            print(f"  {prefix}_s{s}: acc={acc*100:.1f} bal={bal*100:.1f} "
                  f"recall(h/d)={recall[0]*100:.0f}/{recall[1]*100:.0f}")
        if not accs:
            rows.append((disp, "N/A", "N/A", "N/A", "N/A"))
            continue
        rows.append((
            disp,
            f"{st.mean(accs):.1f} ± {st.stdev(accs):.1f}",
            f"{st.mean(bals):.1f} ± {st.stdev(bals):.1f}",
            f"{st.mean(r0):.1f}",
            f"{st.mean(r1):.1f}",
        ))

    print("\n===== healthy/disease 反向 汇总 =====")
    print(f"{'方法':<14}{'acc(best)':<18}{'balanced(best)':<18}{'recall_healthy':<16}{'recall_disease'}")
    for disp, a, b, r0, r1 in rows:
        print(f"{disp:<14}{a:<18}{b:<18}{r0:<16}{r1}")

    # 保存 JSON 供后续合并
    out = {}
    for (prefix, kind, disp), (_, a, b, r0, r1) in zip(METHODS, rows):
        out[disp] = {"acc": a, "balanced": b, "recall_healthy": r0, "recall_disease": r1}
    with open(RES / "hd_rev_balanced_summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVED] {RES / 'hd_rev_balanced_summary.json'}")


if __name__ == "__main__":
    main()
