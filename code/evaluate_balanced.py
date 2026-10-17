# -*- coding: utf-8 -*-
"""
evaluate_balanced.py — 对 healthy vs disease 的 best checkpoint 重新评估，
输出 accuracy 与 balanced accuracy（用于处理目标域类别失衡）。

仅用于反向（field->lab，目标域 BRACOL 测试集 healthy79/disease426 严重失衡）。
"""
import csv
import json
from pathlib import Path

import torch
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


def build_supervised(ckpt):
    model = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model


def build_dann(ckpt):
    import train_dann
    model = train_dann.DANN(2)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model


def build_cwdann(ckpt):
    import train_cwdann
    model = train_cwdann.CW_DANN(2, conditional=True)
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
            else:
                logits, _ = model(x, 0.0)
            preds += logits.argmax(1).cpu().tolist()
            trues += y.tolist()
    # accuracy
    acc = sum(p == t for p, t in zip(preds, trues)) / max(len(trues), 1)
    # per-class recall
    from collections import Counter
    recall = {}
    for c in range(2):
        idx = [i for i, t in enumerate(trues) if t == c]
        if not idx:
            recall[c] = 0.0
        else:
            recall[c] = sum(preds[i] == c for i in idx) / len(idx)
    bal = (recall[0] + recall[1]) / 2.0
    return acc, bal, recall


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = TestDS(ROOT / "data" / "processed" / "test_T_hd_rev.csv")
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    print(f"[INFO] test={len(test_ds)}")

    seeds = ["42", "123", "2024", "7", "99", "13", "777"]
    for kind, prefix in [("supervised", "P2_hd_rev"),
                         ("dann", "P7_dann_hd_rev"),
                         ("cwdann", "CW-DANN_hd_rev")]:
        accs, bals = [], []
        for s in seeds:
            ckpt = RES / f"{prefix}_s{s}_best.pth"
            if kind == "supervised":
                model = build_supervised(ckpt)
            elif kind == "dann":
                model = build_dann(ckpt)
            else:
                model = build_cwdann(ckpt)
            model = model.to(device)
            acc, bal, recall = evaluate(model, test_loader, device, kind)
            accs.append(acc)
            bals.append(bal)
            print(f"  {prefix}_s{s}: acc={acc*100:.1f}%  bal={bal*100:.1f}%  "
                  f"recall(h/d)={recall[0]*100:.0f}/{recall[1]*100:.0f}")
        import statistics as st
        print(f"  => {prefix}  mean acc={st.mean(accs)*100:.1f}±{st.stdev(accs)*100:.1f}%   "
              f"mean bal={st.mean(bals)*100:.1f}±{st.stdev(bals)*100:.1f}%")


if __name__ == "__main__":
    main()
