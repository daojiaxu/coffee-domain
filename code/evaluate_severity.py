# -*- coding: utf-8 -*-
"""
evaluate_severity.py — 对 rust 严重度 4 级的 best checkpoint 重新评估，
输出 accuracy、MAE（平均等级误差）与 per-level recall。

MAE = mean(|pred_level - true_level|)，等级 1~4（内部映射为 0~3）。
"""
import csv
from collections import Counter
from pathlib import Path

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "outputs" / "results"

L2I = {"level1": 0, "level2": 1, "level3": 2, "level4": 3}
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


def build_supervised(ckpt, nc=4):
    model = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, nc)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model


def build_dann(ckpt, nc=4):
    import train_dann
    model = train_dann.DANN(nc)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model


def build_cwdann(ckpt, nc=4):
    import train_cwdann
    model = train_cwdann.CW_DANN(nc, conditional=True)
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
    acc = sum(p == t for p, t in zip(preds, trues)) / max(len(trues), 1)
    mae = sum(abs(p - t) for p, t in zip(preds, trues)) / max(len(trues), 1)
    recall = {}
    for c in range(4):
        idx = [i for i, t in enumerate(trues) if t == c]
        recall[c] = sum(preds[i] == c for i in idx) / len(idx) if idx else 0.0
    return acc, mae, recall


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = ["42", "123", "2024", "7", "99", "13", "777"]

    for direction, test_csv, p2p, dannp, cwp in [
        ("正向 BRACOL->RoCoLe", "test_T_sev.csv", "P2_sev", "P7_dann_sev", "CW-DANN_sev"),
        ("反向 RoCoLe->BRACOL", "test_T_sev_rev.csv", "P2_sev_rev", "P7_dann_sev_rev", "CW-DANN_sev_rev"),
    ]:
        print(f"\n===== {direction} =====")
        test_ds = TestDS(ROOT / "data" / "processed" / test_csv)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
        print(f"[INFO] test={len(test_ds)}")
        for kind, prefix in [("supervised", p2p), ("dann", dannp), ("cwdann", cwp)]:
            accs, maes = [], []
            for s in seeds:
                ckpt = RES / f"{prefix}_s{s}_best.pth"
                if kind == "supervised":
                    model = build_supervised(ckpt)
                elif kind == "dann":
                    model = build_dann(ckpt)
                else:
                    model = build_cwdann(ckpt)
                model = model.to(device)
                acc, mae, recall = evaluate(model, test_loader, device, kind)
                accs.append(acc)
                maes.append(mae)
                print(f"  {prefix}_s{s}: acc={acc*100:.1f}%  MAE={mae:.2f}  "
                      f"recall={[f'{recall[c]*100:.0f}' for c in range(4)]}")
            import statistics as st
            print(f"  => {prefix}  mean acc={st.mean(accs)*100:.1f}±{st.stdev(accs)*100:.1f}%   "
                  f"mean MAE={st.mean(maes):.2f}±{st.stdev(maes):.2f}")


if __name__ == "__main__":
    main()
