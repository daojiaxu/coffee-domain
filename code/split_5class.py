# -*- coding: utf-8 -*-
"""
split_5class.py — BRACOL 5 类细粒度任务的防泄漏分层划分。

BRACOL 的 group = "br_{id}" 即图片 id（每张叶片独立），故按 label 分层随机划分：
  70% train / 30% test，用固定 seed 保证可复现。

输出：
  data/processed/bracol_5class_train.csv
  data/processed/bracol_5class_test.csv

用法：python split_5class.py --seed 42 --test_ratio 0.3
"""
import csv
import random
import argparse
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "processed" / "bracol_5class.csv"
OUT = ROOT / "data" / "processed"

FIELDS = ["image_path", "label", "dataset", "group", "split"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_ratio", type=float, default=0.3)
    args = p.parse_args()
    random.seed(args.seed)

    rows: List[Dict] = []
    with open(SRC, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)

    train, test = [], []
    for label, items in by_label.items():
        random.shuffle(items)
        n_test = max(1, int(round(len(items) * args.test_ratio)))
        for r in items[:n_test]:
            r["split"] = "test"
            test.append(r)
        for r in items[n_test:]:
            r["split"] = "train"
            train.append(r)

    for name, data in [("bracol_5class_train.csv", train),
                       ("bracol_5class_test.csv", test)]:
        path = OUT / name
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for r in data:
                w.writerow(r)
        print("[OK]", name, f"({len(data)} 行)")

    from collections import Counter
    print("\n===== 划分统计 =====")
    print("train:", dict(Counter(r["label"] for r in train)))
    print("test: ", dict(Counter(r["label"] for r in test)))


if __name__ == "__main__":
    main()
