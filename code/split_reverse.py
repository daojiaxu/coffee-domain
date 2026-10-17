# -*- coding: utf-8 -*-
"""
split_reverse.py — 反向域对 RoCoLe(field)→BRACOL(lab) 的数据划分。

与正向（BRACOL→RoCoLe，preprocess.py）对称，用于研究域适应方向的对称性：
  源域 S = RoCoLe（田间自然，有标签 healthy/rust）
  目标域 T = BRACOL（白底受控，无标签池 + 测试集）

任务：healthy vs rust 二分类（两数据集唯一公共语义）。

划分：
  - RoCoLe 全部 healthy/rust 作源域训练（split=train）
  - BRACOL 按 group(br_{id}，每张叶片独立) 分层：30% 测试集 / 70% 无标签池

输出（data/processed/）：
  labeled_S_rev.csv    源域 RoCoLe 训练（有标签）
  ssl_pool_T_rev.csv   目标域 BRACOL 无标签池
  test_T_rev.csv       目标域 BRACOL 测试集（有标签，仅评估）

用法：python split_reverse.py --seed 42 --test_ratio 0.3
"""
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

BRACOL_STRESS = {"0": "healthy", "2": "rust"}  # 反向任务只取 healthy/rust


def map_rocole(cls: str) -> Optional[str]:
    c = cls.strip().lower()
    if c == "healthy":
        return "healthy"
    if c.startswith("rust_level"):
        return "rust"
    return None  # red_spider_mite 丢弃


def parse_rocole() -> List[Dict]:
    csv_path = RAW / "rocole" / "Annotations" / "RoCoLE-csv.csv"
    photos = RAW / "rocole" / "Photos"
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fn = r.get("External ID", "").strip()
            if not fn:
                continue
            img = photos / fn
            if not img.exists():
                continue
            try:
                j = json.loads(r.get("Label", "{}"))
                cls = j.get("classification", "")
            except Exception:
                cls = ""
            label = map_rocole(cls)
            if label is None:
                continue
            rows.append({"image_path": str(img), "label": label,
                         "dataset": "rocole", "group": f"rc_{fn}"})
    print("[INFO] RoCoLe (源域):", len(rows), "张")
    return rows


def parse_bracol() -> List[Dict]:
    csv_path = RAW / "bracol" / "dataset.csv"
    img_dir = RAW / "bracol" / "images"
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fid = r.get("id", "").strip()
            img = img_dir / f"{fid}.jpg"
            if not img.exists():
                continue
            label = BRACOL_STRESS.get(r.get("predominant_stress", "").strip())
            if label is None:
                continue  # miner/phoma/cercospora/多病害 丢弃
            rows.append({"image_path": str(img), "label": label,
                         "dataset": "bracol", "group": f"br_{fid}"})
    print("[INFO] BRACOL (目标域, healthy/rust):", len(rows), "张")
    return rows


def write_csv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_path", "label", "dataset", "group", "split"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("[OK]", path.name, f"({len(rows)} 行)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_ratio", type=float, default=0.3)
    args = p.parse_args()
    random.seed(args.seed)

    S = parse_rocole()   # 源域：RoCoLe（field）
    T = parse_bracol()   # 目标域：BRACOL（lab）

    # 目标域按 group（独立叶片）分层划分测试集，防泄漏
    groups = {}
    for r in T:
        groups.setdefault(r["group"], r)
    group_list = list(groups.values())
    random.shuffle(group_list)
    n_test = int(len(group_list) * args.test_ratio)
    test_groups = {g["group"] for g in group_list[:n_test]}
    test_T, pool_T = [], []
    for r in T:
        (test_T if r["group"] in test_groups else pool_T).append(r)

    for r in S:
        r["split"] = "train"
    for r in test_T:
        r["split"] = "test"
    for r in pool_T:
        r["split"] = "unlabeled"

    write_csv(OUT / "labeled_S_rev.csv", S)
    write_csv(OUT / "test_T_rev.csv", test_T)
    write_csv(OUT / "ssl_pool_T_rev.csv", pool_T)

    print("\n===== 反向域对类别统计 =====")
    for name, rows in [("S源域(RoCoLe)", S), ("T测试(BRACOL)", test_T),
                       ("T无标注池(BRACOL)", pool_T)]:
        print(f"  {name}: {dict(Counter(r['label'] for r in rows))}")


if __name__ == "__main__":
    main()
