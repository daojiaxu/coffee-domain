# -*- coding: utf-8 -*-
"""
preprocess.py — 类别对齐 + 防泄漏划分 + 生成统一索引 CSV。

【两域方案】源域 S = BRACOL（白底受控），目标域 T = RoCoLe（田间自然）
  核心任务：healthy vs rust 二分类（两数据集唯一公共语义）

实际数据结构（data/raw/）：
  bracol/   dataset.csv + images/{id}.jpg
            CSV 列: id, predominant_stress(0~5), miner, rust, phoma, cercospora, severity
            predominant_stress: 0=healthy 1=miner 2=rust 3=phoma 4=cercospora 5=多病害混合
  rocole/   Annotations/RoCoLE-csv.csv + Photos/*.jpg
            CSV 列含 External ID（文件名）、Label（JSON，内含 classification 字段）
            classification: healthy / rust_level_1~4 / red_spider_mite

输出（data/processed/）：
  labeled_S.csv       源域 BRACOL 的 healthy/rust（训练集）
  test_T.csv          目标域 RoCoLe 测试集（有标签，仅评估）
  ssl_pool_T.csv      目标域 RoCoLe 其余图（无标签，自监督预训练）
  ssl_pool_all.csv    BRACOL + RoCoLe 全量无标签（自监督预训练）
  bracol_5class.csv   BRACOL 五类（补充任务）

用法：python preprocess.py --seed 42 --test_ratio 0.3
"""
import csv
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

# BRACOL predominant_stress 映射（5=多病害混合，丢弃）
BRACOL_STRESS = {"0": "healthy", "1": "miner", "2": "rust", "3": "phoma", "4": "cercospora"}


def map_rocole(cls: str) -> Optional[str]:
    c = cls.strip().lower()
    if c == "healthy":
        return "healthy"
    if c.startswith("rust_level"):
        return "rust"
    return None  # red_spider_mite 丢弃


def parse_bracol() -> List[Dict]:
    root = RAW / "bracol"
    csv_path = root / "dataset.csv"
    img_dir = root / "images"
    if not csv_path.exists():
        raise FileNotFoundError("未找到 BRACOL dataset.csv: " + str(csv_path))
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fid = r.get("id", "").strip()
            img = img_dir / f"{fid}.jpg"
            if not img.exists():
                continue  # 缺图跳过（下载不完整）
            label = BRACOL_STRESS.get(r.get("predominant_stress", "").strip())
            if label is None:
                continue  # 丢弃多病害混合(5)
            rows.append({"image_path": str(img), "label": label,
                         "dataset": "bracol", "group": f"br_{fid}"})
    print("[INFO] BRACOL:", len(rows), "张")
    return rows


def parse_rocole() -> List[Dict]:
    root = RAW / "rocole"
    csv_path = root / "Annotations" / "RoCoLE-csv.csv"
    photos = root / "Photos"
    if not csv_path.exists():
        raise FileNotFoundError("未找到 RoCoLe 标注 CSV: " + str(csv_path))
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
                continue  # 丢弃 red_spider_mite
            rows.append({"image_path": str(img), "label": label,
                         "dataset": "rocole", "group": f"rc_{fn}"})
    print("[INFO] RoCoLe:", len(rows), "张 (healthy/rust)")
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

    S = parse_bracol()
    T = parse_rocole()

    S_bin = [r for r in S if r["label"] in ("healthy", "rust")]
    T_bin = [r for r in T if r["label"] in ("healthy", "rust")]

    # 目标域按 group 分层划分测试集
    groups = {}
    for r in T_bin:
        groups.setdefault(r["group"], r)
    group_list = list(groups.values())
    random.shuffle(group_list)
    n_test = int(len(group_list) * args.test_ratio)
    test_groups = {g["group"] for g in group_list[:n_test]}
    test_T, pool_T = [], []
    for r in T_bin:
        (test_T if r["group"] in test_groups else pool_T).append(r)

    for r in S_bin:
        r["split"] = "train"
    for r in test_T:
        r["split"] = "test"
    for r in pool_T:
        r["split"] = "unlabeled"

    ssl_all = S_bin + T_bin
    bracol_5 = [r for r in S if r["label"] in ("healthy", "miner", "rust", "phoma", "cercospora")]

    write_csv(OUT / "labeled_S.csv", S_bin)
    write_csv(OUT / "test_T.csv", test_T)
    write_csv(OUT / "ssl_pool_T.csv", pool_T)
    write_csv(OUT / "ssl_pool_all.csv", ssl_all)
    write_csv(OUT / "bracol_5class.csv", bracol_5)

    from collections import Counter
    print("\n===== 类别统计 =====")
    for name, rows in [("S源域(BRACOL)", S_bin), ("T测试(RoCoLe)", test_T),
                       ("T无标注池(RoCoLe)", pool_T)]:
        print(f"  {name}: {dict(Counter(r['label'] for r in rows))}")


if __name__ == "__main__":
    main()
