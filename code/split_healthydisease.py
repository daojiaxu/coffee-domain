# -*- coding: utf-8 -*-
"""
split_healthydisease.py — healthy vs disease 聚合二分类的数据划分（正向 + 反向）。

把之前被丢弃的病害类别全部聚合进 "disease" 类，最大化数据利用率，并贴近真实
部署语义（农户只关心"叶子上有没有病"，而非具体哪种病）：

  BRACOL: predominant_stress 0=healthy → healthy；1(miner)/2(rust)/3(phoma)/4(cercospora) → disease；5(多病害混合) 丢弃
  RoCoLe: classification healthy → healthy；rust_level_1~4 + red_spider_mite → disease

输出两套域对（均为 healthy/disease 二分类）：
  正向 BRACOL(lab)→RoCoLe(field):
    labeled_S_hd.csv      源域 BRACOL 训练（有标签）
    test_T_hd.csv         目标域 RoCoLe 测试集（有标签，仅评估）
    ssl_pool_T_hd.csv     目标域 RoCoLe 无标签池
  反向 RoCoLe(field)→BRACOL(lab):
    labeled_S_hd_rev.csv  源域 RoCoLe 训练（有标签）
    test_T_hd_rev.csv     目标域 BRACOL 测试集
    ssl_pool_T_hd_rev.csv 目标域 BRACOL 无标签池

划分（与 preprocess.py / split_reverse.py 一致）：目标域按 group（独立叶片）分层，
30% 测试集 / 70% 无标签池；源域全部作训练集。

用法：python split_healthydisease.py --seed 42 --test_ratio 0.3
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

# BRACOL predominant_stress → healthy/disease（5=多病害混合，丢弃）
BRACOL_HD = {"0": "healthy", "1": "disease", "2": "disease",
             "3": "disease", "4": "disease"}


def map_rocole_hd(cls: str) -> Optional[str]:
    c = cls.strip().lower()
    if c == "healthy":
        return "healthy"
    if c.startswith("rust_level"):
        return "disease"
    if c == "red_spider_mite":
        return "disease"
    return None


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
            label = BRACOL_HD.get(r.get("predominant_stress", "").strip())
            if label is None:
                continue
            rows.append({"image_path": str(img), "label": label,
                         "dataset": "bracol", "group": f"br_{fid}"})
    print("[INFO] BRACOL (healthy/disease):", len(rows), "张")
    return rows


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
                cls = json.loads(r.get("Label", "{}")).get("classification", "")
            except Exception:
                cls = ""
            label = map_rocole_hd(cls)
            if label is None:
                continue
            rows.append({"image_path": str(img), "label": label,
                         "dataset": "rocole", "group": f"rc_{fn}"})
    print("[INFO] RoCoLe (healthy/disease):", len(rows), "张")
    return rows


def split_target(T: List[Dict], test_ratio: float, rng: random.Random):
    """目标域按 group 分层，返回 (test_rows, pool_rows)。"""
    groups = {}
    for r in T:
        groups.setdefault(r["group"], r)
    group_list = list(groups.values())
    rng.shuffle(group_list)
    n_test = int(len(group_list) * test_ratio)
    test_groups = {g["group"] for g in group_list[:n_test]}
    test_rows, pool_rows = [], []
    for r in T:
        (test_rows if r["group"] in test_groups else pool_rows).append(r)
    return test_rows, pool_rows


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
    rng = random.Random(args.seed)

    B = parse_bracol()
    R = parse_rocole()

    # ---- 正向 BRACOL(lab) -> RoCoLe(field) ----
    S_fwd = [dict(r) for r in B]
    test_fwd, pool_fwd = split_target(R, args.test_ratio, rng)
    for r in S_fwd:
        r["split"] = "train"
    for r in test_fwd:
        r["split"] = "test"
    for r in pool_fwd:
        r["split"] = "unlabeled"
    write_csv(OUT / "labeled_S_hd.csv", S_fwd)
    write_csv(OUT / "test_T_hd.csv", test_fwd)
    write_csv(OUT / "ssl_pool_T_hd.csv", pool_fwd)

    # ---- 反向 RoCoLe(field) -> BRACOL(lab) ----
    S_rev = [dict(r) for r in R]
    test_rev, pool_rev = split_target(B, args.test_ratio, rng)
    for r in S_rev:
        r["split"] = "train"
    for r in test_rev:
        r["split"] = "test"
    for r in pool_rev:
        r["split"] = "unlabeled"
    write_csv(OUT / "labeled_S_hd_rev.csv", S_rev)
    write_csv(OUT / "test_T_hd_rev.csv", test_rev)
    write_csv(OUT / "ssl_pool_T_hd_rev.csv", pool_rev)

    print("\n===== healthy vs disease 类别统计 =====")
    for name, rows in [
        ("正向 源域S(BRACOL)", S_fwd), ("正向 目标T测试(RoCoLe)", test_fwd),
        ("正向 目标T无标注(RoCoLe)", pool_fwd),
        ("反向 源域S(RoCoLe)", S_rev), ("反向 目标T测试(BRACOL)", test_rev),
        ("反向 目标T无标注(BRACOL)", pool_rev),
    ]:
        print(f"  {name}: {dict(Counter(r['label'] for r in rows))}")


if __name__ == "__main__":
    main()
