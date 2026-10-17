# -*- coding: utf-8 -*-
"""
split_severity.py — rust 严重度 4 级（序回归）跨域数据划分（正向 + 反向）。

两侧都有锈病严重度 1~4 级标注，语义可对齐：
  BRACOL: predominant_stress=2(rust) 的 severity 列 1/2/3/4 → level1~4
  RoCoLe: classification=rust_level_1~4 → level1~4

任务：rust 严重度 4 类分类（序回归，可另报 MAE）。

输出两套域对：
  正向 BRACOL(lab)→RoCoLe(field):
    labeled_S_sev.csv      源域 BRACOL rust 训练（severity 1~4）
    test_T_sev.csv         目标域 RoCoLe rust 测试集
    ssl_pool_T_sev.csv     目标域 RoCoLe rust 无标签池
  反向 RoCoLe(field)→BRACOL(lab):
    labeled_S_sev_rev.csv  源域 RoCoLe rust 训练（rust_level 1~4）
    test_T_sev_rev.csv     目标域 BRACOL rust 测试集
    ssl_pool_T_sev_rev.csv 目标域 BRACOL rust 无标签池

划分（与其它 split 脚本一致）：目标域按 group（独立叶片）分层，30% 测试 / 70% 无标签池。

用法：python split_severity.py --seed 42 --test_ratio 0.3
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


def map_bracol_sev(ps: str, sev: str) -> Optional[str]:
    """仅取 rust(predominant_stress=2) 的 severity 1~4。"""
    if ps.strip() != "2":
        return None
    s = sev.strip()
    if s in ("1", "2", "3", "4"):
        return f"level{s}"
    return None


def map_rocole_sev(cls: str) -> Optional[str]:
    c = cls.strip().lower()
    if c == "rust_level_1":
        return "level1"
    if c == "rust_level_2":
        return "level2"
    if c == "rust_level_3":
        return "level3"
    if c == "rust_level_4":
        return "level4"
    return None


def parse_bracol_sev() -> List[Dict]:
    csv_path = RAW / "bracol" / "dataset.csv"
    img_dir = RAW / "bracol" / "images"
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fid = r.get("id", "").strip()
            img = img_dir / f"{fid}.jpg"
            if not img.exists():
                continue
            label = map_bracol_sev(r.get("predominant_stress", ""),
                                   r.get("severity", ""))
            if label is None:
                continue
            rows.append({"image_path": str(img), "label": label,
                         "dataset": "bracol", "group": f"br_{fid}"})
    print("[INFO] BRACOL rust severity 1-4:", len(rows), "张")
    return rows


def parse_rocole_sev() -> List[Dict]:
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
            label = map_rocole_sev(cls)
            if label is None:
                continue
            rows.append({"image_path": str(img), "label": label,
                         "dataset": "rocole", "group": f"rc_{fn}"})
    print("[INFO] RoCoLe rust severity 1-4:", len(rows), "张")
    return rows


def split_target(T: List[Dict], test_ratio: float, rng: random.Random):
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

    B = parse_bracol_sev()
    R = parse_rocole_sev()

    # 正向 BRACOL(lab) -> RoCoLe(field)
    S_fwd = [dict(r) for r in B]
    test_fwd, pool_fwd = split_target(R, args.test_ratio, rng)
    for r in S_fwd:
        r["split"] = "train"
    for r in test_fwd:
        r["split"] = "test"
    for r in pool_fwd:
        r["split"] = "unlabeled"
    write_csv(OUT / "labeled_S_sev.csv", S_fwd)
    write_csv(OUT / "test_T_sev.csv", test_fwd)
    write_csv(OUT / "ssl_pool_T_sev.csv", pool_fwd)

    # 反向 RoCoLe(field) -> BRACOL(lab)
    S_rev = [dict(r) for r in R]
    test_rev, pool_rev = split_target(B, args.test_ratio, rng)
    for r in S_rev:
        r["split"] = "train"
    for r in test_rev:
        r["split"] = "test"
    for r in pool_rev:
        r["split"] = "unlabeled"
    write_csv(OUT / "labeled_S_sev_rev.csv", S_rev)
    write_csv(OUT / "test_T_sev_rev.csv", test_rev)
    write_csv(OUT / "ssl_pool_T_sev_rev.csv", pool_rev)

    print("\n===== rust severity 1-4 类别统计 =====")
    for name, rows in [
        ("正向 源域S(BRACOL)", S_fwd), ("正向 目标T测试(RoCoLe)", test_fwd),
        ("正向 目标T无标注(RoCoLe)", pool_fwd),
        ("反向 源域S(RoCoLe)", S_rev), ("反向 目标T测试(BRACOL)", test_rev),
        ("反向 目标T无标注(BRACOL)", pool_rev),
    ]:
        print(f"  {name}: {dict(Counter(r['label'] for r in rows))}")


if __name__ == "__main__":
    main()
