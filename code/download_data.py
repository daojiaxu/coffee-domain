# -*- coding: utf-8 -*-
"""
download_data.py — 数据下载指引（两域方案：BRACOL + RoCoLe）。

两个数据集均在 Mendeley Data，需浏览器登录后手动下载（脚本只做存在性检查与路径提示）。
  - BRACOL 原始版：https://data.mendeley.com/datasets/yy2k5y8mxg/1
  - RoCoLe：        https://data.mendeley.com/datasets/c5yvn32dzg/2

下载后解压结构（本脚本会自动核对）：
  data/raw/bracol/    dataset.csv + images/*.jpg
  data/raw/rocole/    Annotations/RoCoLE-csv.csv + Photos/*.jpg

用法：python download_data.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

CHECKS = {
    "BRACOL": {
        "url": "https://data.mendeley.com/datasets/yy2k5y8mxg/1",
        "need": [RAW / "bracol" / "dataset.csv", RAW / "bracol" / "images"],
    },
    "RoCoLe": {
        "url": "https://data.mendeley.com/datasets/c5yvn32dzg/2",
        "need": [RAW / "rocole" / "Annotations" / "RoCoLE-csv.csv", RAW / "rocole" / "Photos"],
    },
}


def main():
    ok = True
    for name, cfg in CHECKS.items():
        missing = [p for p in cfg["need"] if not p.exists()]
        if missing:
            ok = False
            print(f"[缺少] {name} 未就绪，请从 {cfg['url']} 手动下载并解压到 data/raw/")
            for m in missing:
                print(f"       缺少: {m.relative_to(ROOT) if m.is_relative_to(ROOT) else m}")
        else:
            # 顺带统计图片数
            img_dir = RAW / "bracol" / "images" if name == "BRACOL" else RAW / "rocole" / "Photos"
            n = len(list(img_dir.glob("*.jpg")))
            print(f"[就绪] {name} 已就绪（{n} 张图）")
    if not ok:
        sys.exit(1)
    print("\n全部数据集就绪，可运行: python code/preprocess.py")


if __name__ == "__main__":
    main()
