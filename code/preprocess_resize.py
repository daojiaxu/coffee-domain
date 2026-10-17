# -*- coding: utf-8 -*-
"""
preprocess_resize.py — 把原始大图等比缩放到短边 256 的缓存小图，大幅加速训练数据加载。

背景：BRACOL/RoCoLe 原图为 ~1024x2048 大图，PIL 解码+缩放约 28ms/张；
      256 小图仅约 0.5ms/张（快 ~50 倍）。训练时读小图可消除数据加载瓶颈。

缓存输出：data/processed/imgs256/{bracol|rocole}/{原文件名}.jpg（保持宽高比，短边 256）
"""
import os
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "imgs256"

SRC = {
    "bracol": RAW / "bracol" / "images",
    "rocole": RAW / "rocole" / "Photos",
}


def main():
    total = 0
    for ds, src_dir in SRC.items():
        if not src_dir.exists():
            print(f"[SKIP] 源目录不存在: {src_dir}")
            continue
        dst_dir = OUT / ds
        dst_dir.mkdir(parents=True, exist_ok=True)
        files = [f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for f in files:
            dst = dst_dir / f"{Path(f).stem}.jpg"
            if dst.exists():
                continue  # 已缓存
            im = Image.open(src_dir / f).convert("RGB")
            w, h = im.size
            scale = 256 / min(w, h)
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
            im.save(dst, "JPEG", quality=90)
            total += 1
        print(f"[{ds}] 源图 {len(files)} 张，本次新增缓存 {total} 张")
    print("[DONE] 缓存小图生成完成")


if __name__ == "__main__":
    main()
