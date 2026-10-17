# -*- coding: utf-8 -*-
"""Verify CDAN per-seed healthy-class recall on hd_rev (CPU)."""
import torch
from pathlib import Path
from torch.utils.data import DataLoader

import evaluate_balanced_sota as ev

ROOT = ev.ROOT
RES = ev.RES
device = torch.device("cpu")
test_ds = ev.TestDS(ROOT / "data" / "processed" / "test_T_hd_rev.csv")
loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

print("=== CDAN hd_rev per-seed ===")
r0s = []
for s in ev.SEEDS:
    ckpt = RES / f"CDAN_hd_rev_s{s}_best.pth"
    if not ckpt.exists():
        print(f"[MISS] {s}")
        continue
    model = ev.build("cdan", ckpt).to(device)
    acc, bal, recall = ev.evaluate(model, loader, device, "cdan")
    r0 = recall[0] * 100
    r0s.append(r0)
    print(f"  s{s}: acc={acc*100:.1f} bal={bal*100:.1f} recall_healthy={r0:.1f} recall_disease={recall[1]*100:.1f}")
print(f"\nrecall_healthy range = {min(r0s):.1f}% .. {max(r0s):.1f}%  (mean {sum(r0s)/len(r0s):.1f}%)")
