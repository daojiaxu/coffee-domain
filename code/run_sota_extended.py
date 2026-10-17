# -*- coding: utf-8 -*-
"""
run_sota_extended.py — 把 4 个 SOTA 基线方法补跑到「反向域对」与「healthy/disease 正向」，
以验证方向不对称 / 任务难度梯度两条结论在 SOTA 方法下是否依然成立。

任务矩阵（2 任务 × 4 方法 × 7 种子 = 56 次训练）：
  A. 反向 field->lab（healthy/rust）：labeled_S_rev / ssl_pool_T_rev / test_T_rev
  B. 正向 lab->field（healthy/disease）：labeled_S_hd / ssl_pool_T_hd / test_T_hd

方法：DeepCORAL / DAN / CDAN / SHOT（与正向主任务完全同协议同脚本同种子）。

特点：可断点续跑（已有 *_metrics.json 的跳过），全程追加日志到
outputs/sota_extended.log。
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
RES = ROOT / "outputs" / "results"
LOG = ROOT / "outputs" / "sota_extended.log"

PY = r"E:\Anaconda3\envs\llm_RE\python.exe"
SEEDS = [42, 123, 2024, 7, 99, 13, 777]

METHODS = {
    "DeepCORAL": "train_deepcoral.py",
    "DAN": "train_dan.py",
    "CDAN": "train_cdan.py",
    "SHOT": "train_shot.py",
}

TASKS = [
    {  # A. 反向 field->lab, healthy/rust
        "name": "rev",
        "src": "labeled_S_rev.csv",
        "tgt": "ssl_pool_T_rev.csv",
        "test": "test_T_rev.csv",
        "labels": None,           # 默认 healthy,rust
    },
    {  # B. 正向 lab->field, healthy/disease
        "name": "hd",
        "src": "labeled_S_hd.csv",
        "tgt": "ssl_pool_T_hd.csv",
        "test": "test_T_hd.csv",
        "labels": "healthy,disease",
    },
    {  # C. 反向 field->lab, healthy/disease
        "name": "hd_rev",
        "src": "labeled_S_hd_rev.csv",
        "tgt": "ssl_pool_T_hd_rev.csv",
        "test": "test_T_hd_rev.csv",
        "labels": "healthy,disease",
    },
]


def log(msg: str):
    line = f"[{__import__('datetime').datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    RES.mkdir(parents=True, exist_ok=True)
    total = done = 0
    for task in TASKS:
        for method, script in METHODS.items():
            for seed in SEEDS:
                tag = f"{method}_{task['name']}_s{seed}"
                total += 1
                metrics = RES / f"{tag}_metrics.json"
                if metrics.exists():
                    log(f"[SKIP] {tag} 已存在，跳过")
                    done += 1
                    continue
                cmd = [
                    PY, str(CODE / script),
                    "--src", str(ROOT / "data" / "processed" / task["src"]),
                    "--tgt", str(ROOT / "data" / "processed" / task["tgt"]),
                    "--test", str(ROOT / "data" / "processed" / task["test"]),
                    "--seed", str(seed),
                    "--tag", tag,
                ]
                if method == "SHOT":
                    cmd += ["--src_epochs", "50", "--adapt_epochs", "15"]
                else:
                    cmd += ["--epochs", "50"]
                if task["labels"]:
                    cmd += ["--labels", task["labels"]]
                log(f"[RUN {done+1}/{total}] {' '.join(cmd[-8:])}")
                r = subprocess.run(cmd, cwd=str(CODE), capture_output=True, text=True,
                                   encoding="utf-8", errors="replace")
                out = r.stdout + r.stderr
                with open(LOG, "a", encoding="utf-8") as f:
                    f.write(out)
                if r.returncode != 0:
                    log(f"[FAIL] {tag} 退出码 {r.returncode}")
                    tail = "\n".join(out.strip().splitlines()[-15:])
                    log(tail)
                else:
                    done += 1
                    log(f"[OK] {tag} 完成 ({done}/{total})")
    log(f"=== ALL DONE {done}/{total} ===")


if __name__ == "__main__":
    main()
