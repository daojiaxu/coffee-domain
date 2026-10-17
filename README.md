# Coffee UDA Benchmark

**Unsupervised Domain Adaptation for Cross-Domain Coffee Leaf Disease Recognition: A Systematic Benchmark of Methods, Directions, and Task Difficulty**

Official code release accompanying the paper by *Jiaxu Dao and Zhuoying Li* (School of Technology, Pu'er University).

This repository provides a systematic, fair, and reproducible benchmark of unsupervised domain adaptation (UDA) for coffee leaf disease recognition across two visually distinct image domains: the laboratory BRACOL collection (white-background, controlled illumination) and the field RoCoLe collection (natural backgrounds, uncontrolled illumination). Seven methods are compared over **three benchmark tasks × two transfer directions × seven random seeds**, under both an oracle best-epoch protocol and a label-free fixed-epoch protocol, with paired significance testing.

## Key findings

- **UDA is effective in the forward direction.** All six adaptation methods significantly outperform the source-only baseline on lab→field healthy/rust transfer (paired Wilcoxon, p ≤ 0.047).
- **Transfer is asymmetric.** In the reverse direction (field→lab) the domain gap is much smaller and adaptation yields no significant benefit; SHOT even hurts (−6.0 pp).
- **A task-difficulty gradient.** The adaptation gain grows with task difficulty for five of six methods (healthy/rust → healthy/disease → severity), with Deep CORAL as the notable exception.
- **SHOT is the most label-free-stable**, while CW-DANN (class-weighted conditional DANN with focal loss) is most robust to class imbalance.

## Datasets

Both datasets are public and hosted on Mendeley Data (manual download required):

| Dataset | Domain | Images | Link |
|---|---|---|---|
| BRACOL | Laboratory (white background) | 1,747 | https://data.mendeley.com/datasets/yy2k5y8mxg/1 |
| RoCoLe | Field (natural background) | 1,560 | https://data.mendeley.com/datasets/c5yvn32dzg/2 |

`code/download_data.py` checks that both datasets are in place and prints the expected directory layout. The raw images are **not** shipped with this repository.

## Repository structure

```
.
├── code/                       # all source code
│   ├── download_data.py        # dataset download guidance / checks
│   ├── preprocess.py           # build per-split CSV indices from raw data
│   ├── preprocess_resize.py    # build 256px cached thumbnails (optional)
│   ├── split_*.py              # task-configuration split generators
│   ├── uda_common.py           # shared data loading / evaluation / metrics I/O
│   ├── train_supervised.py     # P2  source-only baseline
│   ├── train_dann.py           # DANN
│   ├── train_cwdann.py         # CW-DANN (conditional + focal)
│   ├── train_deepcoral.py      # Deep CORAL
│   ├── train_dan.py            # DAN (MK-MMD)
│   ├── train_cdan.py           # CDAN
│   ├── train_shot.py           # SHOT (source-free)
│   ├── train_5class.py         # 5-class experiment
│   ├── train_moco.py           # MoCo self-supervised reference point
│   ├── train_simclr.py         # SimCLR self-supervised reference point
│   ├── evaluate_*.py           # balanced-accuracy / severity evaluation
│   ├── analyze_*.py            # paired Wilcoxon analyses
│   ├── plot_*.py               # paper-figure generators
│   ├── gradcam_visualize.py    # Grad-CAM visualizations
│   └── tsne_visualize.py       # t-SNE embeddings
├── data/
│   └── processed/              # 22 split-index CSVs (train / SSL-pool / test)
└── outputs/
    └── results/                # 268 per-seed metrics.json (paper numbers)
```

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 1.12 with CUDA (a single RTX 3090 24 GB was used; a CPU fallback works but is slow)
- See `requirements.txt` for the full list.

## Quick start

```bash
# 1. Download BRACOL and RoCoLe (Mendeley, manual) and verify
python code/download_data.py

# 2. Build split indices
python code/preprocess.py

# 3. (Optional) Build 256px thumbnails to speed up training I/O
python code/preprocess_resize.py
```

The split-index CSVs for all six task configurations are already included under `data/processed/` so that the exact paper splits can be reused without regenerating them. To regenerate a split from scratch:

```bash
python code/split_reverse.py --seed 42 --test_ratio 0.3     # reverse healthy/rust
python code/split_healthydisease.py --seed 42               # healthy/disease
python code/split_severity.py --seed 42                     # severity
```

## Training

Each method follows the same CLI: a labeled source index (`--src`), an unlabeled target pool (`--tgt`), and a target test set (`--test`). Example for DANN on the forward healthy/rust task:

```bash
python code/train_dann.py \
  --src data/processed/labeled_S.csv \
  --tgt data/processed/ssl_pool_T.csv \
  --test data/processed/test_T.csv \
  --seed 42 --tag P7_dann_s42
```

Task-configuration CSV mapping:

| Task | Direction | `--src` | `--tgt` | `--test` |
|---|---|---|---|---|
| healthy/rust | lab→field | `labeled_S.csv` | `ssl_pool_T.csv` | `test_T.csv` |
| healthy/rust | field→lab | `labeled_S_rev.csv` | `ssl_pool_T_rev.csv` | `test_T_rev.csv` |
| healthy/disease | lab→field | `labeled_S_hd.csv` | `ssl_pool_T_hd.csv` | `test_T_hd.csv` |
| healthy/disease | field→lab | `labeled_S_hd_rev.csv` | `ssl_pool_T_hd_rev.csv` | `test_T_hd_rev.csv` |
| severity (1–4) | lab→field | `labeled_S_sev.csv` | `ssl_pool_T_sev.csv` | `test_T_sev.csv` |
| severity (1–4) | field→lab | `labeled_S_sev_rev.csv` | `ssl_pool_T_sev_rev.csv` | `test_T_sev_rev.csv` |

Method-specific notes:

- **P2** (`train_supervised.py`) uses `--train`/`--test` (no target pool).
- **CW-DANN** (`train_cwdann.py`) adds `--focal --conditional --gamma 2`.
- **Deep CORAL** (`train_deepcoral.py`) adds `--lambda_coral 100`; **DAN** (`train_dan.py`) adds `--lambda_mmd 1`.
- **SHOT** (`train_shot.py`) uses `--src_epochs 50 --adapt_epochs 15` instead of `--epochs`.

`run_sota_extended.py` orchestrates the full sweep (7 methods × 7 seeds × task configurations) with checkpoint-resume.

## Reproducing the paper results

All per-seed metrics are stored in `outputs/results/*_metrics.json` and aggregate exactly to the mean ± std and paired Wilcoxon p-values reported in the paper. Recompute them with:

```bash
python code/analyze_results.py          # forward healthy/rust main table
python code/analyze_sota_extended.py    # reverse + healthy/disease tables
python code/analyze_hd_rev.py           # reverse healthy/disease balanced accuracy
python code/evaluate_severity.py        # severity accuracy + MAE
python code/evaluate_balanced_sota.py   # balanced accuracy on imbalanced test sets
```

## Citation

```bibtex
@article{dao2026coffee,
  title  = {Unsupervised Domain Adaptation for Cross-Domain Coffee Leaf Disease
            Recognition: A Systematic Benchmark of Methods, Directions, and
            Task Difficulty},
  author = {Dao, Jiaxu and Li, Zhuoying},
  journal= {Agriculture},
  year   = {2026},
  doi    = {10.3390/agriculture0000000}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
