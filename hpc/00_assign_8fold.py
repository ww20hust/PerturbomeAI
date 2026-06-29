#!/usr/bin/env python3
"""Stage 0: one-shot 8-fold cohort assignment (shared by every locus).

Writes ``cohort/cv_folds.npy`` (per-sample fold id) and
``cohort/train_val_map.json`` (the 5/2/1 train/val/test rotation).

Usage:
    python 00_assign_8fold.py --n-samples 4000 --n-folds 8 --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HPC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HPC_DIR))

from hpc_lib import cohort  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="8-fold cohort assignment.")
    parser.add_argument("--n-samples", type=int, default=int(os.environ.get("N_SAMPLES", 4000)))
    parser.add_argument("--n-folds", type=int, default=int(os.environ.get("N_FOLDS", 8)))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 42)))
    args = parser.parse_args()

    fold_ids = cohort.assign_cohort_folds(args.n_samples, args.n_folds, args.seed)
    cohort.save_cohort(fold_ids, args.n_folds)
    counts = [int((fold_ids == f).sum()) for f in range(args.n_folds)]
    print(f"[assign] N={args.n_samples} folds={args.n_folds} per-fold counts={counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
