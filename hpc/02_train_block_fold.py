#!/usr/bin/env python3
"""Stage 2: the smallest job - train every locus in one (block, fold).

Loads the shared feature memmap read-only plus the block's label memmap, masks
train / validation / test rows via the 5/2/1 fold map, and trains the weighted
XGBoost (scale_pos_weight + sample_weight, early stopping on the 2 validation
folds) for every locus in the block on this single fold. The held-out test fold
is scored immediately and saved, and the wall time is recorded for the
throughput projection.

This is the unit of work the GPU wave pool schedules: the 8 folds of one block
run as 8 of these jobs, one per GPU.

Usage (normally launched by the wave pool):
    python 02_train_block_fold.py --block 0 --fold 0 [--device cpu|cuda]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

HPC_DIR = Path(__file__).resolve().parent
REPO_ROOT = HPC_DIR.parent
sys.path.insert(0, str(HPC_DIR))
sys.path.insert(0, str(REPO_ROOT))

from hpc_lib import block_cache, cohort, paths, throughput  # noqa: E402
from perturbomeai import scorer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train one (block, fold).")
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", type=str, default=os.environ.get("PERTURBOMEAI_DEVICE", "cpu"))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 42)))
    args = parser.parse_args()

    t0 = time.time()
    x = block_cache.open_feature_memmap()            # read-only, shared
    labels = block_cache.open_block_labels(args.block)
    fold_ids = cohort.load_fold_ids()
    tv_map = cohort.load_train_val_map()
    roles = tv_map[str(args.fold)]

    tr_idx = np.flatnonzero(np.isin(fold_ids, roles["train"]))
    va_idx = np.flatnonzero(np.isin(fold_ids, roles["val"]))
    te_idx = np.flatnonzero(np.isin(fold_ids, roles["test"]))

    # Materialise the needed rows once for this job (memmap file is never reparsed).
    x_tr = np.asarray(x[tr_idx], dtype=np.float32)
    x_va = np.asarray(x[va_idx], dtype=np.float32)
    x_te = np.asarray(x[te_idx], dtype=np.float32)

    params = scorer.XGBParams(device=args.device)
    n_loci = labels.shape[1]
    test_scores = np.full((len(te_idx), n_loci), np.nan, dtype=np.float64)

    for j in range(n_loci):
        y = np.asarray(labels[:, j], dtype=int)
        y_tr, y_va = y[tr_idx], y[va_idx]
        if (y_tr == 1).sum() < 1 or (y_tr == 0).sum() < 1:
            continue
        if (y_va == 1).sum() < 1 or (y_va == 0).sum() < 1:
            continue
        n_pos = int((y_tr == 1).sum())
        n_neg = int((y_tr == 0).sum())
        spw = n_neg / n_pos if n_pos > 0 else 1.0
        model = scorer.build_xgb(spw, random_state=args.seed + args.fold + j, params=params)
        scorer.fit_weighted(model, x_tr, y_tr, x_va, y_va, spw)
        test_scores[:, j] = model.predict_proba(x_te)[:, 1]

    np.save(paths.fold_scores_path(args.block, args.fold), test_scores)
    np.save(paths.fold_test_idx_path(args.block, args.fold), te_idx)
    elapsed = time.time() - t0
    throughput.record_timing(args.block, args.fold, n_loci, elapsed)
    print(
        f"[train] block={args.block} fold={args.fold} device={args.device} "
        f"loci={n_loci} test_n={len(te_idx)} {elapsed:.2f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
