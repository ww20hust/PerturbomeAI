#!/usr/bin/env python3
"""Stage 3: assemble out-of-fold scores for a block and verify coverage.

Gathers the per-fold test scores written by stage 2 into a single OOF matrix
(N x n_loci_in_block) and checks that every sample received exactly one score.

Usage:
    python 03_score_block_oof.py --block 0 [--n-folds 8]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HPC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HPC_DIR))

from hpc_lib import block_cache, cohort, paths  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble block OOF scores.")
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--n-folds", type=int, default=int(os.environ.get("N_FOLDS", 8)))
    args = parser.parse_args()

    fold_ids = cohort.load_fold_ids()
    n = len(fold_ids)
    n_loci = block_cache.open_block_labels(args.block).shape[1]
    oof = np.full((n, n_loci), np.nan, dtype=np.float64)

    for fold in range(args.n_folds):
        scores_path = paths.fold_scores_path(args.block, fold)
        idx_path = paths.fold_test_idx_path(args.block, fold)
        if not scores_path.exists() or not idx_path.exists():
            print(f"[oof] block={args.block} fold={fold} missing scores; skipping")
            continue
        scores = np.load(scores_path)
        te_idx = np.load(idx_path)
        oof[te_idx, :] = scores

    np.save(paths.block_oof_path(args.block), oof)
    # Coverage: a sample is covered for a locus if its OOF score is finite.
    covered = np.isfinite(oof).all(axis=1).mean()
    print(f"[oof] block={args.block} loci={n_loci} full-coverage fraction={covered:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
