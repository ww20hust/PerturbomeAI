#!/usr/bin/env python3
"""Stage 1: build the shared feature memmap and per-block label memmaps.

Synthesises a large locus matrix with ``perturbomeai.simdata`` (so the chapter
runs without private genotype data), prepares the shared imputed feature matrix
once, and partitions loci into blocks. The feature matrix is written a single
time as a float32 memmap and read-only thereafter (no repeated parsing per locus).

Usage:
    python 01_build_block_cache.py --n-samples 4000 --n-loci 40 --block-size 20
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HPC_DIR = Path(__file__).resolve().parent
REPO_ROOT = HPC_DIR.parent
sys.path.insert(0, str(HPC_DIR))
sys.path.insert(0, str(REPO_ROOT))

from hpc_lib import block_cache  # noqa: E402
from perturbomeai import simdata  # noqa: E402
from perturbomeai.features import INPUT_FEATURES  # noqa: E402
from perturbomeai.preprocess import fit_column_quantile_transform  # noqa: E402


def _prepare_shared_features(features_df, seed: int) -> np.ndarray:
    """Quantile-transform and median-impute the shared feature matrix (fast path)."""
    qt = fit_column_quantile_transform(features_df, INPUT_FEATURES, random_state=seed)
    qt_df = qt.transform(features_df)
    x = qt_df[INPUT_FEATURES].to_numpy(dtype=np.float32)
    col_median = np.nanmedian(x, axis=0)
    col_median = np.where(np.isfinite(col_median), col_median, 0.0)
    idx = np.where(~np.isfinite(x))
    x[idx] = np.take(col_median, idx[1])
    return x


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the HPC block cache.")
    parser.add_argument("--n-samples", type=int, default=int(os.environ.get("N_SAMPLES", 4000)))
    parser.add_argument("--n-loci", type=int, default=int(os.environ.get("N_LOCI", 40)))
    parser.add_argument("--block-size", type=int, default=int(os.environ.get("BLOCK_SIZE", 20)))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 42)))
    args = parser.parse_args()

    bank = simdata.make_biobank(
        n_samples=args.n_samples,
        n_loci=args.n_loci,
        n_proteins=2,  # proteins unused here; keep tiny
        seed=args.seed,
    )
    x = _prepare_shared_features(bank.features, args.seed)
    labels = bank.labels[bank.locus_names].to_numpy(dtype=np.int8)
    manifest = block_cache.build_block_cache(
        x, labels, list(bank.locus_names), block_size=args.block_size
    )
    print(
        f"[block_cache] X={manifest['n_samples']}x{manifest['n_features']} "
        f"loci={manifest['n_loci']} blocks={manifest['n_blocks']} "
        f"(block_size={manifest['block_size']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
