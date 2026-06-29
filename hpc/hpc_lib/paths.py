"""Result-root and artifact path helpers.

The result root comes from the ``PERTURBOMEAI_HPC_ROOT`` environment variable
(set by ``scripts/env.sh``) and defaults to ``hpc/hpc_result`` inside the repo,
so a run writes everything under one configurable directory.
"""

from __future__ import annotations

import os
from pathlib import Path

HPC_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = HPC_DIR.parent


def result_root() -> Path:
    env = os.environ.get("PERTURBOMEAI_HPC_ROOT")
    root = Path(env) if env else HPC_DIR / "hpc_result"
    root.mkdir(parents=True, exist_ok=True)
    return root


# --- cohort ---------------------------------------------------------------
def cohort_dir() -> Path:
    d = result_root() / "cohort"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cv_folds_path() -> Path:
    return cohort_dir() / "cv_folds.npy"


def train_val_map_path() -> Path:
    return cohort_dir() / "train_val_map.json"


# --- shared feature memmap -------------------------------------------------
def feature_dir() -> Path:
    d = result_root() / "features"
    d.mkdir(parents=True, exist_ok=True)
    return d


def x_memmap_path() -> Path:
    return feature_dir() / "X_float32.dat"


def x_shape_path() -> Path:
    return feature_dir() / "X_shape.json"


# --- blocks ----------------------------------------------------------------
def blocks_dir() -> Path:
    d = result_root() / "blocks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def block_dir(block_id: int) -> Path:
    d = blocks_dir() / f"block_{block_id:04d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def block_labels_path(block_id: int) -> Path:
    return block_dir(block_id) / "labels_int8.dat"


def block_labels_shape_path(block_id: int) -> Path:
    return block_dir(block_id) / "labels_shape.json"


def block_locus_ids_path(block_id: int) -> Path:
    return block_dir(block_id) / "locus_ids.json"


def block_manifest_path() -> Path:
    return blocks_dir() / "block_manifest.json"


def fold_scores_path(block_id: int, fold: int) -> Path:
    return block_dir(block_id) / f"fold{fold}_test_scores.npy"


def fold_test_idx_path(block_id: int, fold: int) -> Path:
    return block_dir(block_id) / f"fold{fold}_test_idx.npy"


def block_oof_path(block_id: int) -> Path:
    return block_dir(block_id) / "oof_raw.npy"


# --- merged outputs --------------------------------------------------------
def merged_dir() -> Path:
    d = result_root() / "merged"
    d.mkdir(parents=True, exist_ok=True)
    return d


def merged_metrics_path() -> Path:
    return merged_dir() / "locus_metrics.csv"


def timing_dir() -> Path:
    d = result_root() / "timing"
    d.mkdir(parents=True, exist_ok=True)
    return d


def timing_path(block_id: int, fold: int) -> Path:
    return timing_dir() / f"block{block_id:04d}_fold{fold}.json"
