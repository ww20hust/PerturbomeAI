"""One-shot 8-fold cohort assignment shared by every locus.

The genome-scale efficiency trick starts here: instead of re-splitting the
cohort for each of hundreds of thousands of loci, we assign every individual a
single ``cv_test_fold in 0..7`` once. The same split is reused for all loci, so
fold membership can be precomputed and broadcast, and the 8 fold-models of any
locus share identical train/val/test partitions (mapping cleanly onto 8 GPUs).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from perturbomeai.folds import fold_roles  # noqa: E402

from . import paths  # noqa: E402


def assign_cohort_folds(n_samples: int, n_folds: int = 8, seed: int = 42) -> np.ndarray:
    """Assign each sample one fold id in ``0..n_folds-1`` via a shuffled even split."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_samples)
    fold_ids = np.empty(n_samples, dtype=np.int32)
    fold_ids[order] = np.arange(n_samples) % n_folds
    return fold_ids


def build_train_val_map(n_folds: int = 8) -> dict[str, dict[str, list[int]]]:
    """For each test fold, the 5 train / 2 val / 1 test fold assignment."""
    out: dict[str, dict[str, list[int]]] = {}
    for test_fold in range(n_folds):
        train, val, test = fold_roles(test_fold, n_folds)
        out[str(test_fold)] = {"train": train, "val": val, "test": test}
    return out


def save_cohort(fold_ids: np.ndarray, n_folds: int = 8) -> None:
    np.save(paths.cv_folds_path(), fold_ids)
    with open(paths.train_val_map_path(), "w", encoding="utf-8") as f:
        json.dump(build_train_val_map(n_folds), f, indent=2)


def load_fold_ids() -> np.ndarray:
    return np.load(paths.cv_folds_path())


def load_train_val_map() -> dict[str, dict[str, list[int]]]:
    with open(paths.train_val_map_path(), encoding="utf-8") as f:
        return json.load(f)
