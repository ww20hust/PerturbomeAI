"""8-fold cross-validation with a 5 / 2 / 1 train / validation / test rotation.

Each sample is assigned once to one of 8 stratified folds. For every rotation we
designate exactly:
    - 5 folds for training,
    - 2 folds for validation (used for XGBoost early stopping),
    - 1 fold for test (held out; scored out-of-fold).

Rotating the test fold across all 8 positions means every sample receives
exactly one out-of-fold score. The 8-fold choice is deliberate: it maps one
locus's 8 cross-validation models one-to-one onto 8 GPUs (see ``hpc/``).
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold

N_FOLDS = 8


def assign_fold_ids(y: np.ndarray, n_folds: int = N_FOLDS, seed: int = 42) -> np.ndarray:
    """Assign each sample to a fold in ``0..n_folds-1`` via stratified K-fold."""
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_ids = np.full(y.shape[0], -1, dtype=np.int32)
    dummy_x = np.zeros((len(y), 1))
    for fold_id, (_, val_idx) in enumerate(skf.split(dummy_x, y)):
        fold_ids[val_idx] = fold_id
    return fold_ids


def fold_roles(test_fold: int, n_folds: int = N_FOLDS) -> tuple[list[int], list[int], list[int]]:
    """Return (train_folds, val_folds, test_folds) for a given test fold.

    5 train, 2 validation, 1 test, defined cyclically relative to ``test_fold``.
    """
    train = [(test_fold + i) % n_folds for i in range(1, 6)]
    val = [(test_fold + 6) % n_folds, (test_fold + 7) % n_folds]
    test = [test_fold]
    return train, val, test


def indices_for_folds(fold_ids: np.ndarray, folds: list[int]) -> np.ndarray:
    """Indices of samples whose fold id is in ``folds``."""
    return np.flatnonzero(np.isin(fold_ids, folds))
