"""Quantile-align OOF scores to a reference CV fold distribution."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import rankdata


def align_fold_scores(
    scores_t: np.ndarray,
    ref_scores: np.ndarray,
    *,
    ref_sorted: np.ndarray | None = None,
) -> np.ndarray:
    if ref_sorted is None:
        ref_sorted = np.sort(ref_scores.astype(np.float64))
    else:
        ref_sorted = ref_sorted.astype(np.float64)
    n_ref = len(ref_sorted)
    scores_t = scores_t.astype(np.float64)
    n_t = len(scores_t)
    if n_ref < 2 or n_t < 1:
        return scores_t.copy()

    ranks = rankdata(scores_t, method="average")
    if n_t == 1:
        u = np.array([0.5], dtype=np.float64)
    else:
        u = (ranks - 1.0) / (n_t - 1.0)

    q_pos = u * (n_ref - 1)
    lo = np.floor(q_pos).astype(int)
    hi = np.ceil(q_pos).astype(int)
    w = q_pos - lo
    aligned = (1.0 - w) * ref_sorted[lo] + w * ref_sorted[hi]
    return aligned.astype(np.float64)


def align_oof_scores(
    scores: np.ndarray,
    cv_folds: np.ndarray,
    *,
    reference_fold: int = 0,
) -> Tuple[np.ndarray, str]:
    out = scores.copy().astype(np.float64)
    ref_mask = cv_folds == int(reference_fold)
    ref_scores = scores[ref_mask]
    ref_finite = ref_scores[np.isfinite(ref_scores)]
    if len(ref_finite) < 2:
        return out, "skipped"

    ref_sorted = np.sort(ref_finite.astype(np.float64))
    for fold_id in np.unique(cv_folds):
        fold_id = int(fold_id)
        if fold_id == int(reference_fold):
            continue
        m = cv_folds == fold_id
        s = scores[m]
        finite = np.isfinite(s)
        if not finite.any():
            continue
        idx = np.flatnonzero(m)
        out[idx[finite]] = align_fold_scores(s[finite], ref_finite, ref_sorted=ref_sorted)
    return out, "ok"
