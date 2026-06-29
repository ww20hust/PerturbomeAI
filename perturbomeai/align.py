"""Percentile alignment of out-of-fold scores to a reference fold.

Each of the 8 test folds is scored by a different model, so the raw score
distributions can differ slightly between folds. To make the pooled
Genetic Perturbation Score comparable across individuals, we map every fold's
scores onto the score distribution of a single reference fold using an
empirical mid-rank percentile (quantile) transform.

A sample with empirical rank percentile ``u`` in its own fold is assigned the
value at percentile ``u`` of the reference fold. This preserves within-fold
ranking while equalising the marginal distributions.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def align_fold_scores(
    scores_t: np.ndarray,
    ref_sorted: np.ndarray,
) -> np.ndarray:
    """Map ``scores_t`` onto the sorted reference distribution by rank percentile."""
    ref_sorted = np.asarray(ref_sorted, dtype=np.float64)
    scores_t = np.asarray(scores_t, dtype=np.float64)
    n_ref = len(ref_sorted)
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
    return (1.0 - w) * ref_sorted[lo] + w * ref_sorted[hi]


def align_oof_scores(
    scores: np.ndarray,
    fold_ids: np.ndarray,
    *,
    reference_fold: int = 0,
) -> tuple[np.ndarray, str]:
    """Align every fold's scores to the reference fold's distribution.

    Returns (aligned_scores, status) where status is "ok" or "skipped" (when the
    reference fold has fewer than two finite scores).
    """
    scores = np.asarray(scores, dtype=np.float64)
    fold_ids = np.asarray(fold_ids)
    out = scores.copy()

    ref_mask = fold_ids == int(reference_fold)
    ref_scores = scores[ref_mask]
    ref_finite = ref_scores[np.isfinite(ref_scores)]
    if len(ref_finite) < 2:
        return out, "skipped"

    ref_sorted = np.sort(ref_finite.astype(np.float64))
    for fold_id in np.unique(fold_ids):
        fold_id = int(fold_id)
        if fold_id == int(reference_fold) or fold_id < 0:
            continue
        m = fold_ids == fold_id
        s = scores[m]
        finite = np.isfinite(s)
        if not finite.any():
            continue
        idx = np.flatnonzero(m)
        out[idx[finite]] = align_fold_scores(s[finite], ref_sorted)
    return out, "ok"
