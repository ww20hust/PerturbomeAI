"""Discrimination metrics for carrier vs non-carrier scores.

We summarise how well a score separates carriers (label 1) from non-carriers
(label 0) with four complementary measures:

    - AUC          : ranking-based separation (threshold-free).
    - Cohen's d    : standardised mean difference (parametric effect size).
    - Cliff's delta: non-parametric effect size derived from Mann-Whitney U.
    - p-value      : two-sided Mann-Whitney U test of stochastic dominance.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score, roc_auc_score


def cohens_d(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    n1, n2 = len(scores_pos), len(scores_neg)
    if n1 < 2 or n2 < 2:
        return float("nan")
    m1, m2 = float(np.mean(scores_pos)), float(np.mean(scores_neg))
    s1, s2 = float(np.std(scores_pos, ddof=1)), float(np.std(scores_neg, ddof=1))
    pooled = np.sqrt(((n1 - 1) * s1 * s1 + (n2 - 1) * s2 * s2) / (n1 + n2 - 2))
    if pooled == 0:
        return float("nan")
    return (m1 - m2) / pooled


def _vda_from_u(u_stat: float, n1: int, n2: int) -> float:
    if n1 <= 0 or n2 <= 0:
        return float("nan")
    return u_stat / (n1 * n2)


def cliffs_delta_and_p(scores_pos: np.ndarray, scores_neg: np.ndarray) -> tuple[float, float]:
    """Return (Cliff's delta, two-sided Mann-Whitney p-value).

    Cliff's delta = 2 * VDA - 1 where VDA = U / (n1 * n2); a positive value means
    carriers tend to score higher than non-carriers.
    """
    n1, n2 = len(scores_pos), len(scores_neg)
    if n1 < 1 or n2 < 1:
        return float("nan"), float("nan")
    try:
        u_stat, pvalue = mannwhitneyu(scores_pos, scores_neg, alternative="two-sided")
    except ValueError:
        return float("nan"), float("nan")
    vda = _vda_from_u(float(u_stat), n1, n2)
    delta = 2.0 * vda - 1.0 if np.isfinite(vda) else float("nan")
    return delta, float(pvalue)


def cliffs_delta(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    return cliffs_delta_and_p(scores_pos, scores_neg)[0]


def median_diff(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    if len(scores_pos) < 1 or len(scores_neg) < 1:
        return float("nan")
    return float(np.median(scores_pos) - np.median(scores_neg))


def fold_test_auc(y_test: np.ndarray, scores_test: np.ndarray) -> float:
    y_test = np.asarray(y_test, dtype=np.int8)
    s = np.asarray(scores_test, dtype=np.float64)
    if int((y_test == 1).sum()) < 1 or int((y_test == 0).sum()) < 1:
        return float("nan")
    try:
        return float(roc_auc_score(y_test, s))
    except ValueError:
        return float("nan")


def discrimination_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """AUC, Cohen's d, Cliff's delta, Mann-Whitney p (plus counts and PR-AUC)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels)
    finite = np.isfinite(s) & np.isfinite(y.astype(float))
    s = s[finite]
    y = y[finite].astype(np.int8)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    out: dict[str, float] = {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "prevalence": n_pos / (n_pos + n_neg) if (n_pos + n_neg) else float("nan"),
        "auc": float("nan"),
        "pr_auc": float("nan"),
        "pr_auc_lift": float("nan"),
        "cohens_d": float("nan"),
        "cliffs_delta": float("nan"),
        "mannwhitney_p": float("nan"),
        "median_diff": float("nan"),
    }
    if n_pos < 1 or n_neg < 1:
        return out
    try:
        out["auc"] = float(roc_auc_score(y, s))
    except ValueError:
        pass
    try:
        pr = float(average_precision_score(y, s))
        out["pr_auc"] = pr
        if out["prevalence"] > 0:
            out["pr_auc_lift"] = pr / out["prevalence"]
    except ValueError:
        pass
    pos_s = s[y == 1]
    neg_s = s[y == 0]
    out["cohens_d"] = cohens_d(pos_s, neg_s)
    delta, pval = cliffs_delta_and_p(pos_s, neg_s)
    out["cliffs_delta"] = delta
    out["mannwhitney_p"] = pval
    out["median_diff"] = median_diff(pos_s, neg_s)
    return out
