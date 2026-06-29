"""Evaluation metrics for the simulation chapter.

Mirrors the main package metrics: AUC, Cohen's d, Cliff's delta and the
two-sided Mann-Whitney U p-value, plus PR-AUC and median difference.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score, roc_auc_score


def cohens_d(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    n1, n2 = len(scores_pos), len(scores_neg)
    if n1 < 2 or n2 < 2:
        return np.nan
    m1, m2 = float(np.mean(scores_pos)), float(np.mean(scores_neg))
    s1, s2 = float(np.std(scores_pos, ddof=1)), float(np.std(scores_neg, ddof=1))
    pooled = np.sqrt(((n1 - 1) * s1 * s1 + (n2 - 1) * s2 * s2) / (n1 + n2 - 2))
    if pooled == 0:
        return np.nan
    return (m1 - m2) / pooled


def vda_from_u(u_stat: float, n1: int, n2: int) -> float:
    if n1 <= 0 or n2 <= 0:
        return np.nan
    return u_stat / (n1 * n2)


def cliffs_delta_from_vda(vda: float) -> float:
    if np.isnan(vda):
        return np.nan
    return 2 * vda - 1


def cliffs_delta_and_p(scores_pos: np.ndarray, scores_neg: np.ndarray) -> tuple[float, float]:
    """Return (Cliff's delta, two-sided Mann-Whitney U p-value)."""
    n1, n2 = len(scores_pos), len(scores_neg)
    if n1 < 1 or n2 < 1:
        return np.nan, np.nan
    try:
        u_stat, pvalue = mannwhitneyu(scores_pos, scores_neg, alternative="two-sided")
    except ValueError:
        return np.nan, np.nan
    vda = vda_from_u(float(u_stat), n1, n2)
    return cliffs_delta_from_vda(vda), float(pvalue)


def cliffs_delta(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    return cliffs_delta_and_p(scores_pos, scores_neg)[0]


def median_diff(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    if len(scores_pos) < 1 or len(scores_neg) < 1:
        return np.nan
    return float(np.median(scores_pos) - np.median(scores_neg))


def fold_test_auc(y_test: np.ndarray, scores_test: np.ndarray) -> float:
    y_test = np.asarray(y_test, dtype=np.int8)
    s = np.asarray(scores_test, dtype=np.float64)
    n_pos = int((y_test == 1).sum())
    n_neg = int((y_test == 0).sum())
    if n_pos < 1 or n_neg < 1:
        return np.nan
    try:
        return float(roc_auc_score(y_test, s))
    except ValueError:
        return np.nan


def binary_metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    out = {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "prevalence": n_pos / (n_pos + n_neg) if (n_pos + n_neg) else np.nan,
        "auc": np.nan,
        "pr_auc": np.nan,
        "pr_auc_lift": np.nan,
        "cohens_d": np.nan,
        "cliffs_delta": np.nan,
        "mannwhitney_p": np.nan,
        "median_diff": np.nan,
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
