"""Theoretical d' and linear AUC reference."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .generative import Cohort


def sigma_i(cohort: Cohort, target: int) -> np.ndarray:
    """2x2 covariance of Y given X_target (shared for both classes)."""
    sigma2 = cohort.sigma ** 2
    cov = np.eye(2) * sigma2
    for k in range(cohort.m):
        if k == target:
            continue
        vk = cohort.v[k]
        cov += (cohort.beta[k] ** 2) * cohort.f[k] * (1.0 - cohort.f[k]) * np.outer(vk, vk)
    return cov


def d_prime(cohort: Cohort, target: int) -> float:
    vi = cohort.v[target]
    sig = sigma_i(cohort, target)
    denom = np.sqrt(vi @ sig @ vi)
    if denom <= 0:
        return np.nan
    return abs(cohort.beta[target]) / denom


def auc_linear_theory(cohort: Cohort, target: int) -> float:
    dp = d_prime(cohort, target)
    if not np.isfinite(dp):
        return np.nan
    return float(norm.cdf(dp / np.sqrt(2.0)))


def theory_for_all_variants(cohort: Cohort) -> np.ndarray:
    return np.array([auc_linear_theory(cohort, i) for i in range(cohort.m)])
