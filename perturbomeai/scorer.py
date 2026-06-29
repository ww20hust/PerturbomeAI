"""Weighted XGBoost out-of-fold scoring (the Genetic Perturbation Score).

For a given locus we treat carriers vs non-carriers as a binary problem and
learn the multivariate phenotypic signature with XGBoost. Class imbalance is
handled with a weighted loss:

    scale_pos_weight = n_neg / n_pos   (computed from the training folds only)

and the same weight is applied as per-sample weights on both the training and
the validation sets, so early stopping optimises the imbalance-aware objective.

``oof_score`` runs the 8-fold 5/2/1 rotation, fitting one model per test fold
(early stopping on the 2 validation folds) and scoring the held-out test fold.
It returns the raw out-of-fold scores, the test-fold id per sample, and the
per-fold test AUC. Percentile alignment of the raw scores is a separate step
(see :mod:`perturbomeai.align`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from xgboost import XGBClassifier

from .folds import N_FOLDS, assign_fold_ids, fold_roles, indices_for_folds
from .metrics import fold_test_auc


@dataclass
class XGBParams:
    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 3.0
    gamma: float = 0.1
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 30
    device: str = "cpu"  # "cpu" or "cuda"
    n_jobs: int = 1
    extra: dict = field(default_factory=dict)


def _device_params(device: str) -> dict:
    """Map a device string to XGBoost tree params, robust to version differences."""
    if device == "cuda":
        # xgboost >= 2.0 prefers device="cuda"; older releases use gpu_hist.
        try:
            import xgboost as xgb

            major = int(xgb.__version__.split(".")[0])
        except Exception:
            major = 2
        if major >= 2:
            return {"tree_method": "hist", "device": "cuda"}
        return {"tree_method": "gpu_hist"}
    return {"tree_method": "hist"}


def build_xgb(scale_pos_weight: float, random_state: int, params: XGBParams | None = None) -> XGBClassifier:
    p = params or XGBParams()
    kwargs = dict(
        n_estimators=p.n_estimators,
        max_depth=p.max_depth,
        learning_rate=p.learning_rate,
        subsample=p.subsample,
        colsample_bytree=p.colsample_bytree,
        min_child_weight=p.min_child_weight,
        gamma=p.gamma,
        reg_alpha=p.reg_alpha,
        reg_lambda=p.reg_lambda,
        scale_pos_weight=float(scale_pos_weight),
        objective="binary:logistic",
        eval_metric="aucpr",
        early_stopping_rounds=int(p.early_stopping_rounds) if p.early_stopping_rounds and p.early_stopping_rounds > 0 else None,
        random_state=int(random_state),
        n_jobs=int(p.n_jobs),
        verbosity=0,
    )
    kwargs.update(_device_params(p.device))
    kwargs.update(p.extra)
    return XGBClassifier(**kwargs)


def fit_weighted(
    model: XGBClassifier,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    scale_pos_weight: float,
) -> XGBClassifier:
    """Fit with positive-weighted samples and early stopping on the validation set."""
    sw_tr = np.where(y_tr == 1, scale_pos_weight, 1.0)
    sw_va = np.where(y_val == 1, scale_pos_weight, 1.0)
    fit_kwargs = dict(sample_weight=sw_tr, eval_set=[(x_val, y_val)], verbose=False)
    try:
        model.fit(x_tr, y_tr, sample_weight_eval_set=[sw_va], **fit_kwargs)
    except TypeError:
        # Older XGBoost without sample_weight_eval_set support.
        model.fit(x_tr, y_tr, **fit_kwargs)
    return model


@dataclass
class OOFResult:
    oof_raw: np.ndarray         # (N,) out-of-fold scores, NaN where unscored
    fold_ids: np.ndarray        # (N,) test-fold id per sample, -1 if unscored
    fold_metrics: list[dict]    # per test-fold AUC + counts


def oof_score(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_folds: int = N_FOLDS,
    seed: int = 42,
    params: XGBParams | None = None,
) -> OOFResult:
    """8-fold 5/2/1 out-of-fold scoring with a weighted XGBoost per fold."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y).astype(int)
    fold_ids = assign_fold_ids(y, n_folds=n_folds, seed=seed)
    oof_raw = np.full(y.shape[0], np.nan, dtype=np.float64)
    test_fold_ids = np.full(y.shape[0], -1, dtype=np.int32)
    fold_metrics: list[dict] = []

    for test_fold in range(n_folds):
        train_f, val_f, _ = fold_roles(test_fold, n_folds)
        tr_idx = indices_for_folds(fold_ids, train_f)
        va_idx = indices_for_folds(fold_ids, val_f)
        te_idx = indices_for_folds(fold_ids, [test_fold])
        y_tr = y[tr_idx]
        y_va = y[va_idx]
        if (y_tr == 1).sum() < 1 or (y_tr == 0).sum() < 1:
            continue
        if (y_va == 1).sum() < 1 or (y_va == 0).sum() < 1:
            continue
        n_pos = int((y_tr == 1).sum())
        n_neg = int((y_tr == 0).sum())
        spw = n_neg / n_pos if n_pos > 0 else 1.0
        model = build_xgb(spw, random_state=seed + test_fold + 1, params=params)
        fit_weighted(model, x[tr_idx], y_tr, x[va_idx], y_va, spw)
        te_scores = model.predict_proba(x[te_idx])[:, 1]
        oof_raw[te_idx] = te_scores
        test_fold_ids[te_idx] = test_fold
        y_te = y[te_idx]
        fold_metrics.append(
            {
                "fold": test_fold,
                "auc": fold_test_auc(y_te, te_scores),
                "n_pos_test": int((y_te == 1).sum()),
                "n_neg_test": int((y_te == 0).sum()),
            }
        )
    return OOFResult(oof_raw=oof_raw, fold_ids=test_fold_ids, fold_metrics=fold_metrics)


def feature_importance(x: np.ndarray, y: np.ndarray, *, seed: int = 42, params: XGBParams | None = None) -> np.ndarray:
    """Gain-based feature importance from a single full-data weighted XGBoost."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y).astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"Need both classes for importance: pos={n_pos}, neg={n_neg}")
    spw = n_neg / n_pos
    p = params or XGBParams()
    # No early stopping for the full-data fit.
    p_full = XGBParams(**{**p.__dict__, "early_stopping_rounds": 0})
    model = build_xgb(spw, random_state=seed, params=p_full)
    sw = np.where(y == 1, spw, 1.0)
    model.fit(x, y, sample_weight=sw, verbose=False)
    return np.asarray(model.feature_importances_, dtype=float)
