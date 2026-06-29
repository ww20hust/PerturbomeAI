"""XGBoost inference with 8-fold CV (5 train / 2 val / 1 test) and score alignment."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from .features import expand_phenotype
from .metrics import binary_metrics, fold_test_auc
from .score_align import align_oof_scores


def _classifier_cfg(cfg: dict) -> dict:
    return cfg["classifier"]


def fold_roles(test_fold: int, n_folds: int = 8) -> tuple[list[int], list[int], list[int]]:
    """Assign 5 train, 2 val, 1 test folds relative to test_fold."""
    train = [(test_fold + i) % n_folds for i in range(1, 6)]
    val = [(test_fold + 6) % n_folds, (test_fold + 7) % n_folds]
    test = [test_fold]
    return train, val, test


class PhenotypeXGB:
    """XGBoost on expanded phenotype features [Y1, Y2, r, Y1*Y2]."""

    def __init__(self, cfg: dict, seed: int):
        c = _classifier_cfg(cfg)
        self._clf = XGBClassifier(
            max_depth=c["max_depth"],
            n_estimators=c["n_estimators"],
            learning_rate=c["learning_rate"],
            subsample=c["subsample"],
            colsample_bytree=c["colsample_bytree"],
            objective="binary:logistic",
            eval_metric=c.get("eval_metric", "logloss"),
            early_stopping_rounds=c["early_stopping_rounds"],
            random_state=seed,
            n_jobs=1,
        )
        self._scale_pos_weight: float = 1.0

    def fit(
        self,
        Y_train: np.ndarray,
        y_train: np.ndarray,
        Y_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "PhenotypeXGB":
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        if n_pos > 0:
            self._scale_pos_weight = n_neg / n_pos
        else:
            self._scale_pos_weight = 1.0
        self._clf.set_params(scale_pos_weight=self._scale_pos_weight)
        sw_tr = np.where(y_train == 1, self._scale_pos_weight, 1.0)
        sw_va = np.where(y_val == 1, self._scale_pos_weight, 1.0)
        X_tr = expand_phenotype(Y_train)
        X_va = expand_phenotype(Y_val)
        self._clf.fit(
            X_tr, y_train,
            sample_weight=sw_tr,
            eval_set=[(X_va, y_val)],
            sample_weight_eval_set=[sw_va],
            verbose=False,
        )
        return self

    def predict_proba(self, Y: np.ndarray) -> np.ndarray:
        X = expand_phenotype(Y)
        return self._clf.predict_proba(X)


def assign_fold_ids(y: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Assign each sample to fold 0..n_folds-1 via stratified K-fold."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_ids = np.full(y.shape[0], -1, dtype=np.int32)
    dummy_X = np.zeros((len(y), 1))
    for fold_id, (_, val_idx) in enumerate(skf.split(dummy_X, y)):
        fold_ids[val_idx] = fold_id
    return fold_ids


def indices_for_folds(fold_ids: np.ndarray, folds: list[int]) -> np.ndarray:
    return np.flatnonzero(np.isin(fold_ids, folds))


def _indices_for_folds(fold_ids: np.ndarray, folds: list[int]) -> np.ndarray:
    return indices_for_folds(fold_ids, folds)


def fit_xgb(
    Y_train: np.ndarray,
    y_train: np.ndarray,
    Y_val: np.ndarray,
    y_val: np.ndarray,
    cfg: dict,
    seed: int,
) -> PhenotypeXGB:
    return PhenotypeXGB(cfg, seed).fit(Y_train, y_train, Y_val, y_val)


def oof_predict(
    Y: np.ndarray,
    y: np.ndarray,
    cfg: dict,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """8-fold OOF: 5 train / 2 val / 1 test, score-aligned to fold 0."""
    n_folds = cfg["cv"]["n_folds"]
    fold_ids = assign_fold_ids(y, n_folds, seed)
    oof_raw = np.full(y.shape[0], np.nan, dtype=np.float64)
    test_fold_ids = np.full(y.shape[0], -1, dtype=np.int32)
    fold_metrics: list[dict] = []

    for test_fold in range(n_folds):
        train_f, val_f, _ = fold_roles(test_fold, n_folds)
        tr_idx = _indices_for_folds(fold_ids, train_f)
        va_idx = _indices_for_folds(fold_ids, val_f)
        te_idx = _indices_for_folds(fold_ids, [test_fold])
        if (y[tr_idx] == 1).sum() < 1 or (y[tr_idx] == 0).sum() < 1:
            continue
        if (y[va_idx] == 1).sum() < 1 or (y[va_idx] == 0).sum() < 1:
            continue
        model = fit_xgb(
            Y[tr_idx], y[tr_idx], Y[va_idx], y[va_idx], cfg, seed + test_fold,
        )
        te_scores = model.predict_proba(Y[te_idx])[:, 1]
        oof_raw[te_idx] = te_scores
        test_fold_ids[te_idx] = test_fold
        y_te = y[te_idx]
        fold_metrics.append({
            "fold": test_fold,
            "auc": fold_test_auc(y_te, te_scores),
            "n_pos_test": int((y_te == 1).sum()),
            "n_neg_test": int((y_te == 0).sum()),
        })

    aligned, _ = align_oof_scores(oof_raw, test_fold_ids, reference_fold=0)
    return aligned, test_fold_ids, fold_metrics


def median_boundary_probs(
    Y: np.ndarray,
    y: np.ndarray,
    grid_Y: np.ndarray,
    cfg: dict,
    seed: int,
) -> np.ndarray:
    """Train 8 models (5 train + 2 val each); return median predict_proba on grid."""
    n_folds = cfg["cv"]["n_folds"]
    fold_ids = assign_fold_ids(y, n_folds, seed)
    all_probs = []
    for test_fold in range(n_folds):
        train_f, val_f, _ = fold_roles(test_fold, n_folds)
        tr_idx = _indices_for_folds(fold_ids, train_f)
        va_idx = _indices_for_folds(fold_ids, val_f)
        if (y[tr_idx] == 1).sum() < 1 or (y[tr_idx] == 0).sum() < 1:
            continue
        if (y[va_idx] == 1).sum() < 1 or (y[va_idx] == 0).sum() < 1:
            continue
        model = fit_xgb(
            Y[tr_idx], y[tr_idx], Y[va_idx], y[va_idx], cfg, seed + test_fold,
        )
        all_probs.append(model.predict_proba(grid_Y)[:, 1])
    if not all_probs:
        raise ValueError("No valid folds for median boundary")
    return np.median(np.stack(all_probs, axis=0), axis=0)


def knn_median_oof_scores(
    Y: np.ndarray,
    oof_scores: np.ndarray,
    grid_Y: np.ndarray,
    k: int = 1000,
) -> np.ndarray:
    """Grid score = median aligned OOF among k nearest phenotype-space neighbors."""
    n = len(Y)
    k = min(int(k), n)
    tree = cKDTree(np.asarray(Y, dtype=np.float64))
    _, idx = tree.query(np.asarray(grid_Y, dtype=np.float64), k=k)
    if k == 1:
        idx = np.asarray(idx)[:, np.newaxis]
    return np.median(oof_scores[idx], axis=1)


def boundary_top_f_for_freq(f: float) -> float:
    """Tiered top-f fraction for prevalence boundary by variant frequency."""
    f = float(f)
    if f < 0.002:
        return 0.05
    if f < 0.007:
        return 0.15
    if f < 0.03:
        return 0.30
    return 0.45


def mean_prevalence_boundary_field(
    Y: np.ndarray,
    y: np.ndarray,
    grid_Y: np.ndarray,
    f: float,
    cfg: dict,
    seed: int,
    knn_k: int = 1000,
) -> np.ndarray:
    """Tiered top-f boundary on KNN median of aligned OOF grid scores."""
    aligned, _, _ = oof_predict(Y, y, cfg, seed)
    grid_scores = knn_median_oof_scores(Y, aligned, grid_Y, k=knn_k)
    top_f = boundary_top_f_for_freq(f)
    return prevalence_matched_labels(grid_scores, top_f)


def prevalence_matched_labels(scores: np.ndarray, f: float) -> np.ndarray:
    """Top-f grid scores -> 1 (carrier), remainder -> 0."""
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return scores
    f = float(np.clip(f, 1.0 / n, 1.0 - 1.0 / n))
    thresh = np.quantile(scores, 1.0 - f)
    return (scores >= thresh).astype(np.float64)


def theoretical_n_pos_train(ratio: float, n_pos_full: int) -> int:
    """Theoretical train positive count after stratified downsample."""
    return max(1, int(np.floor(ratio * n_pos_full)))


def theoretical_n_neg_train(ratio: float, n_neg_full: int) -> int:
    """Theoretical train negative count after stratified downsample."""
    return int(np.floor(ratio * n_neg_full))


def reference_train_pos_count(n_cohort: int, f: float, train_frac: float = 5 / 8) -> int:
    """Expected carrier count in 5/8 train pool (design frequency f)."""
    return max(1, int(np.floor(n_cohort * train_frac * f)))


def reference_train_neg_count(n_cohort: int, f: float, train_frac: float = 5 / 8) -> int:
    """Expected non-carrier count in 5/8 train pool (design frequency f)."""
    n_train = int(np.floor(n_cohort * train_frac))
    return n_train - reference_train_pos_count(n_cohort, f, train_frac)


def downsample_ratios(n_pos_full: int) -> list[float]:
    """Halving ratios from 1.0 until train positive count reaches 1."""
    if n_pos_full <= 1:
        return [1.0]
    ratios = [1.0]
    r = 0.5
    while True:
        ratios.append(r)
        n_pos = max(1, int(np.floor(r * n_pos_full)))
        if n_pos <= 1:
            break
        r /= 2
    return ratios


def stratified_downsample_indices(
    y: np.ndarray,
    ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stratified downsample indices; n_pos = max(1, floor(ratio * n_pos))."""
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    n_pos = max(1, int(np.floor(len(pos_idx) * ratio)))
    n_neg = int(np.floor(len(neg_idx) * ratio))
    sel_pos = rng.choice(pos_idx, size=n_pos, replace=False)
    sel_neg = rng.choice(neg_idx, size=n_neg, replace=False) if n_neg > 0 else np.array([], dtype=int)
    return np.concatenate([sel_pos, sel_neg])


def evaluate_variant_oof_detailed(cohort, variant_idx: int, cfg: dict) -> tuple[dict, list[dict]]:
    y = cohort.X[:, variant_idx]
    scores, _, fold_rows = oof_predict(cohort.Y, y, cfg, cfg["seed"] + variant_idx)
    pooled = binary_metrics(y, scores)
    pooled["variant_id"] = str(cohort.ids[variant_idx])
    pooled["f"] = float(cohort.f[variant_idx])
    pooled["beta"] = float(cohort.beta[variant_idx])
    meta = {
        "variant_id": pooled["variant_id"],
        "f": pooled["f"],
        "beta": pooled["beta"],
    }
    for row in fold_rows:
        row.update(meta)
    return pooled, fold_rows


def evaluate_all_oof_detailed(cohort, cfg: dict) -> tuple[list[dict], list[dict]]:
    pooled_rows: list[dict] = []
    fold_rows: list[dict] = []
    for i in range(cohort.m):
        pooled, folds = evaluate_variant_oof_detailed(cohort, i, cfg)
        pooled_rows.append(pooled)
        fold_rows.extend(folds)
    return pooled_rows, fold_rows


def evaluate_variant_oof(cohort, variant_idx: int, cfg: dict) -> dict:
    pooled, _ = evaluate_variant_oof_detailed(cohort, variant_idx, cfg)
    return pooled


def evaluate_all_oof(cohort, cfg: dict) -> list[dict]:
    pooled, _ = evaluate_all_oof_detailed(cohort, cfg)
    return pooled
