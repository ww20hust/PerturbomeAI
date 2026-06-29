"""Run simulation modules A, B, C."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config_loader import load_config, results_path
from .generative import (
    derive_b2_cohort,
    get_or_create_b1_50k,
    get_or_create_b3_50k,
)
from .inference import (
    assign_fold_ids,
    downsample_ratios,
    evaluate_all_oof_detailed,
    fit_xgb,
    fold_roles,
    reference_train_neg_count,
    reference_train_pos_count,
    stratified_downsample_indices,
    theoretical_n_neg_train,
    theoretical_n_pos_train,
    indices_for_folds,
)
from .metrics import fold_test_auc
from .plots import (
    plot_fig0_freq_effect_prior,
    plot_fig1_phenotype_space,
    plot_fig1_phenotype_space_prevalence,
    plot_fig2_auc_vs_freq,
    plot_fig3_causal_separation,
    plot_fig4_learning_curve,
)


def _metrics_to_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def run_module_a(cfg: dict | None = None) -> pd.DataFrame:
    if cfg is None:
        cfg = load_config()
    rng = np.random.default_rng(cfg["seed"])
    cohort = get_or_create_b1_50k(cfg, rng)
    rows, fold_rows = evaluate_all_oof_detailed(cohort, cfg)
    df = _metrics_to_df(rows)
    df_fold = _metrics_to_df(fold_rows)
    df.to_csv(results_path(cfg, "module_a_metrics.csv"), index=False)
    df_fold.to_csv(results_path(cfg, "module_a_fold_metrics.csv"), index=False)

    plot_fig1_phenotype_space(cfg, cohort, df)
    plot_fig1_phenotype_space_prevalence(cfg, cohort, df)
    plot_fig2_auc_vs_freq(cfg, cohort, df, df_fold)
    return df


def run_module_b(cfg: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if cfg is None:
        cfg = load_config()
    rng = np.random.default_rng(cfg["seed"])
    b1 = get_or_create_b1_50k(cfg, rng)
    b2 = derive_b2_cohort(b1, cfg["beta_med"])
    b3 = get_or_create_b3_50k(cfg, b1, rng)

    rows_b1, fold_b1 = evaluate_all_oof_detailed(b1, cfg)
    rows_b2, fold_b2 = evaluate_all_oof_detailed(b2, cfg)
    rows_b3, fold_b3 = evaluate_all_oof_detailed(b3, cfg)

    df_b1 = _metrics_to_df(rows_b1)
    df_b2 = _metrics_to_df(rows_b2)
    df_b3 = _metrics_to_df(rows_b3)
    df_fold_b1 = _metrics_to_df(fold_b1)
    df_fold_b2 = _metrics_to_df(fold_b2)
    df_fold_b3 = _metrics_to_df(fold_b3)

    df_b1.to_csv(results_path(cfg, "module_b1_metrics.csv"), index=False)
    df_b2.to_csv(results_path(cfg, "module_b2_metrics.csv"), index=False)
    df_b3.to_csv(results_path(cfg, "module_b3_metrics.csv"), index=False)
    df_fold_b1.to_csv(results_path(cfg, "module_b1_fold_metrics.csv"), index=False)
    df_fold_b2.to_csv(results_path(cfg, "module_b2_fold_metrics.csv"), index=False)
    df_fold_b3.to_csv(results_path(cfg, "module_b3_fold_metrics.csv"), index=False)

    plot_fig3_causal_separation(cfg, df_fold_b2, df_fold_b3)
    return df_b1, df_b2, df_b3


def run_module_c(cfg: dict | None = None) -> pd.DataFrame:
    if cfg is None:
        cfg = load_config()
    cohort = get_or_create_b1_50k(cfg, np.random.default_rng(cfg["seed"]))
    n_folds = cfg["cv"]["n_folds"]
    target_ids = set(cfg["module_c"]["variants"])

    rows = []
    for vi in range(cohort.m):
        vid = str(cohort.ids[vi])
        if vid not in target_ids:
            continue
        y = cohort.X[:, vi]
        fold_ids = assign_fold_ids(y, n_folds, cfg["seed"] + vi)
        n_pos_full_theory = reference_train_pos_count(cfg["module_a"]["n"], float(cohort.f[vi]))
        n_neg_full_theory = reference_train_neg_count(cfg["module_a"]["n"], float(cohort.f[vi]))

        for test_fold in range(n_folds):
            train_f, val_f, _ = fold_roles(test_fold, n_folds)
            tr_idx = indices_for_folds(fold_ids, train_f)
            va_idx = indices_for_folds(fold_ids, val_f)
            te_idx = indices_for_folds(fold_ids, [test_fold])

            Y_train_full = cohort.Y[tr_idx]
            y_train_full = y[tr_idx]
            Y_val = cohort.Y[va_idx]
            y_val = y[va_idx]
            Y_test = cohort.Y[te_idx]
            y_test = y[te_idx]

            n_pos_full = int((y_train_full == 1).sum())
            for ratio in downsample_ratios(n_pos_full):
                seed_pt = cfg["seed"] + vi + test_fold + int(ratio * 1e6)
                ds_idx = stratified_downsample_indices(
                    y_train_full, ratio, rng=np.random.default_rng(seed_pt),
                )
                Y_tr = Y_train_full[ds_idx]
                y_tr = y_train_full[ds_idx]
                n_pos = int((y_tr == 1).sum())
                n_neg = int((y_tr == 0).sum())
                n_pos_theory = theoretical_n_pos_train(ratio, n_pos_full_theory)
                n_neg_theory = theoretical_n_neg_train(ratio, n_neg_full_theory)
                model = fit_xgb(Y_tr, y_tr, Y_val, y_val, cfg, seed_pt)
                scores = model.predict_proba(Y_test)[:, 1]
                rows.append({
                    "variant_id": vid,
                    "f": float(cohort.f[vi]),
                    "beta": float(cohort.beta[vi]),
                    "downsample_ratio": ratio,
                    "n_pos_train": n_pos,
                    "n_pos_train_theory": n_pos_theory,
                    "n_neg_train": n_neg,
                    "n_neg_train_theory": n_neg_theory,
                    "test_fold": test_fold,
                    "auc": fold_test_auc(y_test, scores),
                })

    df = pd.DataFrame(rows)
    df.to_csv(results_path(cfg, "module_c_metrics.csv"), index=False)
    plot_fig4_learning_curve(cfg, df)
    return df


def run_fig0(cfg: dict | None = None) -> None:
    if cfg is None:
        cfg = load_config()
    plot_fig0_freq_effect_prior(cfg)


def run_all(cfg: dict | None = None) -> dict:
    if cfg is None:
        cfg = load_config()
    run_fig0(cfg)
    df_a = run_module_a(cfg)
    df_b1, df_b2, df_b3 = run_module_b(cfg)
    df_c = run_module_c(cfg)
    summary = {
        "module_a_auc_range": [float(df_a["auc"].min()), float(df_a["auc"].max())],
        "module_c_rows": len(df_c),
    }
    with open(results_path(cfg, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
