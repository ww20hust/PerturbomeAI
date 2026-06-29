"""Feature ablation of the Genetic Perturbation Score.

To quantify how much each top feature contributes to the score, we rank
features by XGBoost gain on the full data, take the Top-K, and then re-run the
full 8-fold scoring for every subset of those K features that is dropped
(2^K scenarios, from ``full`` to ``ablate_all``). For each scenario we report
the four discrimination metrics, so the degradation curve isolates the
contribution of individual features and their combinations.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .align import align_oof_scores
from .features import feature_short_name
from .metrics import discrimination_metrics
from .scorer import XGBParams, feature_importance, oof_score


def build_ablation_configs(top_features: list[str]) -> list[tuple[str, list[str]]]:
    """All subsets of ``top_features`` to drop; named full / ablate_<...> / ablate_all."""
    k = len(top_features)
    configs: list[tuple[str, list[str]]] = []
    for r in range(k + 1):
        for dropped in itertools.combinations(top_features, r):
            dropped_list = list(dropped)
            if r == 0:
                scenario = "full"
            elif r == k:
                scenario = "ablate_all"
            else:
                short = sorted(feature_short_name(f) for f in dropped_list)
                scenario = "ablate_" + "_".join(short)
            configs.append((scenario, dropped_list))
    return configs


def top_k_features(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    top_k: int = 4,
    seed: int = 42,
    params: XGBParams | None = None,
) -> list[str]:
    """Rank features by gain importance and return the Top-K names."""
    importance = feature_importance(x, y, seed=seed, params=params)
    order = np.argsort(-importance)
    k = min(int(top_k), len(feature_names))
    return [feature_names[int(order[i])] for i in range(k)]


def run_ablation(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    top_k: int = 4,
    n_folds: int = 8,
    seed: int = 42,
    reference_fold: int = 0,
    params: XGBParams | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Run all 2^K ablation scenarios; return (metrics_table, top_features).

    The metrics table has one row per scenario with the dropped features and the
    four discrimination metrics (AUC, Cohen's d, Cliff's delta, Mann-Whitney p).
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y).astype(int)
    feature_names = list(feature_names)
    if x.shape[1] != len(feature_names):
        raise ValueError("feature_names length must match x columns.")

    top_features = top_k_features(x, y, feature_names, top_k=top_k, seed=seed, params=params)
    configs = build_ablation_configs(top_features)
    name_to_idx = {name: i for i, name in enumerate(feature_names)}

    rows: list[dict] = []
    for scenario, dropped in configs:
        keep = [i for name, i in name_to_idx.items() if name not in set(dropped)]
        if not keep:
            raise ValueError(f"Scenario {scenario}: no features left after drops.")
        res = oof_score(x[:, keep], y, n_folds=n_folds, seed=seed, params=params)
        aligned, _ = align_oof_scores(res.oof_raw, res.fold_ids, reference_fold=reference_fold)
        m = discrimination_metrics(aligned, y)
        rows.append(
            {
                "scenario": scenario,
                "n_dropped": len(dropped),
                "dropped": ",".join(feature_short_name(f) for f in dropped),
                "auc": m["auc"],
                "cohens_d": m["cohens_d"],
                "cliffs_delta": m["cliffs_delta"],
                "mannwhitney_p": m["mannwhitney_p"],
            }
        )
    table = pd.DataFrame(rows).sort_values(["n_dropped", "scenario"]).reset_index(drop=True)
    return table, top_features
