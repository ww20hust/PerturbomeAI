#!/usr/bin/env python3
"""End-to-end PerturbomeAI demo over several loci.

Runs the full pipeline on synthetic (or configured) data:
    synthesise -> quantile transform -> VAE impute -> 8-fold weighted XGBoost
    -> percentile align -> discrimination metrics -> proteomics association
    -> feature ablation.

All outputs land under ``output_dir`` from the config (default
``examples/demo_output``).

Usage:
    python scripts/run_demo.py [--config configs/pipeline.yaml]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from _pipeline import (  # noqa: E402  (local helper)
    REPO_ROOT,
    load_biobank,
    load_config,
    make_xgb_params,
    prepare_features,
    score_locus,
)

from perturbomeai import ablation, proteomics  # noqa: E402


def _resolve_out_dir(cfg: dict) -> Path:
    out = Path(cfg.get("output_dir", "examples/demo_output"))
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="PerturbomeAI end-to-end demo.")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = _resolve_out_dir(cfg)
    print(f"[demo] output dir: {out_dir}")

    bank = load_biobank(cfg)
    features_df = bank["features"]
    labels_df = bank["labels"]
    loci = bank["loci"]
    target_locus = bank["target_locus"]
    print(f"[demo] cohort N={len(features_df)} loci={loci} target={target_locus}")

    print("[demo] preparing features (quantile transform + VAE imputation)...")
    x, pids, feature_order = prepare_features(features_df, cfg)
    n_missing_before = int(features_df[feature_order].isna().to_numpy().sum())
    n_missing_after = int((~np.isfinite(x)).sum())
    print(f"[demo] missing feature cells: {n_missing_before} -> {n_missing_after} after imputation")

    pooled_rows: list[dict] = []
    fold_rows: list[dict] = []
    for locus in loci:
        result = score_locus(x, labels_df, pids, locus, cfg)
        pooled_rows.append(result["pooled"])
        for fm in result["fold_metrics"]:
            fold_rows.append({"locus": locus, **fm})
        result["score_df"].to_csv(out_dir / f"score_{locus}.csv", index=False)
        m = result["pooled"]
        print(
            f"[demo] {locus}: AUC={m['auc']:.3f} cohen_d={m['cohens_d']:.3f} "
            f"cliffs_delta={m['cliffs_delta']:.3f} p={m['mannwhitney_p']:.2e} "
            f"(n_pos={m['n_pos']}, n_neg={m['n_neg']})"
        )

    metrics_df = pd.DataFrame(pooled_rows)
    metrics_df.to_csv(out_dir / "locus_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out_dir / "locus_fold_metrics.csv", index=False)

    # Proteomics association on the target locus aligned score.
    proteomics_done = False
    if bank.get("proteomics") is not None:
        print(f"[demo] proteomics association on {target_locus}...")
        target_result = score_locus(x, labels_df, pids, target_locus, cfg)
        score_df = target_result["score_df"][["pid", "score"]]
        try:
            prot_cfg = cfg["proteomics"]
            results = proteomics.differential_proteins(
                score_df,
                bank["proteomics"],
                score_q_low=prot_cfg["score_q_low"],
                score_q_high=prot_cfg["score_q_high"],
                min_group_size=prot_cfg["min_group_size"],
            )
            results.to_csv(out_dir / f"proteomics_{target_locus}.csv", index=False)
            proteomics.volcano_plot(
                results,
                out_dir / f"proteomics_volcano_{target_locus}.png",
                fdr_threshold=prot_cfg["fdr_threshold"],
                label_top_n=prot_cfg["label_top_n"],
                title=f"{target_locus} score-driven proteome",
            )
            n_sig = int((results["p_adj_bh"] < prot_cfg["fdr_threshold"]).sum())
            print(f"[demo] proteomics: {n_sig} proteins FDR<{prot_cfg['fdr_threshold']}")
            proteomics_done = True
        except ValueError as exc:
            print(f"[demo] proteomics skipped: {exc}")

    # Ablation on the target locus.
    print(f"[demo] feature ablation on {target_locus}...")
    label_map = dict(zip(labels_df["pid"].astype(str), labels_df[target_locus]))
    y = np.array([label_map.get(pid, np.nan) for pid in pids], dtype=float)
    keep = np.isfinite(y)
    abl_table, top_features = ablation.run_ablation(
        x[keep],
        y[keep].astype(int),
        feature_order,
        top_k=cfg["ablation"]["top_k"],
        n_folds=cfg["scorer"]["n_folds"],
        seed=cfg.get("seed", 42),
        reference_fold=cfg["scorer"]["reference_fold"],
        params=make_xgb_params(cfg),
    )
    abl_table.to_csv(out_dir / f"ablation_{target_locus}.csv", index=False)
    full_auc = float(abl_table.loc[abl_table["scenario"] == "full", "auc"].iloc[0])
    all_auc = float(abl_table.loc[abl_table["scenario"] == "ablate_all", "auc"].iloc[0])
    print(f"[demo] ablation top features={top_features}; AUC full={full_auc:.3f} -> ablate_all={all_auc:.3f}")

    summary = {
        "n_samples": int(len(features_df)),
        "n_loci": len(loci),
        "auc_range": [float(metrics_df["auc"].min()), float(metrics_df["auc"].max())],
        "missing_cells_before": n_missing_before,
        "missing_cells_after": n_missing_after,
        "proteomics_done": proteomics_done,
        "ablation_full_auc": full_auc,
        "ablation_all_auc": all_auc,
        "ablation_top_features": top_features,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[demo] done. Summary: {json.dumps(summary, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
