#!/usr/bin/env python3
"""Stage 4: feature ablation of the score for one locus.

Ranks features by gain, then re-runs the full 8-fold scoring for every subset of
the Top-K features dropped, reporting the four discrimination metrics per
scenario.

Usage:
    python scripts/ablation.py --locus locus_01 [--config configs/pipeline.yaml]
                               [--features examples/demo_output/imputed_features.parquet]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _pipeline import REPO_ROOT, load_biobank, load_config, make_xgb_params, prepare_features

from perturbomeai import ablation
from perturbomeai.features import available_features


def _get_features(cfg, bank, features_path):
    if features_path:
        path = Path(features_path)
        df = pd.read_parquet(path) if path.suffix in {".parquet", ".pq"} else pd.read_csv(path)
        df["pid"] = df["pid"].astype(str)
        feature_order = available_features(df.columns)
        x = df[feature_order].to_numpy(dtype=np.float32)
        return x, df["pid"].tolist(), feature_order
    return prepare_features(bank["features"], cfg)


def main() -> int:
    parser = argparse.ArgumentParser(description="PerturbomeAI ablation stage.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--locus", type=str, default=None, help="Locus to ablate; default = target/first.")
    parser.add_argument("--features", type=str, default=None, help="Imputed feature table from stage 1.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg.get("output_dir", "examples/demo_output"))
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    bank = load_biobank(cfg)
    locus = args.locus or bank.get("target_locus") or bank["loci"][0]
    x, pids, feature_order = _get_features(cfg, bank, args.features)

    label_map = dict(zip(bank["labels"]["pid"].astype(str), bank["labels"][locus]))
    y = np.array([label_map.get(pid, np.nan) for pid in pids], dtype=float)
    keep = np.isfinite(y)

    table, top_features = ablation.run_ablation(
        x[keep],
        y[keep].astype(int),
        feature_order,
        top_k=cfg["ablation"]["top_k"],
        n_folds=cfg["scorer"]["n_folds"],
        seed=cfg.get("seed", 42),
        reference_fold=cfg["scorer"]["reference_fold"],
        params=make_xgb_params(cfg),
    )
    table.to_csv(out_dir / f"ablation_{locus}.csv", index=False)
    print(f"[ablation] {locus} top features={top_features}")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
