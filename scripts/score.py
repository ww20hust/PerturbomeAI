#!/usr/bin/env python3
"""Stage 2: 8-fold weighted XGBoost scoring + percentile alignment + metrics.

For each locus, produces an aligned out-of-fold Genetic Perturbation Score per
individual and the four discrimination metrics. Consumes the imputed feature
matrix from stage 1 when available, otherwise prepares features on the fly.

Usage:
    python scripts/score.py [--config configs/pipeline.yaml] [--locus locus_01]
                            [--features examples/demo_output/imputed_features.parquet]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _pipeline import REPO_ROOT, load_biobank, load_config, prepare_features, score_locus

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
    parser = argparse.ArgumentParser(description="PerturbomeAI scoring stage.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--locus", type=str, default=None, help="Single locus; default = all.")
    parser.add_argument("--features", type=str, default=None, help="Imputed feature table from stage 1.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg.get("output_dir", "examples/demo_output"))
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    bank = load_biobank(cfg)
    x, pids, _feature_order = _get_features(cfg, bank, args.features)
    loci = [args.locus] if args.locus else bank["loci"]

    pooled_rows = []
    for locus in loci:
        result = score_locus(x, bank["labels"], pids, locus, cfg)
        result["score_df"].to_csv(out_dir / f"score_{locus}.csv", index=False)
        pooled_rows.append(result["pooled"])
        m = result["pooled"]
        print(
            f"[score] {locus}: AUC={m['auc']:.3f} cohen_d={m['cohens_d']:.3f} "
            f"cliffs_delta={m['cliffs_delta']:.3f} p={m['mannwhitney_p']:.2e}"
        )
    pd.DataFrame(pooled_rows).to_csv(out_dir / "locus_metrics.csv", index=False)
    print(f"[score] wrote metrics for {len(loci)} loci to {out_dir / 'locus_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
