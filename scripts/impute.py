#!/usr/bin/env python3
"""Stage 1: quantile-transform features and impute missing cells with the VAE.

Writes ``imputed_features.parquet`` (and .csv) under the configured output dir:
the dense, imputed feature matrix consumed by the scoring stage.

Usage:
    python scripts/impute.py [--config configs/pipeline.yaml]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _pipeline import REPO_ROOT, load_biobank, load_config, prepare_features


def main() -> int:
    parser = argparse.ArgumentParser(description="PerturbomeAI imputation stage.")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg.get("output_dir", "examples/demo_output"))
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    bank = load_biobank(cfg)
    x, pids, feature_order = prepare_features(bank["features"], cfg)
    df = pd.DataFrame(x, columns=feature_order)
    df.insert(0, "pid", pids)
    df.to_parquet(out_dir / "imputed_features.parquet", index=False)
    df.to_csv(out_dir / "imputed_features.csv", index=False)
    print(f"[impute] wrote {out_dir / 'imputed_features.parquet'} shape={df.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
