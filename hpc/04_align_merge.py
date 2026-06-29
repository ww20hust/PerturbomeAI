#!/usr/bin/env python3
"""Stage 4: per-fold align, merge blocks, per-locus metrics, throughput projection.

For every locus in every block, percentile-aligns the OOF scores per fold (reusing
``perturbomeai.align``), computes the four discrimination metrics (reusing
``perturbomeai.metrics``), and merges all blocks into one per-locus metrics table.
Finally, it projects the genome-scale wall time from the measured per-job timings.

Usage:
    python 04_align_merge.py [--n-loci-target 300000] [--n-gpus 8]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HPC_DIR = Path(__file__).resolve().parent
REPO_ROOT = HPC_DIR.parent
sys.path.insert(0, str(HPC_DIR))
sys.path.insert(0, str(REPO_ROOT))

from hpc_lib import block_cache, cohort, paths, throughput  # noqa: E402
from perturbomeai.align import align_oof_scores  # noqa: E402
from perturbomeai.metrics import discrimination_metrics  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Align, merge and project.")
    parser.add_argument("--n-loci-target", type=int, default=int(os.environ.get("N_LOCI_TARGET", 300000)))
    parser.add_argument("--n-gpus", type=int, default=int(os.environ.get("N_GPUS_TARGET", 8)))
    parser.add_argument("--n-folds", type=int, default=int(os.environ.get("N_FOLDS", 8)))
    parser.add_argument("--reference-fold", type=int, default=0)
    args = parser.parse_args()

    manifest = block_cache.load_manifest()
    fold_ids = cohort.load_fold_ids()

    rows: list[dict] = []
    for entry in manifest["blocks"]:
        b = entry["block_id"]
        oof = np.load(paths.block_oof_path(b))
        labels = block_cache.open_block_labels(b)
        locus_ids = block_cache.load_block_locus_ids(b)
        for j, locus in enumerate(locus_ids):
            y = np.asarray(labels[:, j], dtype=int)
            raw = oof[:, j]
            aligned, _ = align_oof_scores(raw, fold_ids, reference_fold=args.reference_fold)
            finite = np.isfinite(aligned)
            m = discrimination_metrics(aligned[finite], y[finite])
            rows.append({"block_id": b, "locus": locus, **m})

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(paths.merged_metrics_path(), index=False)
    aucs = metrics_df["auc"].dropna()
    print(
        f"[merge] {len(metrics_df)} loci merged; "
        f"AUC median={aucs.median():.3f} range=[{aucs.min():.3f}, {aucs.max():.3f}]"
    )

    timings = throughput.collect_timings()
    projection = throughput.project_runtime(
        timings, n_loci_target=args.n_loci_target, n_gpus=args.n_gpus, n_folds=args.n_folds
    )
    with open(paths.merged_dir() / "throughput_projection.json", "w", encoding="utf-8") as f:
        json.dump(projection, f, indent=2)
    print("[throughput] projection:")
    print(json.dumps(projection, indent=2))
    print(
        f"[throughput] {args.n_loci_target:,} loci on {args.n_gpus} GPUs: "
        f"~{projection['projected_parallel_days']:.2f} days "
        f"(naive serial ~{projection['naive_serial_days']:.1f} days; "
        f"{projection['speedup_vs_serial']:.1f}x)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
