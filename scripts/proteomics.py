#!/usr/bin/env python3
"""Stage 3: Genetic Perturbation Score driven differential protein association.

Reads a per-individual score table (``pid``, ``score`` from stage 2) and a
proteomics table, then tests proteins between the top and bottom score tails,
writing a results CSV and a volcano plot.

Usage:
    python scripts/proteomics.py --score examples/demo_output/score_locus_01.csv
                                 [--config configs/pipeline.yaml]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _pipeline import REPO_ROOT, load_biobank, load_config

from perturbomeai import proteomics


def main() -> int:
    parser = argparse.ArgumentParser(description="PerturbomeAI proteomics stage.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--score", type=str, required=True, help="Score CSV with pid, score.")
    parser.add_argument("--proteomics", type=str, default=None, help="Proteomics table (overrides config).")
    parser.add_argument("--name", type=str, default="locus", help="Label for output filenames.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg.get("output_dir", "examples/demo_output"))
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    score_df = pd.read_csv(args.score)
    score_df["pid"] = score_df["pid"].astype(str)

    if args.proteomics:
        path = Path(args.proteomics)
        prot_df = pd.read_parquet(path) if path.suffix in {".parquet", ".pq"} else pd.read_csv(path)
        prot_df["pid"] = prot_df["pid"].astype(str)
    else:
        bank = load_biobank(cfg)
        prot_df = bank["proteomics"]
        if prot_df is None:
            raise SystemExit("No proteomics table available (set data.files.proteomics_table or --proteomics).")

    p = cfg["proteomics"]
    results = proteomics.differential_proteins(
        score_df,
        prot_df,
        score_q_low=p["score_q_low"],
        score_q_high=p["score_q_high"],
        min_group_size=p["min_group_size"],
    )
    results.to_csv(out_dir / f"proteomics_{args.name}.csv", index=False)
    proteomics.volcano_plot(
        results,
        out_dir / f"proteomics_volcano_{args.name}.png",
        fdr_threshold=p["fdr_threshold"],
        label_top_n=p["label_top_n"],
        title=f"{args.name} score-driven proteome",
    )
    n_sig = int((results["p_adj_bh"] < p["fdr_threshold"]).sum())
    print(f"[proteomics] {len(results)} proteins tested, {n_sig} FDR<{p['fdr_threshold']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
