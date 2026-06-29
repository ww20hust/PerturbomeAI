"""Measure per-job wall time and project genome-scale runtime.

The projection is derived from the run itself: we time the demo ``(block, fold)``
jobs, compute a per-locus-per-fold cost, and extrapolate to a target number of
loci across the available GPUs. A naive serial estimate is reported alongside so
the speed-up from the wave pool is easy to read off.
"""

from __future__ import annotations

import json

from . import paths


def record_timing(block_id: int, fold: int, n_loci: int, seconds: float) -> None:
    payload = {
        "block_id": int(block_id),
        "fold": int(fold),
        "n_loci": int(n_loci),
        "seconds": float(seconds),
        "seconds_per_locus": float(seconds) / max(1, int(n_loci)),
    }
    with open(paths.timing_path(block_id, fold), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def collect_timings() -> list[dict]:
    out = []
    for p in sorted(paths.timing_dir().glob("block*_fold*.json")):
        with open(p, encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def project_runtime(
    timings: list[dict],
    *,
    n_loci_target: int,
    n_gpus: int,
    n_folds: int = 8,
) -> dict:
    """Project genome-scale wall time from measured per-(block,fold) timings."""
    if not timings:
        raise ValueError("No timings collected; run the training jobs first.")
    per_locus_fold = sum(t["seconds_per_locus"] for t in timings) / len(timings)

    # Total fold-model fits = n_loci_target * n_folds; each costs ~per_locus_fold.
    total_fit_seconds = per_locus_fold * n_loci_target * n_folds
    # The wave pool runs n_gpus fits concurrently.
    parallel_seconds = total_fit_seconds / max(1, n_gpus)
    naive_serial_seconds = total_fit_seconds  # one fit at a time

    return {
        "measured_jobs": len(timings),
        "seconds_per_locus_per_fold": per_locus_fold,
        "n_loci_target": int(n_loci_target),
        "n_folds": int(n_folds),
        "n_gpus": int(n_gpus),
        "projected_parallel_hours": parallel_seconds / 3600.0,
        "projected_parallel_days": parallel_seconds / 86400.0,
        "naive_serial_hours": naive_serial_seconds / 3600.0,
        "naive_serial_days": naive_serial_seconds / 86400.0,
        "speedup_vs_serial": naive_serial_seconds / max(1e-9, parallel_seconds),
    }
