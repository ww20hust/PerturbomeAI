"""Genome-scale HPC chapter: 8-fold to 8-GPU wave-pool scheduling.

Building blocks:
    paths        - config/env-driven result-root and artifact path helpers.
    cohort       - one-shot 8-fold cohort assignment + 5/2/1 train-val map.
    block_cache  - shared feature memmap (load once) + per-block label memmaps.
    gpu_pool     - round-robin device dispatcher (8 folds of a block fill 8 GPUs).
    throughput   - measure per-job wall time and project genome-scale runtime.
"""
