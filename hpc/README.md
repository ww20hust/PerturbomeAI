# Genome-scale HPC chapter: an efficient per-locus modeling scheme

Developing and cross-scoring **hundreds of thousands of loci** is a huge compute
task. Done one locus at a time - or with naive parallelism - it takes months.
This chapter contributes a genome-scale scheme that completes it in **~2 days on
8 GPUs**, and ships a self-contained demo that runs the entire machinery on a
CPU with synthetic loci.

## The core idea: 8 folds map onto 8 GPUs

Every locus needs **8 cross-validation models** (the 5/2/1 rotation). We
deliberately choose 8-fold so that one locus's 8 fold-models map **one-to-one
onto 8 GPUs** and train simultaneously under a single, tidy schedule. Thousands
of loci are pushed through the 8 GPUs per round; the regular structure avoids
idle GPUs and avoids re-reading data.

```
              one block of loci
                     |
       +-------------+-------------+
       |   8 fold-models per locus |
       v                           v
   GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
   fold0 fold1 fold2 fold3 fold4 fold5 fold6 fold7
       \___________ one wave ___________/
```

## Why it is fast (the four design decisions)

1. **One-shot 8-fold cohort assignment** (`00_assign_8fold.py`, `hpc_lib/cohort.py`).
   Each individual gets a single `cv_test_fold in 0..7` once; the same split is
   reused for every locus. Fold membership is precomputed, not recomputed per
   locus.
2. **No-repeated-read block cache** (`01_build_block_cache.py`, `hpc_lib/block_cache.py`).
   The shared imputed feature matrix `X` is written **once** as a float32 memmap
   and opened **read-only** by every job. Loci are partitioned into blocks
   (default 2000/block) with a per-block label memmap and a manifest. The heavy
   feature matrix is never re-parsed or copied per locus.
3. **The smallest job is one `(block, fold)`** (`02_train_block_fold.py`).
   It loads the shared memmap + the block labels, masks train/val/test via the
   5/2/1 map, and trains the weighted XGBoost (`scale_pos_weight` +
   `sample_weight`, early stopping on the 2 validation folds, `device=cuda` when
   available) for **every locus in the block** on that one fold, then scores the
   held-out test fold.
4. **8-fold to 8-GPU wave pool** (`scripts/run_gpu_pool.sh`, `hpc_lib/gpu_pool.py`).
   A wave = `WAVE_BLOCKS` blocks x 8 folds of jobs, dispatched round-robin across
   the GPUs via `CUDA_VISIBLE_DEVICES`. The **overlapping shard pipeline**
   (`scripts/run_all_shards_pipeline.sh`) scores+merges shard *i* in the
   background while shard *i+1* trains in the foreground, keeping the GPUs
   saturated.

Out-of-fold scores are then percentile-aligned per fold and merged
(`03_score_block_oof.py`, `04_align_merge.py`), reusing `perturbomeai.align` and
`perturbomeai.metrics`, producing one aligned score and the four discrimination
metrics (AUC, Cohen's d, Cliff's delta, Mann-Whitney p) per locus.

## Throughput projection

`hpc_lib/throughput.py` times the demo `(block, fold)` jobs, derives a per-locus
cost, and extrapolates to a target locus count across the GPUs - printing the
projected genome-scale wall time next to the naive-serial estimate. Because the
projection is built from your own run, the genome-scale runtime reflects your
hardware rather than a fixed figure.

## Run the demo (CPU, synthetic, no private data)

```bash
# from the repo root, dependencies installed (see ../README.md)
bash hpc/scripts/run_demo.sh
```

This assigns 8 folds, builds the shared memmap + label blocks for synthetic
loci, runs the 8-fold wave pool on a CPU device-pool, assembles OOF, merges
per-locus metrics, and prints the throughput projection. Artifacts land under
`hpc/hpc_result/` (override with `PERTURBOMEAI_HPC_ROOT`).

## Run at scale (8-GPU node)

```bash
GPU_IDS="0,1,2,3,4,5,6,7" \
N_SAMPLES=500000 N_LOCI=200000 BLOCK_SIZE=2000 WAVE_BLOCKS=4 \
bash hpc/scripts/run_all_shards_pipeline.sh
```

Point the cache at real WES/genotype data by replacing the synthetic generator
in `01_build_block_cache.py` with your feature/label loader (the column contract
is `perturbomeai.features.INPUT_FEATURES` + per-locus binary labels) and setting
`PERTURBOMEAI_HPC_ROOT` to your scratch space.

## File map

| Path | Role |
| --- | --- |
| `00_assign_8fold.py` | one-shot 8-fold cohort assignment |
| `01_build_block_cache.py` | shared feature memmap + per-block label memmaps |
| `02_train_block_fold.py` | train all loci in one `(block, fold)` (the wave-pool unit) |
| `03_score_block_oof.py` | assemble OOF scores per block + coverage check |
| `04_align_merge.py` | per-fold align, merge, per-locus metrics, projection |
| `hpc_lib/` | `paths`, `cohort`, `block_cache`, `gpu_pool`, `throughput` |
| `scripts/env.sh` | devices, scale, paths (env-driven) |
| `scripts/run_gpu_pool.sh` | one wave through the 8-GPU pool |
| `scripts/run_all_shards_pipeline.sh` | overlapping shard pipeline |
| `scripts/run_demo.sh` | tiny end-to-end CPU demo |
