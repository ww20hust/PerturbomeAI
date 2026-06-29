#!/usr/bin/env bash
# Environment for the genome-scale HPC chapter.
#
# Override any of these before sourcing, e.g.:  GPU_IDS="0,1,2,3,4,5,6,7" bash run_demo.sh

# Where all artifacts (cohort, feature memmap, blocks, merged metrics) are written.
export PERTURBOMEAI_HPC_ROOT="${PERTURBOMEAI_HPC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/hpc_result}"

# Cross-validation: 8 folds map one-to-one onto 8 GPUs.
export N_FOLDS="${N_FOLDS:-8}"
export SEED="${SEED:-42}"

# Devices for the wave pool:
#   - GPU node:  GPU_IDS="0,1,2,3,4,5,6,7"  -> DEVICES="0,1,...,7"
#   - laptop:    GPU_IDS=""                 -> DEVICES="cpu:N" (N CPU workers)
export GPU_IDS="${GPU_IDS:-}"
if [[ -n "${GPU_IDS}" ]]; then
  export DEVICES="${GPU_IDS}"
else
  export DEVICES="${DEVICES:-cpu:4}"
fi

# Demo data scale (kept small so it runs on a CPU in seconds-to-minutes).
export N_SAMPLES="${N_SAMPLES:-4000}"
export N_LOCI="${N_LOCI:-40}"
export BLOCK_SIZE="${BLOCK_SIZE:-20}"

# How many blocks are pushed through the pool per wave.
export WAVE_BLOCKS="${WAVE_BLOCKS:-2}"

# Genome-scale projection targets (used by 04_align_merge.py).
export N_LOCI_TARGET="${N_LOCI_TARGET:-300000}"
export N_GPUS_TARGET="${N_GPUS_TARGET:-8}"

# Derived: number of blocks = ceil(N_LOCI / BLOCK_SIZE).
export N_BLOCKS="$(( (N_LOCI + BLOCK_SIZE - 1) / BLOCK_SIZE ))"
