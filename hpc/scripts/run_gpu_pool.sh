#!/usr/bin/env bash
# Run one wave through the 8-fold -> 8-GPU wave pool.
#
# A wave = the given blocks x N_FOLDS folds, dispatched round-robin across
# DEVICES (8 folds of one block exactly fill 8 GPUs). Falls back to a CPU worker
# pool when GPU_IDS is empty.
#
# Usage:
#   bash run_gpu_pool.sh "0,1"        # explicit block ids
#   bash run_gpu_pool.sh              # defaults to all blocks 0..N_BLOCKS-1
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPC_DIR="$(cd "${HERE}/.." && pwd)"
source "${HERE}/env.sh"

BLOCKS="${1:-}"
if [[ -z "${BLOCKS}" ]]; then
  BLOCKS="$(seq -s, 0 $((N_BLOCKS - 1)))"
fi

echo "[run_gpu_pool] blocks=${BLOCKS} folds=${N_FOLDS} devices=${DEVICES}"
python "${HPC_DIR}/hpc_lib/gpu_pool.py" \
  --blocks "${BLOCKS}" \
  --n-folds "${N_FOLDS}" \
  --devices "${DEVICES}" \
  --train-script "${HPC_DIR}/02_train_block_fold.py"
