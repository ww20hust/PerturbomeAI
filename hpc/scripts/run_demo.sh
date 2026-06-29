#!/usr/bin/env bash
# Tiny end-to-end HPC demo on synthetic loci (CPU device-pool by default).
#
# Steps: assign 8 folds once -> build shared feature memmap + label blocks ->
# run the 8-fold wave pool over the blocks -> assemble OOF -> align + merge +
# per-locus metrics -> print the genome-scale throughput projection.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPC_DIR="$(cd "${HERE}/.." && pwd)"
source "${HERE}/env.sh"

echo "=== [1/5] one-shot 8-fold cohort assignment ==="
python "${HPC_DIR}/00_assign_8fold.py"

echo "=== [2/5] build shared feature memmap + label blocks ==="
python "${HPC_DIR}/01_build_block_cache.py"

echo "=== [3/5] 8-fold wave pool (train every (block,fold)) ==="
bash "${HERE}/run_gpu_pool.sh"

echo "=== [4/5] assemble out-of-fold scores per block ==="
for b in $(seq 0 $((N_BLOCKS - 1))); do
  python "${HPC_DIR}/03_score_block_oof.py" --block "${b}"
done

echo "=== [5/5] align + merge + per-locus metrics + throughput projection ==="
python "${HPC_DIR}/04_align_merge.py"

echo "Done. Artifacts under ${PERTURBOMEAI_HPC_ROOT}"
