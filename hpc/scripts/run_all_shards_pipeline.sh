#!/usr/bin/env bash
# Overlapping shard pipeline.
#
# Loci are grouped into shards (analogous to chromosomes). While shard i is being
# scored + merged (background), shard i+1 is trained on the 8-GPU pool
# (foreground). This overlaps the post-processing of one shard with the training
# of the next, keeping the GPUs saturated.
#
# For the demo, each shard is WAVE_BLOCKS blocks; everything still runs on the
# configured DEVICES (CPU pool by default).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HPC_DIR="$(cd "${HERE}/.." && pwd)"
source "${HERE}/env.sh"

# One-shot cohort + block cache (shared by all shards).
python "${HPC_DIR}/00_assign_8fold.py"
python "${HPC_DIR}/01_build_block_cache.py"

# Partition blocks into shards of WAVE_BLOCKS blocks each.
shards=()
cur=""
count=0
for b in $(seq 0 $((N_BLOCKS - 1))); do
  if [[ -z "${cur}" ]]; then cur="${b}"; else cur="${cur},${b}"; fi
  count=$((count + 1))
  if [[ "${count}" -ge "${WAVE_BLOCKS}" ]]; then
    shards+=("${cur}"); cur=""; count=0
  fi
done
if [[ -n "${cur}" ]]; then shards+=("${cur}"); fi

post_pid=""
score_and_merge() {  # background post-processing for one shard
  local shard_blocks="$1"
  for b in ${shard_blocks//,/ }; do
    python "${HPC_DIR}/03_score_block_oof.py" --block "${b}"
  done
}

for shard in "${shards[@]}"; do
  echo "[pipeline] training shard blocks=${shard}"
  bash "${HERE}/run_gpu_pool.sh" "${shard}"
  # Wait for the previous shard's post-processing before starting this one's.
  if [[ -n "${post_pid}" ]]; then wait "${post_pid}"; fi
  score_and_merge "${shard}" &
  post_pid="$!"
done
if [[ -n "${post_pid}" ]]; then wait "${post_pid}"; fi

# Final align + merge + throughput projection across all blocks.
python "${HPC_DIR}/04_align_merge.py"
echo "[pipeline] done. Results under ${PERTURBOMEAI_HPC_ROOT}"
