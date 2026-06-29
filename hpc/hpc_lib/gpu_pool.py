"""8-fold to 8-GPU wave-pool dispatcher.

A *wave* is ``WAVE_BLOCKS`` blocks x ``N_FOLDS`` folds of training jobs. Each job
is one ``(block, fold)`` pair; the 8 folds of a single block exactly fill 8 GPUs.
Jobs are dispatched round-robin across the available devices, keeping at most
``len(devices)`` jobs running at once. Each job is pinned to its device via
``CUDA_VISIBLE_DEVICES`` (GPU) or simply run on a CPU worker when no GPU is set.

This is intentionally a thin, dependency-free scheduler (subprocess + polling)
so it runs identically on a laptop (CPU device-pool) and on an 8-GPU node.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HPC_DIR = Path(__file__).resolve().parents[1]


def parse_devices(spec: str) -> list[str]:
    """Parse a device spec like "0,1,2,3" (GPUs) or "cpu:4" (4 CPU workers)."""
    spec = spec.strip()
    if not spec or spec.startswith("cpu"):
        n = 1
        if ":" in spec:
            n = max(1, int(spec.split(":", 1)[1]))
        return ["cpu"] * n
    return [tok.strip() for tok in spec.split(",") if tok.strip() != ""]


def build_jobs(blocks: list[int], n_folds: int) -> list[tuple[int, int]]:
    """All (block, fold) pairs for a wave, ordered fold-major within each block."""
    return [(b, f) for b in blocks for f in range(n_folds)]


def _job_env(device: str) -> dict:
    env = os.environ.copy()
    if device == "cpu":
        env["PERTURBOMEAI_DEVICE"] = "cpu"
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["PERTURBOMEAI_DEVICE"] = "cuda"
        env["CUDA_VISIBLE_DEVICES"] = str(device)
    return env


def dispatch(
    blocks: list[int],
    n_folds: int,
    devices: list[str],
    train_script: Path,
) -> int:
    """Run all (block, fold) jobs across devices with bounded concurrency."""
    jobs = build_jobs(blocks, n_folds)
    n_workers = max(1, len(devices))
    running: list[tuple[subprocess.Popen, tuple[int, int], str]] = []
    pending = list(jobs)
    failures = 0
    print(f"[gpu_pool] {len(jobs)} jobs over {n_workers} device(s): {devices}")

    def launch(job: tuple[int, int], device: str) -> None:
        block, fold = job
        cmd = [sys.executable, str(train_script), "--block", str(block), "--fold", str(fold)]
        proc = subprocess.Popen(cmd, env=_job_env(device))
        running.append((proc, job, device))
        print(f"[gpu_pool] launch block={block} fold={fold} -> device={device}")

    # Prime up to n_workers, tracking which device each worker owns.
    free_devices = list(devices)
    while pending and free_devices:
        launch(pending.pop(0), free_devices.pop(0))

    while running or pending:
        time.sleep(0.05)
        still: list[tuple[subprocess.Popen, tuple[int, int], str]] = []
        for proc, job, device in running:
            ret = proc.poll()
            if ret is None:
                still.append((proc, job, device))
                continue
            if ret != 0:
                failures += 1
                print(f"[gpu_pool] FAILED block={job[0]} fold={job[1]} (exit {ret})")
            if pending:
                launch(pending.pop(0), device)
        running[:] = still
    if failures:
        print(f"[gpu_pool] completed with {failures} failure(s)")
    else:
        print("[gpu_pool] all jobs completed")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="8-fold to 8-GPU wave-pool dispatcher.")
    parser.add_argument("--blocks", type=str, required=True, help="Comma-separated block ids.")
    parser.add_argument("--n-folds", type=int, default=8)
    parser.add_argument("--devices", type=str, default="cpu:4", help='"0,1,..." or "cpu:N".')
    parser.add_argument(
        "--train-script",
        type=str,
        default=str(HPC_DIR / "02_train_block_fold.py"),
    )
    args = parser.parse_args()
    blocks = [int(b) for b in args.blocks.split(",") if b.strip() != ""]
    devices = parse_devices(args.devices)
    failures = dispatch(blocks, args.n_folds, devices, Path(args.train_script))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
