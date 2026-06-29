#!/usr/bin/env python3
"""Run simulation module A, B, C, fig0, or all."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments import run_all, run_fig0, run_module_a, run_module_b, run_module_c


def main() -> None:
    parser = argparse.ArgumentParser(description="Phenotype-to-variant simulation")
    parser.add_argument(
        "--module",
        choices=["fig0", "A", "B", "C", "all"],
        default="all",
        help="Which module to run",
    )
    args = parser.parse_args()

    if args.module == "fig0":
        run_fig0()
    elif args.module == "A":
        run_module_a()
    elif args.module == "B":
        run_module_b()
    elif args.module == "C":
        run_module_c()
    else:
        summary = run_all()
        print(summary)


if __name__ == "__main__":
    main()
