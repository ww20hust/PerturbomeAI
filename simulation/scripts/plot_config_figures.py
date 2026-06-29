#!/usr/bin/env python3
"""Render the frequency-effect-size prior figure (fig0) from the config alone."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_loader import load_config
from src.plots import plot_fig0_freq_effect_prior


def main() -> None:
    cfg = load_config()
    out = plot_fig0_freq_effect_prior(cfg)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
