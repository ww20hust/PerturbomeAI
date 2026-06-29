"""Load simulation configuration from YAML files."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(name: str) -> dict[str, Any]:
    path = ROOT / "config" / name
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config() -> dict[str, Any]:
    cfg = load_yaml("default.yaml")
    var_cfg = load_yaml("variants.yaml")
    cfg["variants"] = var_cfg["variants"]
    cfg["alpha"] = var_cfg["alpha"]
    cfg["c"] = var_cfg["c"]
    cfg["f_med"] = var_cfg["f_med"]
    cfg["beta_med"] = var_cfg["beta_med"]
    return cfg


def variant_arrays(cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    variants = cfg["variants"]
    return {
        "ids": np.array([v["id"] for v in variants], dtype=object),
        "f": np.array([v["f"] for v in variants], dtype=np.float64),
        "beta": np.array([v["beta"] for v in variants], dtype=np.float64),
    }


def resolve_path(cfg: dict[str, Any], *parts: str) -> Path:
    return ROOT.joinpath(*parts)


def cache_path(cfg: dict[str, Any], filename: str) -> Path:
    return resolve_path(cfg, cfg["paths"]["cache"], filename)


def figures_path(cfg: dict[str, Any], filename: str) -> Path:
    return resolve_path(cfg, cfg["paths"]["figures"], filename)


def results_path(cfg: dict[str, Any], filename: str) -> Path:
    return resolve_path(cfg, cfg["paths"]["results"], filename)
