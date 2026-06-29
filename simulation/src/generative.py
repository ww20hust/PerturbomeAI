"""Forward generative model: genotypes, directions, phenotypes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .config_loader import cache_path, variant_arrays


@dataclass
class Cohort:
    Y: np.ndarray          # (N, 2)
    X: np.ndarray          # (N, m) int8
    v: np.ndarray          # (m, 2)
    epsilon: np.ndarray    # (N, 2)
    f: np.ndarray          # (m,)
    beta: np.ndarray       # (m,)
    ids: np.ndarray        # (m,)
    sigma: float

    @property
    def n(self) -> int:
        return self.Y.shape[0]

    @property
    def m(self) -> int:
        return self.X.shape[1]


def beta_from_f(f: np.ndarray, c: float, alpha: float) -> np.ndarray:
    return c * np.power(f * (1.0 - f), alpha)


def sample_directions(rng: np.random.Generator, m: int) -> np.ndarray:
    u = rng.normal(size=(m, 2))
    norms = np.linalg.norm(u, axis=1, keepdims=True)
    return u / norms


def sample_genotypes(rng: np.random.Generator, n: int, f: np.ndarray) -> np.ndarray:
    return (rng.random((n, f.size)) < f).astype(np.int8)


def compute_phenotypes(
    X: np.ndarray,
    v: np.ndarray,
    beta: np.ndarray,
    epsilon: np.ndarray,
) -> np.ndarray:
    # Y_j = sum_i beta_i * v_i * X_ij + eps_j
    effects = (X.astype(np.float64) * beta) @ v  # (N, 2)
    return effects + epsilon


def generate_cohort(
    n: int,
    f: np.ndarray,
    beta: np.ndarray,
    ids: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
    v: Optional[np.ndarray] = None,
    epsilon: Optional[np.ndarray] = None,
) -> Cohort:
    m = f.size
    if v is None:
        v = sample_directions(rng, m)
    if epsilon is None:
        epsilon = rng.normal(scale=sigma, size=(n, 2))
    X = sample_genotypes(rng, n, f)
    Y = compute_phenotypes(X, v, beta, epsilon)
    return Cohort(Y=Y, X=X, v=v, epsilon=epsilon, f=f, beta=beta, ids=ids, sigma=sigma)


def derive_b2_cohort(b1: Cohort, beta_med: float) -> Cohort:
    beta = np.full(b1.m, beta_med, dtype=np.float64)
    Y = compute_phenotypes(b1.X, b1.v, beta, b1.epsilon)
    return Cohort(
        Y=Y, X=b1.X, v=b1.v, epsilon=b1.epsilon,
        f=b1.f.copy(), beta=beta, ids=b1.ids.copy(), sigma=b1.sigma,
    )


def generate_b3_cohort(
    b1: Cohort,
    f_med: float,
    rng: np.random.Generator,
) -> Cohort:
    f = np.full(b1.m, f_med, dtype=np.float64)
    X = sample_genotypes(rng, b1.n, f)
    Y = compute_phenotypes(X, b1.v, b1.beta, b1.epsilon)
    return Cohort(
        Y=Y, X=X, v=b1.v.copy(), epsilon=b1.epsilon,
        f=f, beta=b1.beta.copy(), ids=b1.ids.copy(), sigma=b1.sigma,
    )


def save_cohort(path: Path, cohort: Cohort, extra: Optional[dict] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "Y": cohort.Y,
        "X": cohort.X,
        "v": cohort.v,
        "epsilon": cohort.epsilon,
        "f": cohort.f,
        "beta": cohort.beta,
        "ids": cohort.ids,
        "sigma": cohort.sigma,
    }
    if extra:
        payload.update(extra)
    np.savez_compressed(path, **payload)


def load_cohort(path: Path) -> tuple[Cohort, dict]:
    data = np.load(path, allow_pickle=True)
    cohort = Cohort(
        Y=data["Y"],
        X=data["X"],
        v=data["v"],
        epsilon=data["epsilon"],
        f=data["f"],
        beta=data["beta"],
        ids=data["ids"],
        sigma=float(data["sigma"]),
    )
    extra = {k: data[k] for k in data.files if k not in {
        "Y", "X", "v", "epsilon", "f", "beta", "ids", "sigma"
    }}
    return cohort, extra


def _assert_g1_min_pos(cohort: Cohort, cfg: dict) -> None:
    min_pos = cfg["cv"]["min_pos_total"]
    n_pos_g1 = int(cohort.X[:, 0].sum())
    if n_pos_g1 < min_pos:
        raise ValueError(
            f"G1 n_pos={n_pos_g1} < {min_pos}; delete cache and regenerate or increase module_a.n"
        )


def get_or_create_b1_50k(cfg: dict, rng: np.random.Generator) -> Cohort:
    path = cache_path(cfg, cfg["module_a"]["cache"])
    if path.exists():
        cohort, _ = load_cohort(path)
        _assert_g1_min_pos(cohort, cfg)
        return cohort
    va = variant_arrays(cfg)
    cohort = generate_cohort(
        n=cfg["module_a"]["n"],
        f=va["f"],
        beta=va["beta"],
        ids=va["ids"],
        sigma=cfg["sigma"],
        rng=rng,
    )
    _assert_g1_min_pos(cohort, cfg)
    save_cohort(path, cohort)
    return cohort


def get_or_create_b3_50k(cfg: dict, b1: Cohort, rng: np.random.Generator) -> Cohort:
    path = cache_path(cfg, cfg["module_b"]["cache_b3"])
    if path.exists():
        cohort, _ = load_cohort(path)
        return cohort
    cohort = generate_b3_cohort(b1, cfg["f_med"], rng)
    save_cohort(path, cohort)
    return cohort
