"""Synthetic biobank generator for the self-contained demo.

Real biobank genotype/phenotype tables cannot be redistributed, so the demo
builds a synthetic cohort with the same shape and the same genotype-to-phenotype
structure assumed by PerturbomeAI:

    1. Each locus has an allele frequency ``f`` and a natural-selection effect
       size ``beta = c * [f(1-f)]^alpha`` (rarer alleles have larger effects).
    2. Genotypes ``X_i ~ Bernoulli(f_i)`` are sampled per individual.
    3. A latent physiological state ``Z = sum_i beta_i v_i X_i + noise`` is built
       from per-locus latent directions ``v_i``.
    4. The routine clinical features are a noisy linear projection of ``Z`` onto
       the INPUT_FEATURES space, so each locus imprints a distributed,
       multi-feature signature (exactly what the model learns to decode).
    5. A fraction of feature cells are set missing (MCAR), to exercise the VAE.
    6. A synthetic proteome is coupled to one target locus's latent signal, so
       the score-driven proteomics analysis has real signal to recover.

The output column contract matches the real pipeline (``features.INPUT_FEATURES``,
per-locus binary labels, ``protein_*`` columns), so the same code path runs on
synthetic and real data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import INPUT_FEATURES

# Natural-selection prior constants (shared with the simulation chapter).
SELECTION_ALPHA = -0.3345784927
SELECTION_C = 0.2829933233


@dataclass
class BiobankData:
    features: pd.DataFrame      # pid + INPUT_FEATURES columns (with NaN missingness)
    labels: pd.DataFrame        # pid + one binary column per locus
    proteomics: pd.DataFrame    # pid, age, gender, protein_* columns
    loci_meta: pd.DataFrame     # locus, f, beta
    locus_names: list[str]
    target_locus: str
    feature_names: list[str]    # INPUT_FEATURES (model input columns)


def beta_from_f(f: np.ndarray) -> np.ndarray:
    return SELECTION_C * np.power(f * (1.0 - f), SELECTION_ALPHA)


def make_biobank(
    *,
    n_samples: int = 8000,
    n_loci: int = 6,
    n_proteins: int = 200,
    latent_dim: int = 8,
    missing_rate: float = 0.15,
    protein_target_locus: int = 0,
    f_min: float = 0.03,
    f_max: float = 0.40,
    sigma: float = 1.0,
    seed: int = 42,
) -> BiobankData:
    """Generate a synthetic biobank cohort (see module docstring)."""
    rng = np.random.default_rng(seed)
    pids = [f"S{idx:07d}" for idx in range(n_samples)]

    # Loci: log-spaced frequencies + natural-selection effect sizes.
    f = np.geomspace(f_min, f_max, n_loci)
    beta = beta_from_f(f)
    locus_names = [f"locus_{i + 1:02d}" for i in range(n_loci)]

    # Per-locus latent directions and genotypes.
    v = rng.normal(size=(n_loci, latent_dim))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    x_geno = (rng.random((n_samples, n_loci)) < f).astype(np.int8)

    # Latent physiological state Z = sum_i beta_i v_i X_i + noise.
    effects = (x_geno.astype(np.float64) * beta) @ v  # (N, latent_dim)
    z = effects + rng.normal(scale=sigma, size=(n_samples, latent_dim))

    # Project Z onto the clinical feature space (age handled separately).
    lab_cols = [c for c in INPUT_FEATURES if c != "age"]
    n_lab = len(lab_cols)
    w = rng.normal(size=(n_lab, latent_dim))
    feat = z @ w.T  # (N, n_lab)
    # Give each feature its own location/scale and per-cell noise.
    col_scale = rng.uniform(0.5, 2.0, size=n_lab)
    col_loc = rng.uniform(-1.0, 1.0, size=n_lab)
    feat = feat * col_scale + col_loc + rng.normal(scale=0.5, size=(n_samples, n_lab))

    features = pd.DataFrame(feat, columns=lab_cols)
    features.insert(0, "pid", pids)
    # Age: realistic range, mostly independent of genotype.
    features["age"] = rng.uniform(40.0, 70.0, size=n_samples)

    # Inject MCAR missingness into the lab feature cells only.
    if missing_rate > 0:
        mask = rng.random((n_samples, n_lab)) < missing_rate
        fmat = np.array(features[lab_cols].to_numpy(dtype=float), copy=True)
        fmat[mask] = np.nan
        features[lab_cols] = fmat

    # Reorder feature columns to the canonical contract.
    features = features[["pid"] + INPUT_FEATURES]

    # Labels: one binary carrier column per locus.
    labels = pd.DataFrame({"pid": pids})
    for i, name in enumerate(locus_names):
        labels[name] = x_geno[:, i].astype(int)

    # Proteomics coupled to the target locus's latent signal.
    target_idx = int(np.clip(protein_target_locus, 0, n_loci - 1))
    target_signal = z @ v[target_idx]  # (N,)
    target_signal = (target_signal - target_signal.mean()) / (target_signal.std() + 1e-9)
    loadings = np.zeros(n_proteins)
    n_signal = max(1, int(0.25 * n_proteins))
    signal_idx = rng.choice(n_proteins, size=n_signal, replace=False)
    loadings[signal_idx] = rng.normal(loc=0.0, scale=1.0, size=n_signal)
    age_centered = (features["age"].to_numpy() - 55.0) / 15.0
    age_coef = rng.normal(scale=0.2, size=n_proteins)
    prot = (
        np.outer(target_signal, loadings)
        + np.outer(age_centered, age_coef)
        + rng.normal(scale=1.0, size=(n_samples, n_proteins))
    )
    protein_cols = [f"protein_{j + 1:04d}" for j in range(n_proteins)]
    proteomics = pd.DataFrame(prot, columns=protein_cols)
    proteomics.insert(0, "pid", pids)
    proteomics["age"] = features["age"].to_numpy()
    proteomics["gender"] = rng.integers(0, 2, size=n_samples)

    loci_meta = pd.DataFrame({"locus": locus_names, "f": f, "beta": beta})

    return BiobankData(
        features=features,
        labels=labels,
        proteomics=proteomics,
        loci_meta=loci_meta,
        locus_names=locus_names,
        target_locus=locus_names[target_idx],
        feature_names=list(INPUT_FEATURES),
    )
