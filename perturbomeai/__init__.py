"""PerturbomeAI: a phenotype-to-perturbation framework.

Decode locus-specific genetic perturbations directly from routine clinical
measurements and convert them into an individual-level Genetic Perturbation
Score (GPS).

Pipeline stages (see the module of the same name):
    features    - the routine clinical feature set used as model input.
    preprocess  - per-column quantile transform and age scaling.
    vae_impute  - masked beta-VAE that reconstructs missing input measurements.
    folds       - 8-fold assignment and the 5/2/1 train/validation/test rotation.
    scorer      - weighted XGBoost out-of-fold scoring (the GPS).
    align       - mid-rank percentile alignment of out-of-fold scores.
    metrics     - AUC, Cohen's d, Cliff's delta and the Mann-Whitney p-value.
    proteomics  - GPS-driven differential protein association.
    ablation    - feature-ablation of the score.
    simdata     - synthetic biobank generator for the self-contained demo.
"""

from . import (
    ablation,
    align,
    features,
    folds,
    metrics,
    preprocess,
    proteomics,
    scorer,
    simdata,
    vae_impute,
)

__all__ = [
    "ablation",
    "align",
    "features",
    "folds",
    "metrics",
    "preprocess",
    "proteomics",
    "scorer",
    "simdata",
    "vae_impute",
]

__version__ = "0.1.0"
