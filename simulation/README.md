# Simulation chapter: a controlled testbed for phenotype-to-perturbation decoding

This chapter validates the central premise of PerturbomeAI on a fully controlled
synthetic system, where the ground-truth genotype-to-phenotype map is known
exactly. It answers a simple question: **if a genetic perturbation leaves a
real (but noisy) imprint on a multivariate phenotype, can a classifier recover
carrier status by reversing that map, and how does recoverability depend on the
perturbation's allele frequency and effect size?**

## Forward generative model

Each locus `i` has an allele frequency `f_i` and an effect size `beta_i` drawn
from a natural negative-selection prior:

```
beta_i = c * [f_i (1 - f_i)]^alpha          (alpha < 0)
```

so that rarer alleles carry larger effects (anchored at `f = 0.5 => beta = 0.45`,
with an ~8x ratio between the rarest and most common locus). Genotypes are
sampled `X_i ~ Bernoulli(f_i)`; each locus pushes a 2D phenotype along its own
unit direction `v_i`:

```
Y = sum_i beta_i * v_i * X_i + epsilon ,    epsilon ~ N(0, sigma^2 I)
```

## Reverse inference (the model under test)

For each locus we train XGBoost to predict carrier status from the phenotype,
expanded to `[Y1, Y2, r, Y1*Y2]`. The cross-validation scheme is identical to
the main pipeline: **8-fold, 5 train / 2 validation (early stopping) / 1 test**,
rotated so every individual gets one out-of-fold score. Out-of-fold scores are
percentile-aligned to a reference fold, and we report **AUC, Cohen's d, Cliff's
delta and the two-sided Mann-Whitney p-value**. A closed-form linear-theory AUC
`Phi(d'/sqrt(2))` provides an upper reference.

## Modules

- **fig0** - the frequency vs effect-size prior.
- **Module A** - recover all 20 loci from one cohort; AUC/effect-size metrics per
  locus, phenotype-space decision boundaries, and AUC-vs-frequency.
- **Module B** - causal separation: hold `beta` fixed (vary `f`) and hold `f`
  fixed (vary `beta`) to disentangle the two drivers of recoverability.
- **Module C** - learning curves: stratified downsampling of the training
  carriers to show how recoverability scales with the number of training
  positives.

## Run

```bash
# from the repo root, with dependencies installed (see ../README.md)
cd simulation
python scripts/run_module.py --module A      # or B / C / fig0 / all
# convenience wrapper for everything:
bash scripts/run_all.sh
```

Outputs:

- `results/*.csv` - per-locus and per-fold metrics (`module_a_metrics.csv`, ...).
- `figures/*.png` and `*.svg` - publication figures.
- `results/cache/*.npz` - cached cohorts (delete to regenerate).

## Configuration

- `config/default.yaml` - cohort size, XGBoost hyperparameters, CV layout, paths.
- `config/variants.yaml` - the 20 loci (frequency + effect size) and prior
  constants.

> Note: Module A trains and scores 20 loci x 8 folds and renders phenotype-space
> boundaries; on a CPU this takes a few minutes for the default `N = 50,000`.
> Reduce `module_a.n` for a faster smoke run.
