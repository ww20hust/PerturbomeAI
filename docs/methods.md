# Methods

This document describes the PerturbomeAI methodology in the form of a paper
Methods section. It mirrors the implementation in `perturbomeai/`.

## Inputs and preprocessing

Each individual is represented by 62 routine clinical measurements (blood and
urine chemistry, complete blood count, body size, and age), listed in
`perturbomeai/features.py`. Each non-age feature is mapped to a common uniform
space with a per-column quantile transform fit on observed values only; missing
entries remain missing so that imputation handles them explicitly. Age is scaled
by a fixed divisor.

## Missing-feature imputation (masked beta-VAE)

Clinical panels are incomplete, so missing inputs are imputed with a masked
variational autoencoder before scoring. The encoder consumes an interleaved
`(mask, value)` representation in which a naturally missing cell and a
deliberately hidden observed cell are indistinguishable, both `(0, 0)`. During
training a per-row random fraction of observed cells is hidden, and the decoder
is asked to reconstruct them; the loss is the reconstruction error on the hidden
cells plus a KL term annealed to a target weight `beta` over a warmup. At
inference the encoder reads all observed cells, the posterior mean is decoded,
and the decoder output replaces only the missing cells, producing the complete
feature matrix that the scoring stage consumes.

## Genetic Perturbation Score (weighted XGBoost, 8-fold 5/2/1)

For a locus, carriers (label 1) and non-carriers (label 0) are separated with
gradient-boosted trees (XGBoost) on the imputed feature matrix. Class imbalance
is handled with a weighted loss, `scale_pos_weight = n_neg / n_pos` computed from
the training folds, applied also as per-sample weights on training and
validation.

Cross-validation uses an 8-fold scheme with a fixed role assignment per
rotation: 5 folds for training, 2 folds for validation (for early stopping), and
1 fold for test. The test fold rotates through all 8 positions, so every
individual receives exactly one out-of-fold (OOF) score. The choice of 8 folds
is also what enables the genome-scale schedule in `hpc/`, where the 8 fold-models
of a locus map onto 8 GPUs.

## Score alignment

Because each test fold is scored by a different model, the per-fold score
distributions can differ. We align them by mapping each fold's scores onto a
reference fold's empirical distribution via mid-rank percentiles: a score at
within-fold rank percentile `u` is assigned the value at percentile `u` of the
reference fold. This preserves within-fold ranking while equalising marginals,
giving a comparable pooled score.

## Discrimination metrics

Separation between carriers and non-carriers is quantified with four
complementary measures: the area under the ROC curve (AUC); Cohen's d
(standardised mean difference); Cliff's delta (a non-parametric effect size,
`2 * VDA - 1` with `VDA = U / (n1 n2)`); and the two-sided Mann-Whitney U
p-value. Per-fold test AUC is also reported.

## Score-driven proteomic association

The aligned score behaves like a synthetic phenotype summarising the perturbed
pathway. We compare individuals in the top vs the bottom tail of the score
(default top 20% vs bottom 20%) and test each protein with an ordinary least
squares model `protein ~ group + age [+ sex]`. We report the group coefficient
and its p-value, the Benjamini-Hochberg adjusted p-value across proteins, and
Cohen's d between tails, summarised in a volcano plot.

## Feature ablation

To attribute the score to specific features, we rank features by XGBoost gain,
take the Top-K, and re-run the full 8-fold scoring for every subset of those K
features dropped (`2^K` scenarios). Reporting the four metrics per scenario
isolates the contribution of individual features and their combinations.

## Simulation validation

A controlled forward generative model with a natural negative-selection prior
(`beta = c [f(1-f)]^alpha`) provides ground truth for how recoverability depends
on allele frequency and effect size, using the identical 8-fold 5/2/1 scheme and
metrics. See `simulation/README.md`.

## Genome-scale computation

The per-locus scheme is parallelised across GPUs by mapping the 8 fold-models of
a locus onto 8 GPUs, reading a shared feature memmap once, and overlapping
training with post-processing across shards. A measured throughput projection
extrapolates the per-locus cost to genome scale. See `hpc/README.md`.
