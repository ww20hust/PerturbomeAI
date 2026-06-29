# Usage guide

This guide covers the end-to-end demo, the per-stage CLIs, the config schema,
and how to run on real data. All commands are run from the repository root with
dependencies installed (`pip install -r requirements.txt`).

## 1. End-to-end demo

```bash
python scripts/run_demo.py [--config configs/pipeline.yaml]
```

Runs the whole pipeline on the configured data (synthetic by default) and writes
to `output_dir` (default `examples/demo_output/`). See the README for the list
of outputs.

## 2. Per-stage CLIs

The pipeline is also exposed as four independent, config-driven stages. They
share the same config and can run on synthetic or real data.

### Stage 1 - imputation

```bash
python scripts/impute.py [--config configs/pipeline.yaml]
```

Fits the per-column quantile transform, trains the masked beta-VAE, imputes
missing cells, and writes `imputed_features.parquet` (+ `.csv`) to the output
dir. This artifact is the dense feature matrix consumed by scoring.

### Stage 2 - scoring

```bash
python scripts/score.py [--config ...] [--locus locus_01] \
    [--features examples/demo_output/imputed_features.parquet]
```

Runs 8-fold (5/2/1) weighted XGBoost scoring, percentile-aligns the out-of-fold
scores, and writes `score_<locus>.csv` (pid, score, label, fold_id) plus
`locus_metrics.csv`. If `--features` is omitted, features are prepared on the
fly. If `--locus` is omitted, all configured loci are scored.

### Stage 3 - proteomics association

```bash
python scripts/proteomics.py --score examples/demo_output/score_locus_01.csv \
    [--config ...] [--proteomics /path/to/proteomics.parquet] [--name locus_01]
```

Tests proteins between the top and bottom score tails (OLS adjusting for age and
sex when present), applies BH-FDR, and writes `proteomics_<name>.csv` and a
volcano plot `proteomics_volcano_<name>.png`.

### Stage 4 - ablation

```bash
python scripts/ablation.py [--config ...] [--locus locus_01] \
    [--features examples/demo_output/imputed_features.parquet]
```

Ranks features by XGBoost gain, drops each subset of the Top-K features, re-runs
the 8-fold scoring, and writes `ablation_<locus>.csv` with the four metrics per
scenario (from `full` to `ablate_all`).

## 3. Configuration schema (`configs/pipeline.yaml`)

| Section | Key | Meaning |
| --- | --- | --- |
| `data` | `mode` | `synthetic` (generate demo data) or `files` (read from disk) |
| `data.synthetic` | `n_samples`, `n_loci`, `n_proteins`, `latent_dim`, `missing_rate`, `protein_target_locus` | synthetic cohort controls |
| `data.files` | `feature_table`, `label_table`, `proteomics_table`, `loci` | real-data paths + loci |
| `preprocess` | `age_col`, `age_divisor`, `qt_exclude` | age scaling + quantile-transform exclusions |
| `impute` | `enabled`, `latent_dim`, `hidden_dims`, `beta`, `kl_warmup_epochs`, `epochs`, `batch_size`, `lr`, `weight_decay`, `train_mask_min/max`, `device` | VAE imputation |
| `scorer` | `n_folds`, `reference_fold`, `device`, `xgb.*` | 8-fold scoring + XGBoost hyperparameters |
| `proteomics` | `score_q_low/high`, `fdr_threshold`, `min_group_size`, `label_top_n` | tail definition + FDR |
| `ablation` | `top_k` | number of top features to ablate (2^K scenarios) |
| `output_dir` | - | where artifacts are written |

`device` accepts `auto`, `cpu`, or `cuda`.

## 4. Running on real data

1. Build a feature table with column `pid` and the 62 `INPUT_FEATURES` columns
   (see `perturbomeai/features.py`). Missing values may be left as `NaN`.
2. Build a label table with `pid` and one binary column (0/1) per locus.
3. (Optional) Build a proteomics table with `pid`, `age`, optionally `gender`,
   and `protein_*` columns.
4. Set `data.mode: files` and the paths in `configs/pipeline.yaml`.
5. Run the demo or the per-stage CLIs exactly as above.

## 5. The simulation and HPC chapters

- Simulation: `cd simulation && python scripts/run_module.py --module A`
  (see `simulation/README.md`).
- HPC demo: `bash hpc/scripts/run_demo.sh` (see `hpc/README.md`).
