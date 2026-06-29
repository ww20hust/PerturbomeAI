"""Shared helpers for the PerturbomeAI command-line scripts.

Centralises config loading, data loading (synthetic or from files), feature
preparation (quantile transform + VAE imputation) and per-locus scoring, so the
stage scripts stay thin and consistent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perturbomeai import align, scorer, vae_impute  # noqa: E402
from perturbomeai.features import available_features  # noqa: E402
from perturbomeai.metrics import discrimination_metrics  # noqa: E402
from perturbomeai.preprocess import fit_column_quantile_transform  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs" / "pipeline.yaml"


def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_compute_device(name: str) -> str:
    """Map an 'auto'|'cpu'|'cuda' config value to 'cpu' or 'cuda'."""
    if name == "cpu":
        return "cpu"
    if name == "cuda":
        return "cuda"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_biobank(cfg: dict) -> dict:
    """Return a dict with features/labels/proteomics frames, loci, and metadata."""
    mode = cfg["data"]["mode"]
    if mode == "synthetic":
        from perturbomeai import simdata

        s = cfg["data"]["synthetic"]
        data = simdata.make_biobank(
            n_samples=s["n_samples"],
            n_loci=s["n_loci"],
            n_proteins=s["n_proteins"],
            latent_dim=s["latent_dim"],
            missing_rate=s["missing_rate"],
            protein_target_locus=s["protein_target_locus"],
            seed=cfg.get("seed", 42),
        )
        return {
            "features": data.features,
            "labels": data.labels,
            "proteomics": data.proteomics,
            "loci": list(data.locus_names),
            "target_locus": data.target_locus,
        }
    if mode == "files":
        files = cfg["data"]["files"]
        features = _read_table(files["feature_table"])
        labels = _read_table(files["label_table"])
        proteomics = _read_table(files["proteomics_table"]) if files.get("proteomics_table") else None
        loci = list(files.get("loci") or [c for c in labels.columns if c != "pid"])
        target = loci[0] if loci else None
        for df in (features, labels):
            df["pid"] = df["pid"].astype(str)
        if proteomics is not None:
            proteomics["pid"] = proteomics["pid"].astype(str)
        return {
            "features": features,
            "labels": labels,
            "proteomics": proteomics,
            "loci": loci,
            "target_locus": target,
        }
    raise ValueError(f"Unknown data.mode: {mode!r}")


def make_xgb_params(cfg: dict) -> "scorer.XGBParams":
    x = cfg["scorer"]["xgb"]
    return scorer.XGBParams(
        n_estimators=x["n_estimators"],
        max_depth=x["max_depth"],
        learning_rate=x["learning_rate"],
        subsample=x["subsample"],
        colsample_bytree=x["colsample_bytree"],
        min_child_weight=x["min_child_weight"],
        gamma=x["gamma"],
        reg_alpha=x["reg_alpha"],
        reg_lambda=x["reg_lambda"],
        early_stopping_rounds=x["early_stopping_rounds"],
        device=resolve_compute_device(cfg["scorer"].get("device", "auto")),
    )


def prepare_features(features_df: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, list[str], list[str]]:
    """Quantile-transform features and (optionally) impute missing cells.

    Returns (X imputed float32 matrix, pids, feature_order).
    """
    df = features_df.copy()
    df["pid"] = df["pid"].astype(str)
    feature_order = available_features(df.columns)
    qt = fit_column_quantile_transform(
        df,
        feature_order,
        age_col=cfg["preprocess"]["age_col"],
        age_divisor=cfg["preprocess"]["age_divisor"],
        random_state=cfg.get("seed", 42),
    )
    qt_df = qt.transform(df)
    x = qt_df[feature_order].to_numpy(dtype=np.float32)

    imp = cfg["impute"]
    if imp.get("enabled", True):
        vae_cfg = vae_impute.VAEConfig(
            latent_dim=imp["latent_dim"],
            hidden_dims=tuple(imp["hidden_dims"]),
            beta=imp["beta"],
            kl_warmup_epochs=imp["kl_warmup_epochs"],
            epochs=imp["epochs"],
            batch_size=imp["batch_size"],
            lr=imp["lr"],
            weight_decay=imp["weight_decay"],
            train_mask_min=imp["train_mask_min"],
            train_mask_max=imp["train_mask_max"],
            seed=cfg.get("seed", 42),
            device=imp.get("device", "auto"),
        )
        x_filled, _ = vae_impute.train_and_impute(x, vae_cfg)
    else:
        # Fallback: column-median imputation.
        x_filled = x.copy()
        col_median = np.nanmedian(x_filled, axis=0)
        idx = np.where(~np.isfinite(x_filled))
        x_filled[idx] = np.take(col_median, idx[1])
    return np.asarray(x_filled, dtype=np.float32), df["pid"].tolist(), feature_order


def score_locus(
    x: np.ndarray,
    labels_df: pd.DataFrame,
    pids: list[str],
    locus: str,
    cfg: dict,
) -> dict:
    """Run 8-fold OOF scoring + alignment + metrics for one locus.

    Returns a dict with the aligned score frame, fold metrics, and pooled metrics.
    """
    label_map = dict(zip(labels_df["pid"].astype(str), labels_df[locus]))
    y = np.array([label_map.get(pid, np.nan) for pid in pids], dtype=float)
    keep = np.isfinite(y)
    x_k = x[keep]
    y_k = y[keep].astype(int)
    pids_k = [p for p, k in zip(pids, keep) if k]

    params = make_xgb_params(cfg)
    res = scorer.oof_score(
        x_k, y_k, n_folds=cfg["scorer"]["n_folds"], seed=cfg.get("seed", 42), params=params
    )
    aligned, status = align.align_oof_scores(
        res.oof_raw, res.fold_ids, reference_fold=cfg["scorer"]["reference_fold"]
    )
    score_df = pd.DataFrame({"pid": pids_k, "score": aligned, "label": y_k, "fold_id": res.fold_ids})
    pooled = discrimination_metrics(aligned, y_k)
    pooled["locus"] = locus
    return {
        "locus": locus,
        "score_df": score_df,
        "fold_metrics": res.fold_metrics,
        "pooled": pooled,
        "align_status": status,
    }
