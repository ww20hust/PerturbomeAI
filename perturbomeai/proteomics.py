"""Genetic Perturbation Score driven differential protein association.

Because the score captures the integrated physiological state of the pathway
perturbed by a locus, it behaves like a synthetic phenotype. We test which
plasma proteins differ between individuals in the top vs the bottom tail of the
score (default top 20% vs bottom 20%), adjusting for age (and sex when present)
with an ordinary least squares model:

    protein ~ group + age [+ sex]

We report the ``group`` coefficient (beta) and its two-sided p-value, the
Benjamini-Hochberg adjusted p-value across all tested proteins, and Cohen's d
between the tails. A volcano plot summarises the result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def _ols_group_beta_p(y: np.ndarray, design: np.ndarray, group_col: int) -> tuple[float, float]:
    """OLS via least squares; return (beta, two-sided p) for one design column."""
    n, k = design.shape
    if n <= k:
        return float("nan"), float("nan")
    beta, _res, rank, _sv = np.linalg.lstsq(design, y, rcond=None)
    if rank < k:
        return float("nan"), float("nan")
    resid = y - design @ beta
    dof = n - k
    sigma2 = float(resid @ resid) / dof
    if sigma2 <= 0:
        return float(beta[group_col]), float("nan")
    xtx_inv = np.linalg.inv(design.T @ design)
    se = np.sqrt(sigma2 * xtx_inv[group_col, group_col])
    if se <= 0:
        return float(beta[group_col]), float("nan")
    t_stat = beta[group_col] / se
    pval = 2.0 * stats.t.sf(abs(t_stat), dof)
    return float(beta[group_col]), float(pval)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1, s2 = np.std(a, ddof=1), np.std(b, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 * s1 + (n2 - 1) * s2 * s2) / (n1 + n2 - 2))
    if pooled == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled)


def differential_proteins(
    score_df: pd.DataFrame,
    proteomics_df: pd.DataFrame,
    *,
    score_col: str = "score",
    pid_col: str = "pid",
    protein_prefix: str = "protein",
    covariate_cols: tuple[str, ...] = ("age", "gender"),
    score_q_low: float = 0.20,
    score_q_high: float = 0.80,
    min_group_size: int = 10,
    missing_rate_threshold: float = 0.5,
) -> pd.DataFrame:
    """Differential protein analysis between top and bottom score tails.

    Returns one row per protein with beta, p_value, p_adj_bh, cohens_d, mean
    differences and group sizes. The ``group`` covariate is 1 for the top tail
    and 0 for the bottom tail.
    """
    protein_cols = [c for c in proteomics_df.columns if str(c).startswith(protein_prefix)]
    if not protein_cols:
        raise ValueError(f"No protein columns with prefix {protein_prefix!r} found.")

    df = proteomics_df.copy()
    df[pid_col] = df[pid_col].astype(str)
    sc = score_df[[pid_col, score_col]].copy()
    sc[pid_col] = sc[pid_col].astype(str)
    merged = df.merge(sc, on=pid_col, how="inner")

    # Drop individuals with too many missing proteins.
    pmat = merged[protein_cols].to_numpy(dtype=float)
    miss_rate = np.isnan(pmat).sum(axis=1) / len(protein_cols)
    merged = merged.loc[miss_rate < missing_rate_threshold].reset_index(drop=True)

    # z-score each protein (vectorised to avoid DataFrame fragmentation).
    prot = merged[protein_cols].apply(pd.to_numeric, errors="coerce")
    std = prot.std(axis=0).replace(0.0, np.nan)
    prot_z = (prot - prot.mean(axis=0)) / std
    merged = pd.concat([merged.drop(columns=protein_cols), prot_z.fillna(0.0)], axis=1)

    q_lo = merged[score_col].quantile(score_q_low)
    q_hi = merged[score_col].quantile(score_q_high)
    group = np.where(
        merged[score_col] >= q_hi, 1, np.where(merged[score_col] <= q_lo, 0, -1)
    )
    merged = merged.assign(_group=group)
    ana = merged.loc[merged["_group"] >= 0].reset_index(drop=True)
    n_top = int((ana["_group"] == 1).sum())
    n_bottom = int((ana["_group"] == 0).sum())
    if n_top < min_group_size or n_bottom < min_group_size:
        raise ValueError(f"Tail groups too small (top={n_top}, bottom={n_bottom}).")

    use_cov = [c for c in covariate_cols if c in ana.columns]
    rows: list[dict] = []
    for col in protein_cols:
        sub_cols = [col, "_group"] + use_cov
        sub = ana[sub_cols].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) < (3 + len(use_cov)):
            continue
        y = sub[col].to_numpy(dtype=float)
        design_cols = [np.ones(len(sub)), sub["_group"].to_numpy(dtype=float)]
        for c in use_cov:
            design_cols.append(sub[c].to_numpy(dtype=float))
        design = np.column_stack(design_cols)
        beta, pval = _ols_group_beta_p(y, design, group_col=1)
        top = sub.loc[sub["_group"] == 1, col].to_numpy(dtype=float)
        bot = sub.loc[sub["_group"] == 0, col].to_numpy(dtype=float)
        rows.append(
            {
                "protein": col,
                "beta": beta,
                "p_value": pval,
                "cohens_d": _cohens_d(top, bot),
                "mean_top": float(np.mean(top)) if len(top) else float("nan"),
                "mean_bottom": float(np.mean(bot)) if len(bot) else float("nan"),
                "mean_diff": float(np.mean(top) - np.mean(bot)) if (len(top) and len(bot)) else float("nan"),
                "n_top": int(len(top)),
                "n_bottom": int(len(bot)),
            }
        )

    results = pd.DataFrame(rows)
    if results.empty:
        return results
    valid = results["p_value"].notna()
    results.loc[valid, "p_adj_bh"] = _benjamini_hochberg(results.loc[valid, "p_value"].to_numpy())
    return results.sort_values("p_adj_bh", na_position="last").reset_index(drop=True)


def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def volcano_plot(
    results: pd.DataFrame,
    out_path: str | Path,
    *,
    fdr_threshold: float = 0.05,
    label_top_n: int = 20,
    title: str | None = None,
) -> Path:
    """Save a volcano plot: x = beta, y = -log10(BH-adjusted p)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = results.dropna(subset=["beta", "p_adj_bh"]).copy()
    df["neg_log10_p_adj"] = -np.log10(np.clip(df["p_adj_bh"], 1e-300, None))

    sig_up = (df["p_adj_bh"] < fdr_threshold) & (df["beta"] > 0)
    sig_down = (df["p_adj_bh"] < fdr_threshold) & (df["beta"] < 0)
    sig_none = ~(sig_up | sig_down)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.scatter(df.loc[sig_none, "beta"], df.loc[sig_none, "neg_log10_p_adj"], c="gray", alpha=0.35, s=12)
    ax.scatter(df.loc[sig_up, "beta"], df.loc[sig_up, "neg_log10_p_adj"], c="red", alpha=0.5, s=18)
    ax.scatter(df.loc[sig_down, "beta"], df.loc[sig_down, "neg_log10_p_adj"], c="blue", alpha=0.5, s=18)
    ax.axhline(-np.log10(fdr_threshold), ls="--", lw=0.6, alpha=0.6)
    ax.axvline(0.0, ls="--", lw=0.6, alpha=0.6)
    ax.set_xlabel("Beta (top vs bottom score tail)")
    ax.set_ylabel("-log10(BH-adjusted p-value)")
    if title:
        ax.set_title(title)

    labelled = df.loc[df["p_adj_bh"] < fdr_threshold].nsmallest(label_top_n, "p_adj_bh")
    for _, r in labelled.iterrows():
        short = str(r["protein"]).replace("protein_", "")
        ax.annotate(short, (r["beta"], r["neg_log10_p_adj"]), fontsize=6, alpha=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
