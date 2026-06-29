"""Figure generation for simulation outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

from .config_loader import figures_path  # noqa: E402
from .generative import Cohort, beta_from_f  # noqa: E402
from .inference import (  # noqa: E402
    mean_prevalence_boundary_field,
    median_boundary_probs,
    reference_train_neg_count,
    reference_train_pos_count,
    theoretical_n_neg_train,
    theoretical_n_pos_train,
)
from .theory import theory_for_all_variants  # noqa: E402

LANCET_BLUE = "#00468B"
LANCET_RED = "#ED0000"
LANCET_GREEN = "#42B540"


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _save_figure(fig, out: Path, **kwargs) -> Path:
    """Save figure as PNG and SVG alongside each other."""
    _ensure_dir(out)
    fig.savefig(out, **kwargs)
    svg_kwargs = {k: v for k, v in kwargs.items() if k not in ("dpi", "format")}
    fig.savefig(out.with_suffix(".svg"), format="svg", **svg_kwargs)
    plt.close(fig)
    return out


def _apply_lancet_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })


def _style_lancet_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#E6E6E6", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def _shared_auc_ylim(df: pd.DataFrame, margin: float = 0.03) -> tuple[float, float]:
    vals = df["auc"].dropna()
    if vals.empty:
        return 0.45, 1.02
    lo, hi = float(vals.min()), float(vals.max())
    pad = max(margin, 0.05 * (hi - lo + 1e-6))
    return max(0.4, lo - pad), min(1.02, hi + pad)


def _boxplot_strip(
    ax,
    x_vals: list[float],
    y_groups: list[np.ndarray],
    *,
    color: str,
    width: float = 0.06,
) -> None:
    """Boxplot at each x with overlaid jittered points (Lancet-style)."""
    positions = np.asarray(x_vals, dtype=float)
    bp = ax.boxplot(
        y_groups,
        positions=positions,
        widths=width,
        patch_artist=True,
        showfliers=False,
        zorder=2,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.25)
        patch.set_edgecolor(color)
        patch.set_linewidth(0.9)
    for med in bp["medians"]:
        med.set_color(color)
        med.set_linewidth(1.2)
    for whisker in bp["whiskers"]:
        whisker.set_color(color)
        whisker.set_linewidth(0.8)
    for cap in bp["caps"]:
        cap.set_color(color)
        cap.set_linewidth(0.8)

    rng = np.random.default_rng(0)
    for x0, ys in zip(positions, y_groups):
        ys = np.asarray(ys, dtype=float)
        ys = ys[np.isfinite(ys)]
        if len(ys) == 0:
            continue
        jitter = rng.uniform(-width * 0.22, width * 0.22, size=len(ys))
        ax.scatter(
            x0 + jitter, ys,
            s=18, c=color, edgecolors="white", linewidths=0.4,
            alpha=0.9, zorder=3,
        )


def _boxplot_by_x(
    ax,
    df: pd.DataFrame,
    x_col: str,
    y_col: str = "auc",
    *,
    color: str,
    width: float = 0.06,
) -> None:
    sub = df.dropna(subset=[x_col, y_col])
    x_vals = sorted(sub[x_col].unique())
    y_groups = [sub.loc[sub[x_col] == x, y_col].values for x in x_vals]
    _boxplot_strip(ax, x_vals, y_groups, color=color, width=width)


def _boxplot_by_variant(
    ax,
    fold_df: pd.DataFrame,
    x_col: str,
    y_col: str = "auc",
    *,
    color: str,
    width: float = 0.06,
) -> pd.DataFrame:
    """One box per variant_id at canonical x (configured f or beta)."""
    sub = fold_df.dropna(subset=[x_col, y_col, "variant_id"])
    meta = (
        sub.groupby("variant_id", as_index=False)
        .agg({x_col: "first"})
        .sort_values(x_col)
    )
    x_vals: list[float] = []
    y_groups: list[np.ndarray] = []
    for vid in meta["variant_id"]:
        xs = meta.loc[meta["variant_id"] == vid, x_col].iloc[0]
        ys = sub.loc[sub["variant_id"] == vid, y_col].values
        x_vals.append(float(xs))
        y_groups.append(ys)
    _boxplot_strip(ax, x_vals, y_groups, color=color, width=width)
    return meta


def _set_sparse_xticks(ax, x_positions: np.ndarray, fmt: str, max_ticks: int = 6) -> None:
    """Label a subset of x positions to avoid overlap."""
    x_positions = np.asarray(x_positions, dtype=float)
    n = len(x_positions)
    if n == 0:
        return
    step = max(1, (n - 1) // (max_ticks - 1)) if n > 1 else 1
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    ticks = x_positions[idx]
    ax.set_xticks(ticks)
    ax.set_xticklabels([format(v, fmt) for v in ticks])


def _plot_panel_metrics(ax, m: pd.Series, *, fontsize: float = 7.5) -> None:
    """Metrics anchored just outside the right edge of the parent axes."""
    text = (
        f"AUC: {m['auc']:.3f}\n"
        f"Cohen's d: {m['cohens_d']:.3f}\n"
        f"Cliff's delta: {m['cliffs_delta']:.3f}\n"
        f"Score median diff\n"
        f"(pos - neg): {m['median_diff']:.3f}"
    )
    ax.text(
        1.02, 0.98, text,
        transform=ax.transAxes,
        fontsize=fontsize, va="top", ha="left",
        linespacing=1.35, clip_on=False,
    )


def _make_fig1_grid(n_panels: int, n_cols: int = 4):
    """Rows by columns of loci; metrics sit on the right edge of each axes."""
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 2.85 * n_rows))
    if n_panels == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)
    fig.subplots_adjust(wspace=0.42, hspace=0.42)
    return fig, axes, n_rows, n_cols


def _fig1_panel_ax(axes, panel: int, n_cols: int):
    row, col = divmod(panel, n_cols)
    return axes[row, col]


def plot_fig0_freq_effect_prior(cfg: dict, out: Optional[Path] = None) -> Path:
    if out is None:
        out = figures_path(cfg, "fig0_freq_effect_prior.png")
    _ensure_dir(out)

    variants = cfg["variants"]
    f_pts = np.array([v["f"] for v in variants])
    beta_pts = np.array([v["beta"] for v in variants])
    ids = [v["id"] for v in variants]

    f_curve = np.logspace(np.log10(0.0005), np.log10(0.5), 280)
    beta_curve = beta_from_f(f_curve, cfg["c"], cfg["alpha"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.log10(f_curve), beta_curve, "k-", lw=2,
            label=rf"$\beta=c[f(1-f)]^{{{cfg['alpha']:.3f}}}$")
    ax.scatter(np.log10(f_pts), beta_pts, c="C0", s=50, zorder=3)

    for label in ("G1", "G20"):
        if label in ids:
            i = ids.index(label)
            ax.annotate(label, (np.log10(f_pts[i]), beta_pts[i]), xytext=(5, 5),
                        textcoords="offset points", fontsize=9)

    ax.axhline(
        cfg["beta_med"], color="C1", ls="--", lw=1,
        label=f"Median effect size (beta) = {cfg['beta_med']:.3f}",
    )
    ax.axvline(
        np.log10(cfg["f_med"]), color="C2", ls="--", lw=1,
        label=f"Median allele frequency (f) = {cfg['f_med']}",
    )

    ax.set_xlabel(r"$\log_{10}(f)$")
    ax.set_ylabel(r"Effect size $\beta$")
    ax.set_title("Natural negative selection: frequency-effect size relationship")
    ax.legend(loc="upper right", fontsize=8)
    txt = rf"$c={cfg['c']:.4f}$, anchor: $f=0.5 \Rightarrow \beta=0.45$, G1/G20 $\approx 8\times$"
    ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=8, va="bottom")
    fig.tight_layout()
    _save_figure(fig, out, dpi=150)
    return out


def _make_boundary_grid(Y: np.ndarray, grid_res: int = 120):
    x_min, x_max = Y[:, 0].min() - 0.5, Y[:, 0].max() + 0.5
    y_min, y_max = Y[:, 1].min() - 0.5, Y[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_res),
        np.linspace(y_min, y_max, grid_res),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid


def _plot_decision_boundary(ax, xx, yy, prob: np.ndarray):
    prob = prob.reshape(xx.shape)
    ax.contour(xx, yy, prob, levels=[0.5], colors="black", linewidths=1.2)
    ax.contourf(xx, yy, prob, levels=20, alpha=0.12, cmap="RdBu_r")


def _plot_prevalence_boundary(ax, xx, yy, binary_field: np.ndarray):
    """Top-f binary field from KNN OOF grid scores; contour at 0.5."""
    field = binary_field.reshape(xx.shape)
    ax.contourf(
        xx, yy, field,
        levels=np.linspace(0, 1, 11),
        cmap="RdBu_r",
        alpha=0.15,
    )
    ax.contour(xx, yy, field, levels=[0.5], colors="black", linewidths=1.2)


def _plot_effect_arrow(ax, Y, v_i: np.ndarray, beta_i: float):
    center = np.mean(Y, axis=0)
    span = max(np.ptp(Y[:, 0]), np.ptp(Y[:, 1]), 0.5)
    direction = v_i / (np.linalg.norm(v_i) + 1e-12)
    arrow_len = 0.28 * span * min(beta_i / 3.6, 1.5)
    start = center - 0.5 * arrow_len * direction
    end = center + 0.5 * arrow_len * direction
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=2.2,
        color=LANCET_BLUE,
        zorder=4,
    )
    ax.add_patch(arrow)
    ax.text(
        0.02, 0.98,
        rf"$\mathbf{{v}}=({v_i[0]:.2f},{v_i[1]:.2f})$  $\beta={beta_i:.2f}$",
        transform=ax.transAxes, fontsize=6, va="top", color=LANCET_BLUE,
    )


def plot_fig1_phenotype_space(
    cfg: dict,
    cohort: Cohort,
    metrics_df: pd.DataFrame,
    out: Optional[Path] = None,
) -> Path:
    if out is None:
        out = figures_path(cfg, "fig1_phenotype_space.png")
    _ensure_dir(out)

    order = np.argsort(cohort.f)
    n_cols = 4
    n_panels = len(order)
    fig, axes, _, _ = _make_fig1_grid(n_panels, n_cols=n_cols)
    metrics_by_id = metrics_df.set_index("variant_id")

    for panel, vi in enumerate(order):
        ax = _fig1_panel_ax(axes, panel, n_cols)
        vid = str(cohort.ids[vi])
        y = cohort.X[:, vi]
        ax.scatter(
            cohort.Y[y == 0, 0], cohort.Y[y == 0, 1],
            s=2, c="lightgray", alpha=0.25, rasterized=True,
        )
        ax.scatter(
            cohort.Y[y == 1, 0], cohort.Y[y == 1, 1],
            s=8, c="C3", alpha=0.8, rasterized=True,
        )
        xx, yy, grid = _make_boundary_grid(cohort.Y)
        prob = median_boundary_probs(cohort.Y, y, grid, cfg, cfg["seed"] + vi)
        _plot_decision_boundary(ax, xx, yy, prob)
        _plot_effect_arrow(ax, cohort.Y, cohort.v[vi], float(cohort.beta[vi]))
        ax.set_title(f"{vid}  f={cohort.f[vi]:.4g}", fontsize=9)

        if vid in metrics_by_id.index:
            _plot_panel_metrics(ax, metrics_by_id.loc[vid], fontsize=8.5)
        ax.set_xlabel(r"$Y_1$", fontsize=7)
        ax.set_ylabel(r"$Y_2$", fontsize=7)
        ax.tick_params(labelsize=6)

    for panel in range(n_panels, axes.size):
        _fig1_panel_ax(axes, panel, n_cols).axis("off")

    fig.suptitle(
        "Phenotype space with XGBoost decision boundaries (8-fold median probability = 0.5); "
        "metrics from 8-fold aligned out-of-fold scores",
        fontsize=10, y=1.01,
    )
    _save_figure(fig, out, dpi=150, bbox_inches="tight")
    return out


def plot_fig1_phenotype_space_prevalence(
    cfg: dict,
    cohort: Cohort,
    metrics_df: pd.DataFrame,
    out: Optional[Path] = None,
) -> Path:
    """fig1 variant: tiered top-f boundary on KNN (k=1000) median aligned OOF scores."""
    if out is None:
        out = figures_path(cfg, "fig1_phenotype_space_prevalence.png")
    _ensure_dir(out)

    order = np.argsort(cohort.f)
    n_cols = 4
    n_panels = len(order)
    fig, axes, _, _ = _make_fig1_grid(n_panels, n_cols=n_cols)
    metrics_by_id = metrics_df.set_index("variant_id")

    for panel, vi in enumerate(order):
        ax = _fig1_panel_ax(axes, panel, n_cols)
        vid = str(cohort.ids[vi])
        y = cohort.X[:, vi]
        f_i = float(cohort.f[vi])
        ax.scatter(
            cohort.Y[y == 0, 0], cohort.Y[y == 0, 1],
            s=2, c="lightgray", alpha=0.25, rasterized=True,
        )
        ax.scatter(
            cohort.Y[y == 1, 0], cohort.Y[y == 1, 1],
            s=8, c="C3", alpha=0.8, rasterized=True,
        )
        xx, yy, grid = _make_boundary_grid(cohort.Y)
        mean_field = mean_prevalence_boundary_field(
            cohort.Y, y, grid, f_i, cfg, cfg["seed"] + vi,
        )
        _plot_prevalence_boundary(ax, xx, yy, mean_field)
        _plot_effect_arrow(ax, cohort.Y, cohort.v[vi], float(cohort.beta[vi]))
        ax.set_title(f"{vid}  f={f_i:.4g}", fontsize=9)

        if vid in metrics_by_id.index:
            _plot_panel_metrics(ax, metrics_by_id.loc[vid], fontsize=8.5)
        ax.set_xlabel(r"$Y_1$", fontsize=7)
        ax.set_ylabel(r"$Y_2$", fontsize=7)
        ax.tick_params(labelsize=6)

    for panel in range(n_panels, axes.size):
        _fig1_panel_ax(axes, panel, n_cols).axis("off")

    fig.suptitle(
        "Phenotype space with tiered top-f boundaries (KNN k=1000 median aligned OOF); "
        "metrics from 8-fold aligned out-of-fold scores",
        fontsize=10, y=1.01,
    )
    _save_figure(fig, out, dpi=150, bbox_inches="tight")
    return out


def plot_fig2_auc_vs_freq(
    cfg: dict,
    cohort: Cohort,
    metrics_df: pd.DataFrame,
    fold_metrics_df: pd.DataFrame,
    out: Optional[Path] = None,
) -> Path:
    if out is None:
        out = figures_path(cfg, "fig2_auc_vs_freq.png")
    _ensure_dir(out)
    _apply_lancet_style()

    theory = theory_for_all_variants(cohort)
    f_all = np.sort(cohort.f)
    theory_sorted = theory[np.argsort(cohort.f)]

    fold_df = fold_metrics_df.copy()
    fold_df["log10_f"] = np.log10(fold_df.groupby("variant_id")["f"].transform("first"))

    fig, ax = plt.subplots(figsize=(8, 5))
    _style_lancet_axes(ax)
    meta = _boxplot_by_variant(ax, fold_df, "log10_f", color=LANCET_BLUE, width=0.08)
    ax.plot(np.log10(f_all), theory_sorted, "k--", lw=1.5, label=r"Theory $\Phi(d'/\sqrt{2})$")
    _set_sparse_xticks(ax, meta["log10_f"].values, fmt=".1f")
    ax.set_xlabel(r"$\log_{10}(f)$")
    ax.set_ylabel("Test AUC-ROC")
    ax.set_title("Discriminability versus allele frequency (8-fold cross-validation)")
    ax.legend(loc="lower left", frameon=False)
    ylim = _shared_auc_ylim(fold_df)
    ax.set_ylim(ylim)
    fig.tight_layout()
    _save_figure(fig, out, dpi=150)
    return out


def plot_fig3_causal_separation(
    cfg: dict,
    fold_df_b2: pd.DataFrame,
    fold_df_b3: pd.DataFrame,
    out: Optional[Path] = None,
) -> Path:
    if out is None:
        out = figures_path(cfg, "fig3_causal_separation.png")
    _ensure_dir(out)
    _apply_lancet_style()

    fold_df_b2 = fold_df_b2.copy()
    fold_df_b3 = fold_df_b3.copy()

    def _attach_log10_f(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["log10_f"] = np.log10(out.groupby("variant_id")["f"].transform("first"))
        return out

    def _attach_beta_x(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["x_beta"] = out.groupby("variant_id")["beta"].transform("first")
        return out

    fold_df_b2 = _attach_log10_f(fold_df_b2)
    fold_df_b3 = _attach_beta_x(fold_df_b3)

    ylim = _shared_auc_ylim(
        pd.concat([fold_df_b2, fold_df_b3], ignore_index=True)
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    _style_lancet_axes(axes[0])
    meta_b2 = _boxplot_by_variant(axes[0], fold_df_b2, "log10_f", color=LANCET_RED, width=0.08)
    _set_sparse_xticks(axes[0], meta_b2["log10_f"].values, fmt=".1f")
    axes[0].set_title(rf"Fixed $\beta={cfg['beta_med']:.3f}$")
    axes[0].set_xlabel(r"$\log_{10}(f)$")
    axes[0].set_ylabel("Test AUC-ROC")
    axes[0].set_ylim(ylim)

    _style_lancet_axes(axes[1])
    meta_b3 = _boxplot_by_variant(axes[1], fold_df_b3, "x_beta", color=LANCET_GREEN, width=0.08)
    _set_sparse_xticks(axes[1], meta_b3["x_beta"].values, fmt=".2f")
    axes[1].set_title(rf"Fixed $f={cfg['f_med']}$")
    axes[1].set_xlabel(r"$\beta$")
    axes[1].set_ylim(ylim)

    fig.suptitle("Causal separation: disentangling frequency and effect size (8-fold cross-validation)", y=1.02)
    fig.tight_layout()
    _save_figure(fig, out, dpi=150, bbox_inches="tight")
    return out


def plot_fig4_learning_curve(cfg: dict, df: pd.DataFrame, out: Optional[Path] = None) -> Path:
    if out is None:
        out = figures_path(cfg, "fig4_learning_curve.png")
    _ensure_dir(out)
    _apply_lancet_style()

    variants = cfg["module_c"]["variants"]
    n_cohort = cfg["module_a"]["n"]
    plot_df = df.copy()
    ref_pos_by_vid = {
        vid: reference_train_pos_count(n_cohort, float(f))
        for vid, f in plot_df.groupby("variant_id")["f"].first().items()
    }
    ref_neg_by_vid = {
        vid: reference_train_neg_count(n_cohort, float(f))
        for vid, f in plot_df.groupby("variant_id")["f"].first().items()
    }
    if "n_pos_train_theory" not in plot_df.columns:
        plot_df["n_pos_train_theory"] = plot_df.apply(
            lambda row: theoretical_n_pos_train(row["downsample_ratio"], ref_pos_by_vid[row["variant_id"]]),
            axis=1,
        )
    if "n_neg_train_theory" not in plot_df.columns:
        plot_df["n_neg_train_theory"] = plot_df.apply(
            lambda row: theoretical_n_neg_train(row["downsample_ratio"], ref_neg_by_vid[row["variant_id"]]),
            axis=1,
        )
    ylim = _shared_auc_ylim(plot_df)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = {"G7": LANCET_BLUE, "G17": LANCET_RED}

    for ax, vid in zip(axes, variants):
        sub = plot_df[plot_df["variant_id"] == vid].dropna(subset=["auc"])
        if sub.empty:
            continue
        _style_lancet_axes(ax)
        levels = (
            sub.groupby(["n_pos_train_theory", "n_neg_train_theory"], as_index=False)
            .agg({"downsample_ratio": "first"})
            .sort_values("n_pos_train_theory")
        )
        x_vals = list(range(len(levels)))
        y_groups = [
            sub.loc[
                (sub["n_pos_train_theory"] == int(row.n_pos_train_theory))
                & (sub["n_neg_train_theory"] == int(row.n_neg_train_theory)),
                "auc",
            ].values
            for _, row in levels.iterrows()
        ]
        _boxplot_strip(ax, x_vals, y_groups, color=colors.get(vid, LANCET_BLUE), width=0.35)
        labels = [
            f"({int(row.n_pos_train_theory)}, {int(row.n_neg_train_theory)})"
            for _, row in levels.iterrows()
        ]
        ax.set_xticks(x_vals)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(vid)
        ax.set_xlabel("(n pos, n neg) in train")
        ax.set_ylim(ylim)

    axes[0].set_ylabel("Test AUC-ROC")
    fig.suptitle(
        "Effect of training set size on prediction accuracy (8-fold cross-validation)",
        y=1.02,
    )
    fig.tight_layout()
    _save_figure(fig, out, dpi=150, bbox_inches="tight")
    return out
