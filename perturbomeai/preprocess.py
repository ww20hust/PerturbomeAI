"""Per-column quantile transform and age scaling.

Routine clinical measurements live on heterogeneous scales and have skewed,
heavy-tailed distributions. We map each feature column to a common uniform
[0, 1] space with a per-column ``QuantileTransformer`` (fit on observed values
only, so missing entries stay missing). Age is handled separately by simple
division, because it is already bounded and roughly uniform.

The transform is reversible: ``inverse_transform`` returns columns to their
original units, optionally only for cells that were imputed downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

AGE_DIVISOR = 100.0


@dataclass
class ColumnQuantileTransform:
    """Fitted per-column quantile transform plus age scaling.

    Attributes:
        feature_order: the canonical column order the transform was fit on.
        age_col: column scaled by division instead of quantile transform.
        age_divisor: divisor applied to ``age_col``.
        transformers: mapping column -> fitted QuantileTransformer.
    """

    feature_order: list[str]
    age_col: str = "age"
    age_divisor: float = AGE_DIVISOR
    transformers: dict[str, QuantileTransformer] = field(default_factory=dict)

    @property
    def qt_columns(self) -> list[str]:
        return [c for c in self.feature_order if c != self.age_col]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in self.feature_order:
            s = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
            if col == self.age_col:
                out[col] = s.astype(float) / float(self.age_divisor)
                continue
            qt = self.transformers.get(col)
            vals = s.to_numpy(dtype=float)
            obs = np.isfinite(vals)
            res = np.full(vals.shape, np.nan, dtype=float)
            if qt is not None and obs.any():
                res[obs] = qt.transform(vals[obs].reshape(-1, 1)).ravel()
            out[col] = res
        return out

    def inverse_transform(
        self,
        df_qt: pd.DataFrame,
        imputed_mask: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        out = pd.DataFrame(index=df_qt.index)
        for col in self.feature_order:
            s = pd.to_numeric(df_qt[col], errors="coerce")
            if col == self.age_col:
                out[col] = s.astype(float) * float(self.age_divisor)
                continue
            qt = self.transformers.get(col)
            vals = s.to_numpy(dtype=float)
            res = vals.copy()
            sel = np.isfinite(vals)
            if imputed_mask is not None and col in imputed_mask.columns:
                sel = sel & imputed_mask[col].fillna(False).to_numpy(dtype=bool)
            if qt is not None and sel.any():
                res[sel] = qt.inverse_transform(vals[sel].reshape(-1, 1)).ravel()
            out[col] = res
        return out


def fit_column_quantile_transform(
    df: pd.DataFrame,
    feature_order: list[str],
    *,
    age_col: str = "age",
    age_divisor: float = AGE_DIVISOR,
    n_quantiles: int = 1000,
    random_state: int = 42,
) -> ColumnQuantileTransform:
    """Fit a per-column quantile transform on observed values only."""
    transformers: dict[str, QuantileTransformer] = {}
    for col in feature_order:
        if col == age_col:
            continue
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        obs = vals[np.isfinite(vals)]
        if obs.size < 2:
            continue
        nq = int(min(n_quantiles, obs.size))
        qt = QuantileTransformer(
            n_quantiles=nq,
            output_distribution="uniform",
            subsample=10_000_000,
            random_state=random_state,
        )
        qt.fit(obs.reshape(-1, 1))
        transformers[col] = qt
    return ColumnQuantileTransform(
        feature_order=list(feature_order),
        age_col=age_col,
        age_divisor=age_divisor,
        transformers=transformers,
    )
