"""Phenotype feature expansion for tree-based classifiers."""
from __future__ import annotations

import numpy as np


def expand_phenotype(Y: np.ndarray) -> np.ndarray:
    """Map Y (N, 2) to Phi (N, 4): [Y1, Y2, r, Y1*Y2]."""
    y1 = Y[:, 0]
    y2 = Y[:, 1]
    r = np.sqrt(y1 * y1 + y2 * y2)
    return np.column_stack([y1, y2, r, y1 * y2])
