from __future__ import annotations

import numpy as np
import pandas as pd


def portfolio_weights(
    selected_scores: pd.Series,
    *,
    weighting: str = "equal",
    rank_decay: float = 0.5,
) -> pd.Series:
    """Return long-only portfolio weights aligned to the selected score index."""
    if selected_scores.empty:
        return pd.Series(dtype=float, name="Weight")
    normalized = weighting.lower().replace("-", "_")
    if normalized == "equal":
        weights = pd.Series(
            1.0 / len(selected_scores),
            index=selected_scores.index,
            name="Weight",
        )
        return weights
    if normalized != "rank_decay":
        raise ValueError(f"Unsupported portfolio weighting: {weighting}")
    if not 0.0 < rank_decay <= 1.0:
        raise ValueError("rank_decay must be in the interval (0, 1].")
    ranks = np.arange(len(selected_scores), dtype=float)
    raw = pd.Series(rank_decay**ranks, index=selected_scores.index, name="Weight")
    return raw / raw.sum()
