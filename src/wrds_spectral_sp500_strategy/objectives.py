from __future__ import annotations

import numpy as np
import pandas as pd


OBJECTIVES = (
    "simple_alpha",
    "capm_alpha",
    "rf_capm_alpha",
    "information_ratio",
    "robust_alpha",
)


def add_selection_score(metrics: pd.DataFrame, objective: str = "robust_alpha") -> pd.DataFrame:
    if metrics.empty:
        return metrics.copy()
    if objective not in OBJECTIVES:
        raise ValueError(f"Unsupported selection objective: {objective}")
    frame = metrics.copy()
    frame.loc[:, "SelectionObjective"] = objective
    frame.loc[:, "SelectionScore"] = objective_score(frame, objective)
    return frame


def sort_by_selection_score(metrics: pd.DataFrame, objective: str = "robust_alpha") -> pd.DataFrame:
    frame = add_selection_score(metrics, objective)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["SelectionScore", "SimpleAlphaAnnualized", "CAGR", "Sharpe"],
        ascending=[False, False, False, False],
    )


def objective_score(metrics: pd.DataFrame, objective: str) -> pd.Series:
    if objective == "simple_alpha":
        return numeric(metrics, "SimpleAlphaAnnualized")
    if objective == "capm_alpha":
        return numeric(metrics, "CAPMAlphaAnnualized")
    if objective == "rf_capm_alpha":
        return numeric(metrics, "CAPMAlphaRiskFreeAnnualized")
    if objective == "information_ratio":
        return numeric(metrics, "InformationRatio")
    simple_alpha = numeric(metrics, "SimpleAlphaAnnualized")
    information_ratio = numeric(metrics, "InformationRatio")
    sharpe = numeric(metrics, "Sharpe")
    max_drawdown = numeric(metrics, "MaxDrawdown")
    worst_excess_year = numeric(metrics, "WorstExcessYear")
    excess_year_win_rate = numeric(metrics, "ExcessYearWinRate")
    active_share = numeric(metrics, "ActiveShare")
    months = numeric(metrics, "Months")

    active_penalty = (0.65 - active_share).clip(lower=0.0).fillna(0.10)
    sample_penalty = ((60.0 - months).clip(lower=0.0) / 60.0).fillna(0.0)
    return (
        simple_alpha.fillna(-np.inf)
        + 0.03 * information_ratio.fillna(0.0)
        + 0.02 * sharpe.fillna(0.0)
        + 0.08 * excess_year_win_rate.fillna(0.0)
        + 0.18 * worst_excess_year.fillna(0.0)
        + 0.12 * max_drawdown.fillna(-1.0)
        - 0.08 * active_penalty
        - 0.05 * sample_penalty
    )


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")
