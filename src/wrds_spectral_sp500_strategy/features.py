from __future__ import annotations

import numpy as np
import pandas as pd


def feature_name(window: int) -> str:
    return f"ret_{window}m"


def trailing_horizon_returns(
    returns: pd.DataFrame,
    signal_date: pd.Timestamp,
    permnos: list[int] | pd.Index,
    *,
    windows: tuple[int, ...],
    min_history_months: int,
    include_signal_date: bool = False,
) -> pd.DataFrame:
    max_window = max(windows)
    permno_set = set(int(value) for value in permnos)
    date_mask = (
        returns["Date"] <= signal_date
        if include_signal_date
        else returns["Date"] < signal_date
    )
    panel = returns.loc[date_mask & returns["PERMNO"].isin(permno_set)].copy()
    if panel.empty:
        return pd.DataFrame(columns=[feature_name(window) for window in windows])

    rows: dict[int, dict[str, float]] = {}
    for permno, group in panel.sort_values(["PERMNO", "Date"]).groupby("PERMNO"):
        values = group["AdjReturn"].dropna().to_numpy(dtype=float)[-max_window:]
        if len(values) < min_history_months:
            continue
        row: dict[str, float] = {}
        for window in windows:
            row[feature_name(window)] = (
                float(np.prod(1.0 + values[-window:]) - 1.0)
                if len(values) >= window
                else np.nan
            )
        rows[int(permno)] = row
    features = pd.DataFrame.from_dict(rows, orient="index")
    features.index.name = "PERMNO"
    return features.sort_index()


def augment_return_features(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    for horizon in (3, 6, 9, 12, 18, 24, 36, 48, 60):
        columns = [feature_name(window) for window in range(1, horizon + 1)]
        if not all(column in out.columns for column in columns):
            continue
        frame = out[columns]
        prefix = "ret_horizon"
        out.loc[:, f"{prefix}_mean_{horizon}m"] = frame.mean(axis=1)
        out.loc[:, f"{prefix}_vol_{horizon}m"] = frame.std(axis=1)
        out.loc[:, f"{prefix}_hit_{horizon}m"] = (frame > 0.0).mean(axis=1)
        out.loc[:, f"{prefix}_best_{horizon}m"] = frame.max(axis=1)
        out.loc[:, f"{prefix}_worst_{horizon}m"] = frame.min(axis=1)
        horizon_return = feature_name(horizon)
        one_month_return = feature_name(1)
        if horizon > 1 and horizon_return in out.columns:
            denominator = (1.0 + out[one_month_return]).replace(0.0, np.nan)
            out.loc[:, f"ret_skip1_{horizon}m"] = (1.0 + out[horizon_return]) / denominator - 1.0
        vol_column = f"{prefix}_vol_{horizon}m"
        if horizon_return in out.columns and vol_column in out.columns:
            out.loc[:, f"ret_efficiency_{horizon}m"] = (
                out[horizon_return] / out[vol_column].replace(0.0, np.nan)
            )
    for short_window, long_window in (
        (1, 3),
        (3, 12),
        (4, 6),
        (6, 24),
        (12, 36),
        (24, 60),
    ):
        short_col = feature_name(short_window)
        long_col = feature_name(long_window)
        if short_col in out.columns and long_col in out.columns:
            out.loc[:, f"ret_spread_{short_window}_{long_window}m"] = out[short_col] - out[long_col]
    return out.replace([np.inf, -np.inf], np.nan)


def augment_cluster_features(
    features: pd.DataFrame,
    clusters: pd.Series,
    *,
    residual_columns: tuple[str, ...],
) -> pd.DataFrame:
    out = features.copy()
    if not residual_columns:
        return out
    aligned = out.join(clusters.rename("Cluster"), how="left")
    if aligned["Cluster"].isna().all():
        return out
    grouped = aligned.groupby("Cluster", dropna=True)
    for column in residual_columns:
        if column not in aligned.columns:
            continue
        out.loc[:, f"cluster_resid_{column}"] = aligned[column] - grouped[column].transform("mean")
    return out.replace([np.inf, -np.inf], np.nan)


def zscore_columns(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    means = numeric.mean(axis=0, skipna=True)
    stds = numeric.std(axis=0, skipna=True).replace(0.0, np.nan)
    scored = numeric.subtract(means, axis=1).divide(stds, axis=1)
    return scored.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def fixed_rule_score(
    score_inputs: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    missing = [column for column in weights if column not in score_inputs.columns]
    if missing:
        raise ValueError(f"Missing fixed rule score columns: {', '.join(missing)}")
    normalized = zscore_columns(score_inputs[list(weights)])
    score = pd.Series(0.0, index=normalized.index, name="Score")
    for column, weight in weights.items():
        score = score.add(normalized[column] * float(weight), fill_value=0.0)
    return score
