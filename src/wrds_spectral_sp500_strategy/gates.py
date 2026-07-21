from __future__ import annotations

import numpy as np
import pandas as pd

from wrds_spectral_sp500_strategy.config import GateConditionConfig, GateConfig


def apply_expanding_gate(
    raw_returns: pd.DataFrame,
    gate: GateConfig,
    *,
    oos_start_year: int,
    oos_end_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if raw_returns.empty:
        return raw_returns.copy(), pd.DataFrame()
    frame = raw_returns.copy()
    frame.loc[:, "Date"] = pd.to_datetime(frame["Date"])
    frame.loc[:, "SignalDate"] = pd.to_datetime(frame["SignalDate"])
    if not gate.enabled:
        frame = frame.loc[frame["Date"].dt.year.between(oos_start_year, oos_end_year)].copy()
        frame.loc[:, "TopScoreThreshold"] = np.nan
        frame.loc[:, "MedianWorstThreshold"] = np.nan
        frame.loc[:, "GatePassed"] = True
        frame.loc[:, "PortfolioReturn"] = frame["RawPortfolioReturn"]
        frame.loc[:, "GrossExposure"] = 1.0
        frame.loc[:, "NetExposure"] = 1.0
        frame.loc[:, "ActiveNames"] = frame["RawActiveNames"]
        thresholds = pd.DataFrame(
            [
                {
                    "OOSYear": year,
                    "TrainStartYear": gate.train_start_year,
                    "TrainEndYear": year - 1,
                    "GateEnabled": False,
                    "TrainSignalCount": 0,
                    "TopScoreQuantile": np.nan,
                    "TopScoreThreshold": np.nan,
                    "MedianWorstQuantile": np.nan,
                    "MedianWorstThreshold": np.nan,
                }
                for year in range(oos_start_year, oos_end_year + 1)
            ]
        )
        return frame.sort_values("Date"), thresholds
    if gate.conditions:
        return apply_generic_expanding_gate(
            frame,
            gate,
            oos_start_year=oos_start_year,
            oos_end_year=oos_end_year,
        )
    out_frames: list[pd.DataFrame] = []
    threshold_rows: list[dict[str, object]] = []
    for year in range(oos_start_year, oos_end_year + 1):
        train = (
            frame.loc[frame["Date"].dt.year.between(gate.train_start_year, year - 1)]
            .drop_duplicates("SignalDate")
            .loc[:, ["SignalDate", "top_score", "median_ret_horizon_worst_60m"]]
        )
        apply = frame.loc[frame["Date"].dt.year.eq(year)].copy()
        if apply.empty:
            continue
        top_threshold = train["top_score"].quantile(gate.top_score_quantile)
        worst_threshold = train["median_ret_horizon_worst_60m"].quantile(
            gate.median_worst_quantile
        )
        signal_apply = apply.drop_duplicates("SignalDate").copy()
        top_pass = compare(
            signal_apply["top_score"].astype(float),
            gate.top_score_operator,
            top_threshold,
        )
        worst_pass = compare(
            signal_apply["median_ret_horizon_worst_60m"].astype(float),
            gate.median_worst_operator,
            worst_threshold,
        )
        gate_by_signal = pd.Series(
            (top_pass & worst_pass).to_numpy(dtype=bool),
            index=signal_apply["SignalDate"],
        )
        apply.loc[:, "TopScoreThreshold"] = top_threshold
        apply.loc[:, "MedianWorstThreshold"] = worst_threshold
        apply.loc[:, "GatePassed"] = apply["SignalDate"].map(gate_by_signal).fillna(False)
        apply.loc[:, "PortfolioReturn"] = np.where(
            apply["GatePassed"], apply["RawPortfolioReturn"], 0.0
        )
        apply.loc[:, "GrossExposure"] = np.where(apply["GatePassed"], 1.0, 0.0)
        apply.loc[:, "NetExposure"] = np.where(apply["GatePassed"], 1.0, 0.0)
        apply.loc[:, "ActiveNames"] = np.where(
            apply["GatePassed"], apply["RawActiveNames"], 0
        )
        out_frames.append(apply)
        threshold_rows.append(
            {
                "OOSYear": year,
                "TrainStartYear": gate.train_start_year,
                "TrainEndYear": year - 1,
                "TrainSignalCount": int(len(train)),
                "TopScoreQuantile": gate.top_score_quantile,
                "TopScoreThreshold": top_threshold,
                "MedianWorstQuantile": gate.median_worst_quantile,
                "MedianWorstThreshold": worst_threshold,
            }
        )
    if not out_frames:
        return pd.DataFrame(), pd.DataFrame(threshold_rows)
    gated = pd.concat(out_frames, ignore_index=True).sort_values("Date")
    return gated, pd.DataFrame(threshold_rows)


def apply_generic_expanding_gate(
    frame: pd.DataFrame,
    gate: GateConfig,
    *,
    oos_start_year: int,
    oos_end_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    conditions = tuple(gate.conditions)
    validate_condition_columns(frame, conditions)
    out_frames: list[pd.DataFrame] = []
    threshold_rows: list[dict[str, object]] = []
    metrics = ["SignalDate", *dict.fromkeys(condition.metric for condition in conditions)]
    for year in range(oos_start_year, oos_end_year + 1):
        train = (
            frame.loc[frame["Date"].dt.year.between(gate.train_start_year, year - 1)]
            .drop_duplicates("SignalDate")
            .loc[:, metrics]
        )
        apply = frame.loc[frame["Date"].dt.year.eq(year)].copy()
        if apply.empty:
            continue
        signal_apply = apply.drop_duplicates("SignalDate").copy()
        gate_pass = pd.Series(True, index=signal_apply.index)
        threshold_row: dict[str, object] = {
            "OOSYear": year,
            "TrainStartYear": gate.train_start_year,
            "TrainEndYear": year - 1,
            "GateEnabled": True,
            "TrainSignalCount": int(len(train)),
        }
        for index, condition in enumerate(conditions, start=1):
            threshold = train[condition.metric].quantile(condition.quantile)
            values = pd.to_numeric(signal_apply[condition.metric], errors="coerce")
            gate_pass &= compare(values, condition.operator, threshold)
            threshold_column = f"GateThreshold{index}"
            apply.loc[:, threshold_column] = threshold
            threshold_row.update(
                {
                    f"Condition{index}Metric": condition.metric,
                    f"Condition{index}Operator": condition.operator,
                    f"Condition{index}Quantile": condition.quantile,
                    f"Condition{index}Threshold": threshold,
                }
            )
        gate_by_signal = pd.Series(
            gate_pass.to_numpy(dtype=bool),
            index=signal_apply["SignalDate"],
        )
        apply.loc[:, "GatePassed"] = apply["SignalDate"].map(gate_by_signal).fillna(False)
        apply.loc[:, "PortfolioReturn"] = np.where(
            apply["GatePassed"], apply["RawPortfolioReturn"], 0.0
        )
        apply.loc[:, "GrossExposure"] = np.where(apply["GatePassed"], 1.0, 0.0)
        apply.loc[:, "NetExposure"] = np.where(apply["GatePassed"], 1.0, 0.0)
        apply.loc[:, "ActiveNames"] = np.where(
            apply["GatePassed"], apply["RawActiveNames"], 0
        )
        out_frames.append(apply)
        threshold_rows.append(threshold_row)
    if not out_frames:
        return pd.DataFrame(), pd.DataFrame(threshold_rows)
    return pd.concat(out_frames, ignore_index=True).sort_values("Date"), pd.DataFrame(threshold_rows)


def validate_condition_columns(
    frame: pd.DataFrame,
    conditions: tuple[GateConditionConfig, ...],
) -> None:
    missing = [condition.metric for condition in conditions if condition.metric not in frame.columns]
    if missing:
        raise ValueError(f"Missing gate metric columns: {', '.join(missing)}")


def compare(values: pd.Series, operator: str, threshold: float) -> pd.Series:
    if pd.isna(threshold):
        return pd.Series(False, index=values.index)
    if operator == "<":
        return values < float(threshold)
    if operator == ">":
        return values > float(threshold)
    raise ValueError(f"Unsupported gate operator: {operator}")
