from __future__ import annotations

import pandas as pd


PERIOD_TO_FREQUENCY = {
    "3M": "quarterly",
    "6M": "semiannual",
    "1Y": "yearly",
}


def select_rebalance_dates(
    dates: list[pd.Timestamp] | pd.Series | pd.DatetimeIndex,
    period: str,
) -> list[pd.Timestamp]:
    ordered = pd.Series(pd.to_datetime(list(dates))).dropna().sort_values()
    if ordered.empty:
        return []
    normalized = period.upper()
    if normalized == "1M":
        return list(ordered)
    if normalized == "3M":
        groups = ordered.dt.to_period("Q-DEC").astype(str)
    elif normalized == "6M":
        groups = ordered.dt.year.astype(str) + "H" + ((ordered.dt.month > 6).astype(int) + 1).astype(str)
    elif normalized == "1Y":
        groups = ordered.dt.to_period("Y-DEC").astype(str)
    else:
        raise ValueError(f"Unsupported rebalance period: {period}")
    frame = pd.DataFrame({"Date": ordered, "Group": groups})
    return list(frame.groupby("Group", sort=True)["Date"].max())


def period_frequency(period: str) -> str:
    return PERIOD_TO_FREQUENCY.get(period.upper(), period.lower())


def holding_period_dates(
    returns: pd.DataFrame,
    signal_date: pd.Timestamp,
    next_rebalance_date: pd.Timestamp | None,
) -> pd.Index:
    dates = returns.loc[returns["Date"] > signal_date, "Date"].drop_duplicates()
    if next_rebalance_date is not None:
        dates = dates.loc[dates <= next_rebalance_date]
    return pd.Index(pd.to_datetime(dates).sort_values())

