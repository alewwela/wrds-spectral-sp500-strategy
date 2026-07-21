from __future__ import annotations

import pandas as pd

from wrds_spectral_sp500_strategy.sp500 import MappedUniverse


def broad_wrds_universe_as_of(
    identifier_history: pd.DataFrame,
    signal_date: pd.Timestamp,
    *,
    max_identifier_staleness_days: int = 95,
) -> tuple[MappedUniverse, pd.DataFrame]:
    """Return all current WRDS PERMNOs visible in the PIT identifier panel."""
    signal_date = pd.Timestamp(signal_date)
    current = identifier_history.loc[identifier_history["Date"] <= signal_date].copy()
    if current.empty:
        return _empty_broad_universe(signal_date)

    current = current.sort_values(["PERMNO", "Date"]).groupby("PERMNO", as_index=False).tail(1)
    stale_before = signal_date - pd.Timedelta(days=max_identifier_staleness_days)
    current = current.loc[current["Date"] >= stale_before].copy()
    if current.empty:
        return _empty_broad_universe(signal_date)

    current = current.drop_duplicates("PERMNO", keep="last").copy()
    current.loc[:, "SignalDate"] = signal_date
    current.loc[:, "SourceSnapshotDate"] = current["Date"].max()
    current.loc[:, "SourceSymbol"] = current["FeedSymbol"].fillna("").astype(str)
    missing_symbol = current["SourceSymbol"].str.strip().eq("")
    current.loc[missing_symbol, "SourceSymbol"] = current.loc[missing_symbol, "PERMNO"].astype(str)
    current.loc[:, "MatchedFeedSymbol"] = current["FeedSymbol"].fillna("").astype(str)
    yfticker = (
        current["YFTicker"]
        if "YFTicker" in current.columns
        else pd.Series("", index=current.index)
    )
    current.loc[:, "MatchedYFTicker"] = yfticker.fillna("").astype(str)
    current.loc[:, "MatchedSecurity"] = current["Security"].fillna("").astype(str)
    current.loc[:, "IdentifierDate"] = current["Date"]

    base_columns = [
        "SignalDate",
        "SourceSnapshotDate",
        "SourceSymbol",
        "PERMNO",
        "MatchedFeedSymbol",
        "MatchedYFTicker",
        "MatchedSecurity",
        "IdentifierDate",
    ]
    optional_columns = [
        column
        for column in ("MarketCap", "Exchange", "SIC")
        if column in current.columns
    ]
    mapped_frame = current[base_columns + optional_columns].copy()
    permnos = tuple(sorted(mapped_frame["PERMNO"].astype(int).unique()))
    mapped = MappedUniverse(
        signal_date=signal_date,
        source_snapshot_date=pd.Timestamp(mapped_frame["SourceSnapshotDate"].max()),
        permnos=permnos,
        source_symbol_count=len(mapped_frame),
        mapped_permno_count=len(permnos),
        unmapped_symbols=(),
        ambiguous_symbols=(),
        duplicate_permnos=(),
    )
    return mapped, mapped_frame


def _empty_broad_universe(signal_date: pd.Timestamp) -> tuple[MappedUniverse, pd.DataFrame]:
    mapped = MappedUniverse(
        signal_date=signal_date,
        source_snapshot_date=pd.NaT,
        permnos=(),
        source_symbol_count=0,
        mapped_permno_count=0,
        unmapped_symbols=(),
        ambiguous_symbols=(),
        duplicate_permnos=(),
    )
    return mapped, pd.DataFrame()
