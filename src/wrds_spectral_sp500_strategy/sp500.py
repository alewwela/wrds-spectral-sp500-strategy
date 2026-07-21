from __future__ import annotations

import io
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_SOURCE_REPO = "https://github.com/fja05680/sp500"
DEFAULT_SOURCE_REPO_SLUG = "fja05680/sp500"
DEFAULT_SOURCE_VERSION = "c31ac3cc56f28cf9a02b4e694eff7ceab596a0ff"
DEFAULT_SOURCE_FILE = "S&P 500 Historical Components & Changes (Updated).csv"
DEFAULT_SOURCE_LICENSE = "MIT"
SOURCE_SCHEMA = "date,tickers"
SOURCE_CAVEAT = (
    "Research-grade free constituent source assembled by the upstream maintainer "
    "from Trading Evolved/Clenow data, Wikipedia selected changes, and manual "
    "updates; not an official S&P DJI licensed feed."
)

CLASS_SHARE_RE = re.compile(r"^([A-Z0-9]+)[./]([A-Z])$")
SECURITY_CLASS_RE = re.compile(r";\s*COM\s+([A-Z])\s*;")


@dataclass(frozen=True)
class Sp500SourceConfig:
    repo: str = DEFAULT_SOURCE_REPO
    version: str = DEFAULT_SOURCE_VERSION
    file: str = DEFAULT_SOURCE_FILE
    url: str | None = None


@dataclass(frozen=True)
class Sp500Snapshot:
    signal_date: pd.Timestamp
    source_snapshot_date: pd.Timestamp
    source_symbols: tuple[str, ...]


@dataclass(frozen=True)
class MappedUniverse:
    signal_date: pd.Timestamp
    source_snapshot_date: pd.Timestamp
    permnos: tuple[int, ...]
    source_symbol_count: int
    mapped_permno_count: int
    unmapped_symbols: tuple[str, ...]
    ambiguous_symbols: tuple[str, ...]
    duplicate_permnos: tuple[int, ...]


def default_source_url(
    version: str = DEFAULT_SOURCE_VERSION,
    file_name: str = DEFAULT_SOURCE_FILE,
) -> str:
    quoted_file = urllib.parse.quote(file_name)
    return f"https://raw.githubusercontent.com/{DEFAULT_SOURCE_REPO_SLUG}/{version}/{quoted_file}"


def load_source_snapshots(source: str | Path) -> pd.DataFrame:
    text = read_source_text(source)
    frame = pd.read_csv(io.StringIO(text))
    required = {"date", "tickers"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"S&P 500 source missing required columns: {sorted(missing)}")
    frame = frame.loc[:, ["date", "tickers"]].copy()
    frame.loc[:, "date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame.loc[:, "tickers"] = frame["tickers"].fillna("").astype(str)
    frame = frame.dropna(subset=["date"])
    frame = frame.loc[frame["tickers"].str.strip().ne("")]
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    if frame.empty:
        raise ValueError("S&P 500 source has no usable date/tickers rows.")
    return frame.reset_index(drop=True)


def load_configured_source(config: Sp500SourceConfig) -> pd.DataFrame:
    source = config.url or default_source_url(config.version, config.file)
    return load_source_snapshots(source)


def read_source_text(source: str | Path) -> str:
    text = str(source)
    if text.startswith(("http://", "https://")):
        with urllib.request.urlopen(text, timeout=60) as response:
            return response.read().decode("utf-8-sig")
    return Path(source).read_text(encoding="utf-8-sig")


def parse_tickers(value: object) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in str(value).split(","):
        symbol = normalize_symbol(raw)
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    return tuple(out)


def snapshot_as_of(
    source_snapshots: pd.DataFrame,
    signal_date: pd.Timestamp,
) -> Sp500Snapshot:
    signal_date = pd.Timestamp(signal_date)
    eligible = source_snapshots.loc[source_snapshots["date"] <= signal_date]
    if eligible.empty:
        source_start = source_snapshots["date"].min()
        raise ValueError(
            "S&P 500 source starts after signal date: "
            f"source {source_start.date()}, signal {signal_date.date()}"
        )
    row = eligible.iloc[-1]
    return Sp500Snapshot(
        signal_date=signal_date,
        source_snapshot_date=pd.Timestamp(row["date"]),
        source_symbols=parse_tickers(row["tickers"]),
    )


def map_snapshot_to_permnos(
    snapshot: Sp500Snapshot,
    identifier_history: pd.DataFrame,
    *,
    max_identifier_staleness_days: int = 95,
) -> tuple[MappedUniverse, pd.DataFrame]:
    current = identifier_history.loc[identifier_history["Date"] <= snapshot.signal_date]
    if current.empty:
        mapped = MappedUniverse(
            snapshot.signal_date,
            snapshot.source_snapshot_date,
            (),
            len(snapshot.source_symbols),
            0,
            snapshot.source_symbols,
            (),
            (),
        )
        return mapped, pd.DataFrame()
    current = current.sort_values(["PERMNO", "Date"]).groupby("PERMNO", as_index=False).tail(1)
    stale_before = snapshot.signal_date - pd.Timedelta(days=max_identifier_staleness_days)
    current = current.loc[current["Date"] >= stale_before].copy()
    if current.empty:
        mapped = MappedUniverse(
            snapshot.signal_date,
            snapshot.source_snapshot_date,
            (),
            len(snapshot.source_symbols),
            0,
            snapshot.source_symbols,
            (),
            (),
        )
        return mapped, pd.DataFrame()

    current.loc[:, "FeedSymbolNorm"] = current["FeedSymbol"].map(normalize_symbol)
    current.loc[:, "YFTickerNorm"] = current["YFTicker"].map(normalize_symbol)
    current.loc[:, "SecurityClass"] = current["Security"].map(security_share_class)

    rows: list[dict[str, object]] = []
    unmapped: list[str] = []
    ambiguous: list[str] = []
    for source_symbol in snapshot.source_symbols:
        candidates = candidates_for_symbol(source_symbol, current)
        if candidates.empty:
            unmapped.append(source_symbol)
            continue
        unique_permnos = candidates["PERMNO"].dropna().astype(int).unique()
        if len(unique_permnos) != 1:
            ambiguous.append(source_symbol)
            continue
        row = candidates.iloc[0]
        mapped_row = {
            "SignalDate": snapshot.signal_date,
            "SourceSnapshotDate": snapshot.source_snapshot_date,
            "SourceSymbol": source_symbol,
            "PERMNO": int(row["PERMNO"]),
            "MatchedFeedSymbol": row["FeedSymbol"],
            "MatchedYFTicker": row["YFTicker"],
            "MatchedSecurity": row["Security"],
            "IdentifierDate": row["Date"],
        }
        for column in ("MarketCap", "Exchange", "SIC"):
            if column in row.index:
                mapped_row[column] = row[column]
        rows.append(mapped_row)
    mapped_frame = pd.DataFrame(rows)
    duplicate_permnos: tuple[int, ...] = ()
    if not mapped_frame.empty:
        duplicate_permnos = tuple(
            int(value)
            for value in mapped_frame.loc[
                mapped_frame["PERMNO"].duplicated(keep=False), "PERMNO"
            ]
            .drop_duplicates()
            .sort_values()
        )
        mapped_frame = mapped_frame.drop_duplicates("PERMNO", keep="first")
    permnos = tuple(sorted(mapped_frame["PERMNO"].astype(int).unique())) if not mapped_frame.empty else ()
    mapped = MappedUniverse(
        signal_date=snapshot.signal_date,
        source_snapshot_date=snapshot.source_snapshot_date,
        permnos=permnos,
        source_symbol_count=len(snapshot.source_symbols),
        mapped_permno_count=len(permnos),
        unmapped_symbols=tuple(unmapped),
        ambiguous_symbols=tuple(ambiguous),
        duplicate_permnos=duplicate_permnos,
    )
    return mapped, mapped_frame


def candidates_for_symbol(source_symbol: str, current: pd.DataFrame) -> pd.DataFrame:
    source_symbol = normalize_symbol(source_symbol)
    class_match = CLASS_SHARE_RE.match(source_symbol)
    if class_match:
        base_symbol, share_class = class_match.groups()
        candidates = current.loc[
            current["FeedSymbolNorm"].eq(base_symbol)
            | current["YFTickerNorm"].eq(base_symbol)
            | current["FeedSymbolNorm"].eq(source_symbol)
            | current["YFTickerNorm"].eq(source_symbol)
        ].copy()
        class_candidates = candidates.loc[candidates["SecurityClass"].eq(share_class)]
        return class_candidates if not class_candidates.empty else candidates
    return current.loc[
        current["FeedSymbolNorm"].eq(source_symbol)
        | current["YFTickerNorm"].eq(source_symbol)
    ].copy()


def normalize_symbol(value: object) -> str:
    return str(value or "").strip().upper().replace(" ", "")


def security_share_class(value: object) -> str:
    match = SECURITY_CLASS_RE.search(str(value or "").upper())
    return match.group(1) if match else ""


def mapped_universe_audit_row(mapped: MappedUniverse) -> dict[str, object]:
    return {
        "SignalDate": mapped.signal_date,
        "SourceSnapshotDate": mapped.source_snapshot_date,
        "SourceSymbolCount": mapped.source_symbol_count,
        "MappedPermnoCount": mapped.mapped_permno_count,
        "UnmappedCount": len(mapped.unmapped_symbols),
        "AmbiguousCount": len(mapped.ambiguous_symbols),
        "DuplicatePermnoCount": len(mapped.duplicate_permnos),
        "UnmappedSample": ",".join(mapped.unmapped_symbols[:20]),
        "AmbiguousSample": ",".join(mapped.ambiguous_symbols[:20]),
    }


def source_metadata(
    config: Sp500SourceConfig,
    snapshots: pd.DataFrame,
) -> dict[str, object]:
    return {
        "source_repo": config.repo,
        "source_version": config.version,
        "source_file": config.file,
        "source_url": config.url or default_source_url(config.version, config.file),
        "source_schema": SOURCE_SCHEMA,
        "source_license": DEFAULT_SOURCE_LICENSE,
        "source_terms_caveat": SOURCE_CAVEAT,
        "source_rows": int(len(snapshots)),
        "source_min_date": snapshots["date"].min().strftime("%Y-%m-%d"),
        "source_max_date": snapshots["date"].max().strftime("%Y-%m-%d"),
    }


def unique_signal_dates(dates: Iterable[pd.Timestamp]) -> tuple[pd.Timestamp, ...]:
    return tuple(pd.Timestamp(date) for date in sorted(set(pd.to_datetime(list(dates)))))
