from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile

import pandas as pd


DATE_COLUMNS = ("Date", "AsOfDate", "SnapshotDate", "DataAvailableDate")
RETURN_ALIASES = ("AdjReturn", "Return", "RET", "ret", "mthret", "MthRet")
BENCHMARK_ALIASES = ("BenchmarkReturn", "SPYReturn", *RETURN_ALIASES)


def read_table(path: str | Path, *, usecols: Iterable[str] | None = None) -> pd.DataFrame:
    """Read a CSV, TSV text file, zip-wrapped CSV, or parquet table."""
    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(table_path)
    suffixes = [suffix.lower() for suffix in table_path.suffixes]
    if suffixes[-1:] == [".parquet"]:
        return pd.read_parquet(table_path, columns=list(usecols) if usecols else None)
    if suffixes[-1:] == [".zip"]:
        return pd.read_csv(table_path, compression="zip", usecols=usecols)
    if suffixes[-1:] == [".csv"] or suffixes[-2:] == [".tsv", ".txt"]:
        sep = "\t" if ".tsv" in suffixes else ","
        return pd.read_csv(table_path, sep=sep, usecols=usecols)
    raise ValueError(f"Unsupported table format: {table_path}")


def read_table_header(path: str | Path) -> pd.Index:
    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(table_path)
    suffixes = [suffix.lower() for suffix in table_path.suffixes]
    if suffixes[-1:] == [".parquet"]:
        return pd.read_parquet(table_path).head(0).columns
    if suffixes[-1:] == [".zip"]:
        return pd.read_csv(table_path, compression="zip", nrows=0).columns
    if suffixes[-1:] == [".csv"] or suffixes[-2:] == [".tsv", ".txt"]:
        sep = "\t" if ".tsv" in suffixes else ","
        return pd.read_csv(table_path, sep=sep, nrows=0).columns
    raise ValueError(f"Unsupported table format: {table_path}")


def read_chunk_manifest(
    repo_root: str | Path,
    manifest_path: str | Path,
    *,
    usecols: Iterable[str] | None = None,
) -> pd.DataFrame:
    root = Path(repo_root)
    manifest = Path(manifest_path)
    if not manifest.is_absolute():
        manifest = root / manifest
    chunks = json.loads(manifest.read_text(encoding="utf-8"))
    frames: list[pd.DataFrame] = []
    for chunk in chunks:
        path = root / chunk["path"]
        if path.suffix.lower() == ".zip":
            with ZipFile(path) as archive:
                with archive.open(chunk["inner_path"]) as handle:
                    frames.append(pd.read_csv(handle, usecols=usecols))
        else:
            frames.append(read_table(path, usecols=usecols))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def normalize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in DATE_COLUMNS:
        if column in out.columns:
            out.loc[:, column] = pd.to_datetime(out[column], errors="coerce")
    return out


def load_returns(path: str | Path) -> pd.DataFrame:
    returns = normalize_dates(read_table(path, usecols=["Date", "PERMNO", "AdjReturn"]))
    returns.loc[:, "PERMNO"] = pd.to_numeric(returns["PERMNO"], errors="coerce")
    returns.loc[:, "AdjReturn"] = pd.to_numeric(returns["AdjReturn"], errors="coerce")
    returns = returns.dropna(subset=["Date", "PERMNO", "AdjReturn"]).copy()
    returns.loc[:, "PERMNO"] = returns["PERMNO"].astype("int64")
    return (
        returns[["Date", "PERMNO", "AdjReturn"]]
        .sort_values(["PERMNO", "Date"])
        .drop_duplicates(["PERMNO", "Date"], keep="last")
    )


def load_benchmark(path: str | Path) -> pd.DataFrame:
    frame = normalize_dates(read_table(path))
    return_col = find_first_column(frame, BENCHMARK_ALIASES)
    if return_col is None:
        raise ValueError("Benchmark file must include a return column.")
    if return_col != "BenchmarkReturn":
        frame = frame.rename(columns={return_col: "BenchmarkReturn"})
    frame.loc[:, "BenchmarkReturn"] = pd.to_numeric(
        frame["BenchmarkReturn"], errors="coerce"
    )
    frame = frame.dropna(subset=["Date", "BenchmarkReturn"])
    return frame[["Date", "BenchmarkReturn"]].sort_values("Date").drop_duplicates(
        "Date", keep="last"
    )


def load_risk_free(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    frame = normalize_dates(read_table(path))
    if "RiskFreeReturn" not in frame.columns:
        return None
    frame.loc[:, "RiskFreeReturn"] = pd.to_numeric(
        frame["RiskFreeReturn"], errors="coerce"
    )
    frame = frame.dropna(subset=["Date", "RiskFreeReturn"])
    return frame[["Date", "RiskFreeReturn"]].sort_values("Date").drop_duplicates(
        "Date", keep="last"
    )


def load_factor_frames(
    paths: Iterable[str | Path],
    score_columns: Iterable[str],
) -> list[pd.DataFrame]:
    wanted = set(score_columns)
    frames: list[pd.DataFrame] = []
    for path in paths:
        table_path = Path(path)
        columns = read_table_header(table_path)
        available = [column for column in wanted if column in columns]
        if not available:
            continue
        common = ["Date", "PERMNO", *available]
        if "DataAvailableDate" in columns:
            common.insert(2, "DataAvailableDate")
        frame = normalize_dates(read_table(table_path, usecols=common))
        frame.loc[:, "PERMNO"] = pd.to_numeric(frame["PERMNO"], errors="coerce")
        frame = frame.dropna(subset=["Date", "PERMNO"]).copy()
        frame.loc[:, "PERMNO"] = frame["PERMNO"].astype("int64")
        for column in available:
            frame.loc[:, column] = pd.to_numeric(frame[column], errors="coerce")
        frames.append(frame)
    return frames


def select_factor_scores_as_of(
    factor_frames: list[pd.DataFrame],
    signal_date: pd.Timestamp,
    permnos: Iterable[int],
    score_columns: Iterable[str],
) -> pd.DataFrame:
    """Select latest PIT factor rows with Date and DataAvailableDate <= signal."""
    permno_index = pd.Index([int(value) for value in permnos], name="PERMNO")
    selected = pd.DataFrame(index=permno_index)
    wanted = list(score_columns)

    for frame in factor_frames:
        available = [column for column in wanted if column in frame.columns]
        if not available:
            continue
        current = frame.loc[
            frame["PERMNO"].isin(permno_index) & (frame["Date"] <= signal_date)
        ].copy()
        if "DataAvailableDate" in current.columns:
            current = current.loc[current["DataAvailableDate"] <= signal_date]
        if current.empty:
            continue
        current = current.sort_values(["PERMNO", "Date"])
        latest = current.groupby("PERMNO", as_index=False).tail(1)
        selected = selected.join(latest.set_index("PERMNO")[available], how="left")
    return selected


def factor_availability_audit(
    factor_frames: list[pd.DataFrame],
    signal_date: pd.Timestamp,
    permnos: Iterable[int],
) -> dict[str, object]:
    permno_set = set(int(value) for value in permnos)
    max_factor_date = pd.NaT
    max_available_date = pd.NaT
    future_available_rows_excluded = 0
    for frame in factor_frames:
        current = frame.loc[
            frame["PERMNO"].isin(permno_set) & (frame["Date"] <= signal_date)
        ].copy()
        if current.empty:
            continue
        if "DataAvailableDate" in current.columns:
            future_available_rows_excluded += int(
                current["DataAvailableDate"].gt(signal_date).sum()
            )
            current = current.loc[current["DataAvailableDate"] <= signal_date]
        if current.empty:
            continue
        max_factor_date = max_timestamp(max_factor_date, current["Date"].max())
        if "DataAvailableDate" in current.columns:
            max_available_date = max_timestamp(
                max_available_date, current["DataAvailableDate"].max()
            )
    return {
        "MaxFactorDateUsed": max_factor_date,
        "MaxDataAvailableDateUsed": max_available_date,
        "FutureAvailableRowsExcluded": future_available_rows_excluded,
    }


def load_identifier_history(
    pit_universe_repo: str | Path,
    *,
    manifest_path: str = "datasets/wrds_crsp_ciz_monthly_return_panel_chunks.json",
) -> pd.DataFrame:
    columns = ["Date", "PERMNO", "FeedSymbol", "Security"]
    frame = read_chunk_manifest(pit_universe_repo, manifest_path, usecols=columns)
    frame = normalize_dates(frame)
    if "YFTicker" not in frame.columns:
        frame.loc[:, "YFTicker"] = ""
    frame.loc[:, "PERMNO"] = pd.to_numeric(frame["PERMNO"], errors="coerce")
    frame = frame.dropna(subset=["Date", "PERMNO"]).copy()
    frame.loc[:, "PERMNO"] = frame["PERMNO"].astype("int64")
    for column in ("FeedSymbol", "YFTicker", "Security"):
        frame.loc[:, column] = frame[column].fillna("").astype(str)
    return frame.sort_values(["PERMNO", "Date"]).drop_duplicates(
        ["PERMNO", "Date"], keep="last"
    )


def find_first_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lookup = {_normalize_column_name(column): column for column in frame.columns}
    for candidate in candidates:
        match = lookup.get(_normalize_column_name(candidate))
        if match is not None:
            return str(match)
    return None


def max_timestamp(left: pd.Timestamp, right: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(left):
        return right
    if pd.isna(right):
        return left
    return max(left, right)


def _normalize_column_name(value: object) -> str:
    return "".join(char.lower() for char in str(value) if char.isalnum())
