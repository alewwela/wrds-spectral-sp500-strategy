from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from wrds_spectral_sp500_strategy.clustering import cluster_returns
from wrds_spectral_sp500_strategy.config import StrategyConfig
from wrds_spectral_sp500_strategy.data import (
    factor_availability_audit,
    load_benchmark,
    load_factor_frames,
    load_identifier_history,
    load_returns,
    load_risk_free,
    select_factor_scores_as_of,
)
from wrds_spectral_sp500_strategy.features import (
    augment_cluster_features,
    augment_return_features,
    fixed_rule_score,
    trailing_horizon_returns,
)
from wrds_spectral_sp500_strategy.gates import apply_expanding_gate
from wrds_spectral_sp500_strategy.performance import (
    attach_optional_series,
    performance_summary,
    yearly_returns,
)
from wrds_spectral_sp500_strategy.rebalance import (
    holding_period_dates,
    period_frequency,
    select_rebalance_dates,
)
from wrds_spectral_sp500_strategy.sp500 import (
    load_configured_source,
    map_snapshot_to_permnos,
    mapped_universe_audit_row,
    snapshot_as_of,
    source_metadata,
)


BROAD_UNIVERSE_RESULTS = {
    "3M": {
        "CAGR": 0.09692273416452069,
        "BenchmarkCAGR": 0.0981771592119427,
        "SimpleAlphaAnnualized": -0.0012544250474220142,
        "CAPMAlphaAnnualized": 0.03509198882407172,
        "CAPMAlphaRiskFreeAnnualized": 0.03982946754193706,
        "Beta": 0.8129745476524903,
        "MaxDrawdown": -0.38382432983222037,
        "Sharpe": 0.49893239977197035,
        "InformationRatio": 0.07077259619461054,
        "ActiveShare": 0.5680272108843537,
    },
    "6M": {
        "CAGR": 0.1550494583895663,
        "BenchmarkCAGR": 0.0981771592119427,
        "SimpleAlphaAnnualized": 0.0568722991776236,
        "CAPMAlphaAnnualized": 0.0875125041120588,
        "CAPMAlphaRiskFreeAnnualized": 0.06341303461430958,
        "Beta": 0.8483061304389753,
        "MaxDrawdown": -0.3826307967581578,
        "Sharpe": 0.7121375713393565,
        "InformationRatio": 0.32620427266941504,
        "ActiveShare": 0.5544217687074829,
    },
    "1Y": {
        "CAGR": 0.2549315487145609,
        "BenchmarkCAGR": 0.0981771592119427,
        "SimpleAlphaAnnualized": 0.15675438950261822,
        "CAPMAlphaAnnualized": 0.1723282738766443,
        "CAPMAlphaRiskFreeAnnualized": 0.19666779432775883,
        "Beta": 0.9847660331761697,
        "MaxDrawdown": -0.3506698444687383,
        "Sharpe": 0.9792644131107028,
        "InformationRatio": 0.7033753863868084,
        "ActiveShare": 0.6768707482993197,
    },
}


@dataclass
class SignalResult:
    signal_date: pd.Timestamp
    source_snapshot_date: pd.Timestamp | pd.NaT
    selected: pd.Series
    score_inputs: pd.DataFrame
    mapped_permnos: tuple[int, ...]
    raw_universe_count: int
    mapped_universe_count: int
    status: str
    audit: dict[str, object]
    holdings: pd.DataFrame

    @property
    def valid(self) -> bool:
        return self.status == "ok" and len(self.selected) > 0


@dataclass
class BacktestOutputs:
    summary: pd.DataFrame
    yearly: pd.DataFrame
    comparison: pd.DataFrame
    output_dir: Path


def run_backtests(config: StrategyConfig) -> BacktestOutputs:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    returns = load_returns(config.returns_path)
    benchmark = load_benchmark(config.benchmark_path)
    risk_free = load_risk_free(config.risk_free_path)
    factor_columns = [
        column
        for column in config.score_weights
        if not column.startswith("ret_") and not column.startswith("cluster_")
    ]
    factors = load_factor_frames(config.factor_paths, factor_columns)
    identifier_history = load_identifier_history(config.pit_universe_repo)
    sp500_source = load_configured_source(config.sp500_source)

    all_dates = pd.Series(returns["Date"].dropna().unique())
    all_dates = pd.to_datetime(all_dates).sort_values()
    start = pd.Timestamp(year=config.start_year, month=1, day=1)
    end = pd.Timestamp(config.end_date) if config.end_date else all_dates.max()
    signal_dates = list(all_dates.loc[(all_dates >= start) & (all_dates <= end)])
    rebalance_dates_by_period = {
        period: select_rebalance_dates(signal_dates, period)
        for period in config.rebalance_periods
    }
    unique_rebalance_dates = sorted(
        {date for dates in rebalance_dates_by_period.values() for date in dates}
    )
    return_dates = pd.Index(pd.to_datetime(sorted(returns["Date"].dropna().unique())))
    returns_by_date = {
        date: group.drop_duplicates("PERMNO").set_index("PERMNO")["AdjReturn"].astype(float)
        for date, group in returns.groupby("Date")
    }

    signal_cache: dict[pd.Timestamp, SignalResult] = {}
    audit_rows: list[dict[str, object]] = []
    holding_rows: list[pd.DataFrame] = []
    for index, signal_date in enumerate(unique_rebalance_dates, start=1):
        print(f"signal {index}/{len(unique_rebalance_dates)} {signal_date.date()}", flush=True)
        result = build_signal(
            config,
            signal_date=signal_date,
            returns=returns,
            factors=factors,
            identifier_history=identifier_history,
            sp500_source=sp500_source,
        )
        signal_cache[signal_date] = result
        audit_rows.append(result.audit)
        if not result.holdings.empty:
            holding_rows.append(result.holdings)

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(output_dir / "signal_audit.csv", index=False)
    if holding_rows:
        pd.concat(holding_rows, ignore_index=True).to_csv(
            output_dir / "selected_holdings.csv", index=False
        )

    summary_frames: list[pd.DataFrame] = []
    yearly_frames: list[pd.DataFrame] = []
    for period, rebalance_dates in rebalance_dates_by_period.items():
        period_dir = output_dir / period
        period_dir.mkdir(parents=True, exist_ok=True)
        raw = build_raw_returns_for_period(
            config,
            period=period,
            rebalance_dates=rebalance_dates,
            signal_cache=signal_cache,
            returns_by_date=returns_by_date,
            return_dates=return_dates,
        )
        raw.to_csv(period_dir / "raw_returns.csv", index=False)
        gated, thresholds = apply_expanding_gate(
            raw,
            config.gate,
            oos_start_year=config.oos_start_year,
            oos_end_year=config.oos_end_year,
        )
        thresholds.to_csv(period_dir / "gate_thresholds.csv", index=False)
        if gated.empty:
            attached = gated
        else:
            attached = attach_optional_series(gated, benchmark, risk_free)
        attached.to_csv(period_dir / "returns.csv", index=False)
        frequency = period_frequency(period)
        perf = performance_summary(
            attached,
            rebalance_period=period,
            frequency=frequency,
            benchmark_name="SPY",
        )
        year = yearly_returns(attached, rebalance_period=period, benchmark_name="SPY")
        perf.to_csv(period_dir / "performance_summary.csv", index=False)
        year.to_csv(period_dir / "yearly_returns.csv", index=False)
        summary_frames.append(perf)
        yearly_frames.append(year)

    summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    yearly = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    comparison = compare_with_broad_universe(summary)
    summary.to_csv(output_dir / "summary.csv", index=False)
    yearly.to_csv(output_dir / "yearly_returns.csv", index=False)
    comparison.to_csv(output_dir / "comparison_against_broad_universe.csv", index=False)
    write_run_summary(
        config,
        output_dir,
        source_info=source_metadata(config.sp500_source, sp500_source),
        rebalance_dates_by_period=rebalance_dates_by_period,
        signal_audit=audit,
    )
    write_report(output_dir, config, summary, yearly, comparison, audit)
    return BacktestOutputs(
        summary=summary,
        yearly=yearly,
        comparison=comparison,
        output_dir=output_dir,
    )


def build_signal(
    config: StrategyConfig,
    *,
    signal_date: pd.Timestamp,
    returns: pd.DataFrame,
    factors: list[pd.DataFrame],
    identifier_history: pd.DataFrame,
    sp500_source: pd.DataFrame,
) -> SignalResult:
    snapshot = snapshot_as_of(sp500_source, signal_date)
    mapped, mapped_frame = map_snapshot_to_permnos(
        snapshot,
        identifier_history,
        max_identifier_staleness_days=config.mapping.max_identifier_staleness_days,
    )
    base_audit = mapped_universe_audit_row(mapped)
    base_audit.update(
        {
            "SignalDate": signal_date,
            "Status": "started",
            "FeatureEligibleCount": 0,
            "SelectedCount": 0,
            "TopScore": np.nan,
            "MedianWorst60M": np.nan,
            "MaxFactorDateUsed": pd.NaT,
            "MaxDataAvailableDateUsed": pd.NaT,
            "FutureAvailableRowsExcluded": 0,
        }
    )
    if mapped.mapped_permno_count < config.top_n:
        return empty_signal(signal_date, mapped.source_snapshot_date, mapped, base_audit, "insufficient_mapped_universe")

    features = trailing_horizon_returns(
        returns,
        signal_date,
        list(mapped.permnos),
        windows=config.windows,
        min_history_months=config.min_history_months,
        include_signal_date=config.include_signal_month_return,
    )
    if config.require_current_return:
        current_permnos = pd.Index(
            returns.loc[returns["Date"].eq(signal_date), "PERMNO"].dropna().astype(int).unique()
        )
        features = features.loc[features.index.intersection(current_permnos)]
    if len(features) < max(config.n_clusters, config.min_cluster_size, config.top_n):
        base_audit.update({"FeatureEligibleCount": int(len(features))})
        return empty_signal(signal_date, mapped.source_snapshot_date, mapped, base_audit, "insufficient_return_history")

    clusters = cluster_returns(
        features,
        n_clusters=config.n_clusters,
        nearest_neighbors=config.nearest_neighbors,
        random_state=config.random_state,
        positive_only=config.positive_only_affinity,
    )
    factor_scores = select_factor_scores_as_of(
        factors,
        signal_date,
        features.index,
        [
            column
            for column in config.score_weights
            if not column.startswith("ret_") and not column.startswith("cluster_")
        ],
    )
    score_inputs = augment_cluster_features(
        augment_return_features(features),
        clusters,
        residual_columns=("ret_11m",),
    ).join(factor_scores, how="left")
    try:
        scores = fixed_rule_score(score_inputs, config.score_weights)
    except ValueError as exc:
        base_audit.update({"FeatureEligibleCount": int(len(score_inputs)), "StatusDetail": str(exc)})
        return empty_signal(signal_date, mapped.source_snapshot_date, mapped, base_audit, "missing_score_inputs")
    selected = scores.dropna().sort_values(ascending=False).head(config.top_n)
    if len(selected) < config.top_n:
        base_audit.update({"FeatureEligibleCount": int(len(score_inputs))})
        return empty_signal(signal_date, mapped.source_snapshot_date, mapped, base_audit, "insufficient_scores")

    availability = factor_availability_audit(factors, signal_date, features.index)
    median_worst = metric_median(score_inputs, "ret_horizon_worst_60m")
    holdings = (
        selected.rename("Score")
        .to_frame()
        .reset_index()
        .rename(columns={"index": "PERMNO"})
    )
    holdings.loc[:, "SignalDate"] = signal_date
    holdings.loc[:, "Weight"] = 1.0 / config.top_n
    if not mapped_frame.empty:
        holdings = holdings.merge(
            mapped_frame[
                [
                    "SourceSymbol",
                    "PERMNO",
                    "MatchedFeedSymbol",
                    "MatchedYFTicker",
                    "MatchedSecurity",
                    "IdentifierDate",
                ]
            ],
            on="PERMNO",
            how="left",
        )
    base_audit.update(
        {
            "Status": "ok",
            "FeatureEligibleCount": int(len(score_inputs)),
            "SelectedCount": int(len(selected)),
            "TopScore": float(selected.max()),
            "MedianWorst60M": median_worst,
            **availability,
        }
    )
    return SignalResult(
        signal_date=signal_date,
        source_snapshot_date=mapped.source_snapshot_date,
        selected=selected,
        score_inputs=score_inputs,
        mapped_permnos=mapped.permnos,
        raw_universe_count=mapped.source_symbol_count,
        mapped_universe_count=mapped.mapped_permno_count,
        status="ok",
        audit=base_audit,
        holdings=holdings,
    )


def empty_signal(
    signal_date: pd.Timestamp,
    source_snapshot_date: pd.Timestamp | pd.NaT,
    mapped,
    audit: dict[str, object],
    status: str,
) -> SignalResult:
    audit = dict(audit)
    audit["Status"] = status
    return SignalResult(
        signal_date=signal_date,
        source_snapshot_date=source_snapshot_date,
        selected=pd.Series(dtype=float, name="Score"),
        score_inputs=pd.DataFrame(),
        mapped_permnos=mapped.permnos,
        raw_universe_count=mapped.source_symbol_count,
        mapped_universe_count=mapped.mapped_permno_count,
        status=status,
        audit=audit,
        holdings=pd.DataFrame(),
    )


def build_raw_returns_for_period(
    config: StrategyConfig,
    *,
    period: str,
    rebalance_dates: list[pd.Timestamp],
    signal_cache: dict[pd.Timestamp, SignalResult],
    returns_by_date: dict[pd.Timestamp, pd.Series],
    return_dates: pd.Index,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, signal_date in enumerate(rebalance_dates):
        signal = signal_cache.get(signal_date)
        if signal is None or not signal.valid:
            continue
        next_date = rebalance_dates[index + 1] if index + 1 < len(rebalance_dates) else None
        future_dates = holding_period_dates(
            pd.DataFrame({"Date": return_dates}),
            signal_date,
            next_date,
        )
        if future_dates.empty:
            continue
        selected = signal.selected
        for return_date in future_dates:
            date_returns = returns_by_date.get(return_date)
            if date_returns is None:
                selected_returns = pd.Series(0.0, index=selected.index)
            else:
                selected_returns = date_returns.reindex(selected.index)
                if config.missing_returns_as_cash:
                    selected_returns = selected_returns.fillna(0.0)
                else:
                    selected_returns = selected_returns.dropna()
            if selected_returns.empty:
                continue
            rows.append(
                {
                    "RebalancePeriod": period,
                    "Frequency": period_frequency(period),
                    "SignalDate": signal_date,
                    "SourceSnapshotDate": signal.source_snapshot_date,
                    "Date": return_date,
                    "RawPortfolioReturn": float(selected_returns.sum() / config.top_n),
                    "RawActiveNames": config.top_n,
                    "RawUniverseCount": signal.raw_universe_count,
                    "MappedUniverseCount": signal.mapped_universe_count,
                    "FeatureEligibleCount": int(len(signal.score_inputs)),
                    "top_score": float(selected.max()),
                    "avg_score": float(selected.mean()),
                    "tenth_score": float(selected.min()),
                    "spread_score": float(selected.max() - selected.min()),
                    "median_ret_horizon_worst_60m": metric_median(
                        signal.score_inputs, "ret_horizon_worst_60m"
                    ),
                }
            )
    return pd.DataFrame(rows)


def metric_median(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return np.nan
    return float(frame[column].median(skipna=True))


def compare_with_broad_universe(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if summary.empty:
        return pd.DataFrame()
    metrics = [
        "CAGR",
        "BenchmarkCAGR",
        "SimpleAlphaAnnualized",
        "CAPMAlphaAnnualized",
        "CAPMAlphaRiskFreeAnnualized",
        "Beta",
        "MaxDrawdown",
        "Sharpe",
        "InformationRatio",
        "ActiveShare",
    ]
    for _, row in summary.iterrows():
        period = str(row["RebalancePeriod"])
        broad = BROAD_UNIVERSE_RESULTS.get(period, {})
        out = {"RebalancePeriod": period}
        for metric in metrics:
            sp_value = row.get(metric, np.nan)
            broad_value = broad.get(metric, np.nan)
            out[f"SP500_{metric}"] = sp_value
            out[f"BroadWRDS_{metric}"] = broad_value
            out[f"Difference_{metric}"] = (
                sp_value - broad_value
                if not pd.isna(sp_value) and not pd.isna(broad_value)
                else np.nan
            )
        rows.append(out)
    return pd.DataFrame(rows)


def write_run_summary(
    config: StrategyConfig,
    output_dir: Path,
    *,
    source_info: dict[str, object],
    rebalance_dates_by_period: dict[str, list[pd.Timestamp]],
    signal_audit: pd.DataFrame,
) -> None:
    payload = {
        "method": "current_fixed_top10_spectral_sp500_pit",
        "top_n": config.top_n,
        "portfolio": "long-only, unlevered, equal-weight, no shorts, no leverage",
        "score_weights": config.score_weights,
        "gate": config.gate.__dict__,
        "start_year": config.start_year,
        "oos_start_year": config.oos_start_year,
        "oos_end_year": config.oos_end_year,
        "end_date": config.end_date,
        "rebalance_periods": {
            period: {
                "count": len(dates),
                "first": str(dates[0].date()) if dates else None,
                "last": str(dates[-1].date()) if dates else None,
            }
            for period, dates in rebalance_dates_by_period.items()
        },
        "sp500_source": source_info,
        "identifier_bridge": {
            "source": "WRDS CRSP CIZ monthly return panel audit identifiers",
            "max_identifier_staleness_days": config.mapping.max_identifier_staleness_days,
            "native_permon_keyed_sp500_artifact_present_in_pit_repo": False,
        },
        "signal_status_counts": signal_audit["Status"].value_counts().to_dict()
        if "Status" in signal_audit
        else {},
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )


def write_report(
    output_dir: Path,
    config: StrategyConfig,
    summary: pd.DataFrame,
    yearly: pd.DataFrame,
    comparison: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    lines = [
        "# Current Fixed Top-10 Spectral Strategy on PIT S&P 500",
        "",
        "## Setup",
        "",
        "- Universe: pinned `fja05680/sp500` PIT constituent snapshots, mapped to WRDS PERMNOs using CRSP CIZ historical audit tickers as of each signal date.",
        "- Portfolio: long-only, unlevered, equal-weight top 10.",
        "- Rebalance periods: 3M, 6M, 1Y.",
        "- Score: `-3.0 * EarningsYield + 1.0 * ValueScore - 1.0 * ret_horizon_vol_60m + 2.0 * cluster_resid_ret_11m`, with each input cross-sectionally z-scored.",
        "- Gate: invest only when selected `top_score` is below the prior-year expanding 85th percentile and universe median `ret_horizon_worst_60m` is below the prior-year expanding 80th percentile.",
        "- OOS window: 2001-2025, when data supports the signal and forward-return window.",
        "",
        "## Summary",
        "",
        markdown_table(
            summary,
            [
                "RebalancePeriod",
                "ActiveShare",
                "CAGR",
                "BenchmarkCAGR",
                "SimpleAlphaAnnualized",
                "CAPMAlphaAnnualized",
                "CAPMAlphaRiskFreeAnnualized",
                "Beta",
                "MaxDrawdown",
                "Sharpe",
                "InformationRatio",
            ],
        ),
        "",
        "## Broad Universe Comparison",
        "",
        markdown_table(
            comparison,
            [
                "RebalancePeriod",
                "SP500_CAGR",
                "BroadWRDS_CAGR",
                "Difference_CAGR",
                "SP500_SimpleAlphaAnnualized",
                "BroadWRDS_SimpleAlphaAnnualized",
                "Difference_SimpleAlphaAnnualized",
                "SP500_CAPMAlphaAnnualized",
                "BroadWRDS_CAPMAlphaAnnualized",
                "Difference_CAPMAlphaAnnualized",
                "SP500_MaxDrawdown",
                "BroadWRDS_MaxDrawdown",
            ],
        ),
        "",
        "## Screener / Forward Test",
        "",
        "For a new signal date, load the latest source snapshot with `date <= signal date`, map source tickers to WRDS PERMNOs using only identifier rows dated at or before the signal, compute trailing return features from returns strictly before the signal, select PIT factor rows with `Date <= signal date` and `DataAvailableDate <= signal date`, score the cross-section, and apply the saved expanding gate thresholds trained through the prior completed year.",
        "",
        "The selected names for historical signals are in `selected_holdings.csv`. `gate_thresholds.csv` in each rebalance-period directory records the annual expanding thresholds used for each OOS year.",
        "",
        "## Look-Ahead Assessment",
        "",
        "- S&P 500 membership never uses a source row after the signal date.",
        "- The ticker-to-PERMNO bridge uses CRSP identifier rows observed on or before the signal date, with a bounded staleness window.",
        "- Factor data is selected with both `Date <= signal date` and `DataAvailableDate <= signal date` when availability is present.",
        "- Trailing return features use returns before the signal date because `include_signal_month_return` is false.",
        "- Gate thresholds use only signal metrics from years prior to the OOS return year.",
        "",
        "## Native Artifact Gap",
        "",
        "The WRDS PIT database repo did not contain a native PERMNO-keyed PIT S&P 500 membership artifact. This run uses the pinned constituent source plus a PIT identifier bridge; `docs/wrds_pit_sp500_artifact_request.md` describes the requested database artifact.",
        "",
        "## Output Files",
        "",
        "- `summary.csv`",
        "- `yearly_returns.csv`",
        "- `comparison_against_broad_universe.csv`",
        "- `signal_audit.csv`",
        "- `selected_holdings.csv`",
        "- `3M/`, `6M/`, `1Y/` period-specific returns, thresholds, and performance files",
        "",
    ]
    if not audit.empty:
        lines.extend(
            [
                "## Mapping Coverage",
                "",
                markdown_table(
                    audit[
                        [
                            "SignalDate",
                            "SourceSymbolCount",
                            "MappedPermnoCount",
                            "UnmappedCount",
                            "AmbiguousCount",
                            "Status",
                        ]
                    ].tail(12),
                    [
                        "SignalDate",
                        "SourceSymbolCount",
                        "MappedPermnoCount",
                        "UnmappedCount",
                        "AmbiguousCount",
                        "Status",
                    ],
                ),
                "",
            ]
        )
    output_dir.joinpath("report.md").write_text("\n".join(lines), encoding="utf-8")


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame is None or frame.empty:
        return "_No rows._"
    present = [column for column in columns if column in frame.columns]
    if not present:
        return "_No requested columns._"
    out = frame[present].copy()
    percent_columns = {
        column
        for column in out.columns
        if any(token in column for token in ("CAGR", "Alpha", "Drawdown", "Return", "ActiveShare"))
    }
    float_columns = {"Beta", "Sharpe", "InformationRatio"}
    for column in out.columns:
        if column in percent_columns:
            out.loc[:, column] = out[column].map(format_percent)
        elif column in float_columns:
            out.loc[:, column] = out[column].map(format_float)
        else:
            out.loc[:, column] = out[column].map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(out.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(out.columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in out.astype(str).to_numpy()]
    return "\n".join([header, separator, *body])


def format_percent(value: object) -> str:
    return "" if pd.isna(value) else f"{float(value):.2%}"


def format_float(value: object) -> str:
    return "" if pd.isna(value) else f"{float(value):.2f}"
