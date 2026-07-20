from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wrds_spectral_sp500_strategy.backtest import metric_median
from wrds_spectral_sp500_strategy.clustering import cluster_returns
from wrds_spectral_sp500_strategy.config import GateConfig, StrategyConfig
from wrds_spectral_sp500_strategy.data import (
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
    trailing_horizon_returns,
    zscore_columns,
)
from wrds_spectral_sp500_strategy.gates import apply_expanding_gate
from wrds_spectral_sp500_strategy.performance import attach_optional_series, performance_summary
from wrds_spectral_sp500_strategy.rebalance import (
    holding_period_dates,
    period_frequency,
    select_rebalance_dates,
)
from wrds_spectral_sp500_strategy.sp500 import (
    load_configured_source,
    map_snapshot_to_permnos,
    snapshot_as_of,
)


FUNDAMENTAL_COLUMNS: tuple[str, ...] = (
    "ValueScore",
    "EarningsYield",
    "FCFYield",
    "CFOYield",
    "EBIT_EV",
    "EBITDAYield",
    "SalesYield",
    "BookToMarket",
    "ShareholderYield",
    "QualityScore",
    "ROIC",
    "ROE",
    "ROA",
    "GrossProfitToAssets",
    "OperatingProfitability",
    "GrossMargin",
    "OperatingMargin",
    "NetMargin",
    "AccrualsToAssets",
    "PiotroskiFScore",
    "AssetTurnover",
    "GrowthScore",
    "RevenueGrowthYoY",
    "EBITDAGrowthYoY",
    "EarningsGrowthYoY",
    "FCFGrowthYoY",
    "BookEquityGrowthYoY",
    "RevenueCAGR3Y",
    "EBITDACAGR3Y",
    "FCFCAGR3Y",
    "AssetGrowth",
    "AssetGrowthYoY",
    "BalanceSheetScore",
    "NetDebtToEBITDA",
    "DebtToAssets",
    "DebtToEquity",
    "InterestCoverage",
    "CashToAssets",
    "CurrentRatio",
    "QuickRatio",
    "CapitalAllocationScore",
    "CapexToSales",
    "CapexToAssets",
    "RAndDToSales",
    "SGAToSales",
    "BuybackYield",
    "NetIssuance",
    "NetIssuanceYield",
    "DividendPayoutRatio",
    "MomentumScore",
    "Momentum12_1",
    "Momentum6_1",
    "Momentum3_1",
    "RiskScore",
    "Volatility12M",
    "MaxDrawdown12M",
    "LiquidityScore",
    "LogMarketCap",
    "DollarVolumeRank",
    "Turnover",
    "AmihudIlliquidity",
)

CORE_FUNDAMENTAL_COLUMNS: tuple[str, ...] = (
    "ValueScore",
    "EarningsYield",
    "FCFYield",
    "CFOYield",
    "EBIT_EV",
    "BookToMarket",
    "ShareholderYield",
    "QualityScore",
    "ROIC",
    "AccrualsToAssets",
    "GrowthScore",
    "RevenueCAGR3Y",
    "FCFCAGR3Y",
    "BalanceSheetScore",
    "DebtToAssets",
    "CashToAssets",
    "CapitalAllocationScore",
    "BuybackYield",
    "NetIssuanceYield",
    "MomentumScore",
    "Momentum12_1",
    "Momentum6_1",
    "RiskScore",
    "Volatility12M",
    "MaxDrawdown12M",
)

CLUSTER_RESIDUAL_WINDOWS: tuple[int, ...] = (
    1,
    2,
    3,
    4,
    5,
    6,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    18,
    24,
    30,
    36,
    48,
    59,
    60,
)


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    weights: dict[str, float]


@dataclass(frozen=True)
class GatePolicy:
    name: str
    gate: GateConfig | None


@dataclass
class SignalSlice:
    signal_date: pd.Timestamp
    source_snapshot_date: pd.Timestamp
    scores: pd.DataFrame
    median_worst_60m: float
    raw_universe_count: int
    mapped_universe_count: int
    feature_eligible_count: int


def main() -> int:
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config = StrategyConfig.from_yaml(args.config)
    if args.top_n is not None:
        config = replace(config, top_n=args.top_n)

    returns = load_returns(config.returns_path)
    benchmark = load_benchmark(config.benchmark_path)
    risk_free = load_risk_free(config.risk_free_path)
    returns_by_date = {
        date: group.drop_duplicates("PERMNO").set_index("PERMNO")["AdjReturn"].astype(float)
        for date, group in returns.groupby("Date")
    }
    return_dates = pd.Index(pd.to_datetime(sorted(returns["Date"].dropna().unique())))

    candidates = build_candidates(
        args.random_candidates,
        seed=args.seed,
        include_cluster_residuals=args.include_cluster_residuals,
        feature_set=args.feature_set,
    )
    if args.top_from_metrics is not None:
        keep_names = candidate_allowlist(args.top_from_metrics, args.top_count)
        candidates = [candidate for candidate in candidates if candidate.name in keep_names]
    factor_columns = sorted(
        {
            column
            for candidate in candidates
            for column in candidate.weights
            if not column.startswith("ret_") and not column.startswith("cluster_")
        }
    )
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
    requested_periods = tuple(period.strip().upper() for period in args.periods.split(","))
    rebalance_dates_by_period = {
        period: dates
        for period, dates in rebalance_dates_by_period.items()
        if period.upper() in requested_periods
    }
    unique_rebalance_dates = sorted(
        {date for dates in rebalance_dates_by_period.values() for date in dates}
    )
    print(
        f"building PIT S&P signal cache: {len(unique_rebalance_dates)} signal dates, "
        f"{len(candidates)} candidates",
        flush=True,
    )
    signal_cache = build_signal_cache(
        config,
        unique_rebalance_dates,
        returns=returns,
        factors=factors,
        identifier_history=identifier_history,
        sp500_source=sp500_source,
        include_cluster_residuals=args.include_cluster_residuals,
        factor_columns=factor_columns,
    )
    available_candidates = [
        candidate
        for candidate in candidates
        if all_columns_available(candidate, signal_cache)
    ]
    print(
        f"evaluating {len(available_candidates)} candidates after column availability checks",
        flush=True,
    )
    policies = build_gate_policies(args.gate_set)
    metrics = run_screen(
        config,
        available_candidates,
        policies,
        rebalance_dates_by_period,
        signal_cache,
        returns_by_date=returns_by_date,
        return_dates=return_dates,
        benchmark=benchmark,
        risk_free=risk_free,
    )
    metrics = metrics.sort_values(
        ["SimpleAlphaAnnualized", "CAGR", "Sharpe"],
        ascending=[False, False, False],
    )
    metrics.to_csv(output_dir / "candidate_metrics.csv", index=False)
    write_candidate_catalog(available_candidates, output_dir)
    if metrics.empty:
        raise RuntimeError("No tunable candidate produced returns.")

    best = metrics.iloc[0].to_dict()
    best_candidate = next(
        candidate for candidate in available_candidates if candidate.name == best["Candidate"]
    )
    best_policy = next(policy for policy in policies if policy.name == best["GatePolicy"])
    tuned_config = write_tuned_config(
        config,
        best_candidate,
        best_policy,
        str(best["RebalancePeriod"]),
        output_dir,
    )
    write_report(output_dir, metrics, best_candidate, best_policy, tuned_config)

    display = metrics[
        [
            "Candidate",
            "Family",
            "GatePolicy",
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
        ]
    ].head(args.show)
    print(display.to_string(index=False))
    print(f"wrote tuning outputs to {output_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="In-sample tune PIT S&P 500 top-10 score parameters."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "local.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "tuned_sp500_alpha_search",
    )
    parser.add_argument("--random-candidates", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--show", type=int, default=12)
    parser.add_argument(
        "--periods",
        default="3M,6M,1Y",
        help="Comma-separated rebalance periods to test.",
    )
    parser.add_argument(
        "--gate-set",
        choices=("no_gate", "focused"),
        default="focused",
    )
    parser.add_argument(
        "--top-from-metrics",
        type=Path,
        default=None,
        help="Optional prior candidate_metrics.csv used to keep only top candidates.",
    )
    parser.add_argument("--top-count", type=int, default=200)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument(
        "--feature-set",
        choices=("return", "core", "all"),
        default="core",
        help="Candidate input breadth. 'return' skips WRDS factor loading.",
    )
    parser.add_argument(
        "--include-cluster-residuals",
        action="store_true",
        help="Include slower spectral cluster residual return features in the search.",
    )
    return parser.parse_args()


def candidate_allowlist(metrics_path: Path, top_count: int) -> set[str]:
    metrics = pd.read_csv(metrics_path)
    if metrics.empty or "Candidate" not in metrics:
        return set()
    sort_columns = [
        column
        for column in ("SimpleAlphaAnnualized", "CAGR", "Sharpe")
        if column in metrics
    ]
    if sort_columns:
        metrics = metrics.sort_values(sort_columns, ascending=[False] * len(sort_columns))
    return set(metrics["Candidate"].head(top_count).astype(str))


def build_signal_cache(
    config: StrategyConfig,
    signal_dates: list[pd.Timestamp],
    *,
    returns: pd.DataFrame,
    factors: list[pd.DataFrame],
    identifier_history: pd.DataFrame,
    sp500_source: pd.DataFrame,
    include_cluster_residuals: bool,
    factor_columns: list[str],
) -> dict[pd.Timestamp, SignalSlice]:
    cache: dict[pd.Timestamp, SignalSlice] = {}
    residual_columns = (
        tuple(f"ret_{window}m" for window in CLUSTER_RESIDUAL_WINDOWS)
        if include_cluster_residuals
        else ()
    )
    for index, signal_date in enumerate(signal_dates, start=1):
        if index == 1 or index % 10 == 0 or index == len(signal_dates):
            print(f"  signal {index}/{len(signal_dates)} {signal_date.date()}", flush=True)
        snapshot = snapshot_as_of(sp500_source, signal_date)
        mapped, _mapped_frame = map_snapshot_to_permnos(
            snapshot,
            identifier_history,
            max_identifier_staleness_days=config.mapping.max_identifier_staleness_days,
        )
        if mapped.mapped_permno_count < config.top_n:
            continue
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
                returns.loc[returns["Date"].eq(signal_date), "PERMNO"]
                .dropna()
                .astype(int)
                .unique()
            )
            features = features.loc[features.index.intersection(current_permnos)]
        if len(features) < max(config.n_clusters, config.min_cluster_size, config.top_n):
            continue
        clusters = (
            cluster_returns(
                features,
                n_clusters=config.n_clusters,
                nearest_neighbors=config.nearest_neighbors,
                random_state=config.random_state,
                positive_only=config.positive_only_affinity,
            )
            if residual_columns
            else pd.Series(index=features.index, dtype="float64", name="Cluster")
        )
        factor_scores = select_factor_scores_as_of(
            factors,
            signal_date,
            features.index,
            factor_columns,
        )
        inputs = augment_cluster_features(
            augment_return_features(features),
            clusters,
            residual_columns=residual_columns,
        ).join(factor_scores, how="left")
        cache[signal_date] = SignalSlice(
            signal_date=signal_date,
            source_snapshot_date=mapped.source_snapshot_date,
            scores=zscore_columns(inputs),
            median_worst_60m=metric_median(inputs, "ret_horizon_worst_60m"),
            raw_universe_count=mapped.source_symbol_count,
            mapped_universe_count=mapped.mapped_permno_count,
            feature_eligible_count=len(inputs),
        )
    return cache


def build_candidates(
    random_candidates: int,
    *,
    seed: int,
    include_cluster_residuals: bool,
    feature_set: str,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    if feature_set != "return":
        candidates.append(
            Candidate(
            "current_fixed_rule",
            "baseline",
            {
                "EarningsYield": -3.0,
                "ValueScore": 1.0,
                "ret_horizon_vol_60m": -1.0,
                "cluster_resid_ret_11m": 2.0,
            },
            )
        )
    if feature_set != "return":
        candidates.extend(domain_composites())
    fundamentals = {
        "return": (),
        "core": CORE_FUNDAMENTAL_COLUMNS,
        "all": FUNDAMENTAL_COLUMNS,
    }[feature_set]
    columns = list(dict.fromkeys((*fundamentals, *return_columns(include_cluster_residuals))))
    for column in columns:
        candidates.append(Candidate(f"single_pos_{column}", "single_feature", {column: 1.0}))
        candidates.append(Candidate(f"single_neg_{column}", "single_feature", {column: -1.0}))

    rng = random.Random(seed)
    signed_pool = [(column, sign) for column in columns for sign in (-1.0, 1.0)]
    for index in range(random_candidates):
        size = rng.choice((2, 3, 4, 5, 6))
        picks = rng.sample(signed_pool, size)
        weights: dict[str, float] = {}
        for column, sign in picks:
            magnitude = rng.choice((0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0))
            weights[column] = weights.get(column, 0.0) + sign * magnitude
        weights = {column: weight for column, weight in weights.items() if weight != 0.0}
        if weights:
            candidates.append(
                Candidate(f"random_{index:04d}", "random_sparse_combo", weights)
            )
    return dedupe_candidates(candidates)


def return_columns(include_cluster_residuals: bool = True) -> tuple[str, ...]:
    raw = [f"ret_{window}m" for window in range(1, 61)]
    horizon_features: list[str] = []
    for horizon in (3, 6, 9, 12, 18, 24, 36, 48, 60):
        horizon_features.extend(
            [
                f"ret_horizon_mean_{horizon}m",
                f"ret_horizon_vol_{horizon}m",
                f"ret_horizon_hit_{horizon}m",
                f"ret_horizon_best_{horizon}m",
                f"ret_horizon_worst_{horizon}m",
                f"ret_efficiency_{horizon}m",
            ]
        )
        if horizon > 1:
            horizon_features.append(f"ret_skip1_{horizon}m")
    spreads = [
        "ret_spread_1_3m",
        "ret_spread_3_12m",
        "ret_spread_4_6m",
        "ret_spread_6_24m",
        "ret_spread_12_36m",
        "ret_spread_24_60m",
    ]
    residuals = (
        [f"cluster_resid_ret_{window}m" for window in CLUSTER_RESIDUAL_WINDOWS]
        if include_cluster_residuals
        else []
    )
    return tuple([*raw, *horizon_features, *spreads, *residuals])


def domain_composites() -> list[Candidate]:
    return [
        Candidate(
            "quality_value_low_accrual",
            "domain_composite",
            {
                "QualityScore": 4.0,
                "ValueScore": 3.0,
                "ROIC": 2.0,
                "GrossProfitToAssets": 2.0,
                "OperatingProfitability": 2.0,
                "FCFYield": 2.0,
                "AccrualsToAssets": -3.0,
                "DebtToAssets": -1.5,
                "Momentum12_1": 1.0,
                "Volatility12M": -1.0,
            },
        ),
        Candidate(
            "shareholder_yield_quality",
            "domain_composite",
            {
                "ShareholderYield": 4.0,
                "BuybackYield": 3.0,
                "NetIssuanceYield": -3.0,
                "FCFYield": 3.0,
                "CFOYield": 2.0,
                "ROIC": 2.0,
                "AccrualsToAssets": -2.0,
                "Momentum6_1": 1.0,
            },
        ),
        Candidate(
            "defensive_quality_momentum",
            "domain_composite",
            {
                "QualityScore": 4.0,
                "BalanceSheetScore": 3.0,
                "RiskScore": -4.0,
                "Volatility12M": -3.0,
                "MaxDrawdown12M": 3.0,
                "DebtToAssets": -2.0,
                "InterestCoverage": 2.0,
                "Momentum12_1": 2.0,
                "ret_horizon_worst_60m": 2.0,
            },
        ),
        Candidate(
            "growth_at_reasonable_value",
            "domain_composite",
            {
                "RevenueCAGR3Y": 2.0,
                "EBITDACAGR3Y": 2.0,
                "FCFCAGR3Y": 2.0,
                "GrowthScore": 2.0,
                "FCFYield": 2.0,
                "EBIT_EV": 2.0,
                "ROIC": 2.0,
                "AssetGrowth": -1.5,
                "NetIssuanceYield": -1.5,
                "Momentum12_1": 1.0,
            },
        ),
        Candidate(
            "return_residual_quality",
            "domain_composite",
            {
                "cluster_resid_ret_11m": 3.0,
                "ret_skip1_12m": 2.0,
                "ret_horizon_worst_60m": 3.0,
                "ret_horizon_vol_36m": -2.0,
                "QualityScore": 3.0,
                "ValueScore": 2.0,
                "AccrualsToAssets": -2.0,
            },
        ),
        Candidate(
            "yearly_broad_fixed_rule",
            "legacy_broad_params",
            {
                "ret_horizon_vol_60m": 5.0,
                "ret_8m": -8.0,
                "ret_horizon_vol_9m": 8.0,
                "ret_spread_4_6m": 1.0,
            },
        ),
        Candidate(
            "quarterly_broad_fixed_rule",
            "legacy_broad_params",
            {
                "ret_12m": 21.0,
                "ret_14m": -8.0,
                "ret_30m": -21.0,
                "ret_skip1_3m": 2.0,
                "ret_3m": -5.0,
                "ret_horizon_worst_48m": 5.0,
                "ret_skip1_24m": -5.0,
                "ret_spread_1_3m": -2.0,
                "ret_spread_12_36m": -1.0,
                "RiskScore": -0.5,
                "ret_59m": 0.5,
                "ret_skip1_60m": -0.5,
            },
        ),
    ]


def build_gate_policies(gate_set: str = "focused") -> list[GatePolicy]:
    policies = [GatePolicy("no_gate", None)]
    if gate_set == "no_gate":
        return policies
    specs = [
        ("<", 0.85, "<", 0.80),
        ("<", 0.80, "<", 0.80),
        ("<", 0.90, "<", 0.80),
        ("<", 0.95, "<", 0.80),
        ("<", 0.85, "<", 0.65),
        ("<", 0.85, "<", 0.50),
        (">", 0.20, ">", 0.20),
        (">", 0.35, ">", 0.20),
        (">", 0.50, ">", 0.20),
        (">", 0.20, ">", 0.35),
        (">", 0.20, ">", 0.50),
        (">", 0.65, ">", 0.35),
        ("<", 0.80, ">", 0.20),
        ("<", 0.90, ">", 0.20),
        (">", 0.20, "<", 0.80),
        (">", 0.35, "<", 0.80),
    ]
    for top_op, top_q, worst_op, worst_q in specs:
        policies.append(
            GatePolicy(
                f"top_score_{top_op}_q{top_q:.2f}_median_worst_{worst_op}_q{worst_q:.2f}",
                GateConfig(
                    train_start_year=1996,
                    top_score_quantile=top_q,
                    median_worst_quantile=worst_q,
                    top_score_operator=top_op,
                    median_worst_operator=worst_op,
                ),
            )
        )
    return policies


def run_screen(
    config: StrategyConfig,
    candidates: list[Candidate],
    policies: list[GatePolicy],
    rebalance_dates_by_period: dict[str, list[pd.Timestamp]],
    signal_cache: dict[pd.Timestamp, SignalSlice],
    *,
    returns_by_date: dict[pd.Timestamp, pd.Series],
    return_dates: pd.Index,
    benchmark: pd.DataFrame,
    risk_free: pd.DataFrame | None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for index, candidate in enumerate(candidates, start=1):
        if index == 1 or index % 100 == 0 or index == len(candidates):
            print(f"  candidate {index}/{len(candidates)} {candidate.name}", flush=True)
        for period, rebalance_dates in rebalance_dates_by_period.items():
            raw = build_raw_returns_for_candidate(
                config,
                candidate,
                period,
                rebalance_dates,
                signal_cache,
                returns_by_date=returns_by_date,
                return_dates=return_dates,
            )
            if raw.empty:
                continue
            for policy in policies:
                gated = apply_policy(
                    raw,
                    policy,
                    oos_start_year=config.oos_start_year,
                    oos_end_year=config.oos_end_year,
                )
                if gated.empty:
                    continue
                attached = attach_optional_series(gated, benchmark, risk_free)
                perf = performance_summary(
                    attached,
                    rebalance_period=period,
                    frequency=period_frequency(period),
                    benchmark_name="SPY",
                )
                if perf.empty:
                    continue
                perf.insert(0, "GatePolicy", policy.name)
                perf.insert(0, "Family", candidate.family)
                perf.insert(0, "Candidate", candidate.name)
                rows.append(perf)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_raw_returns_for_candidate(
    config: StrategyConfig,
    candidate: Candidate,
    period: str,
    rebalance_dates: list[pd.Timestamp],
    signal_cache: dict[pd.Timestamp, SignalSlice],
    *,
    returns_by_date: dict[pd.Timestamp, pd.Series],
    return_dates: pd.Index,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, signal_date in enumerate(rebalance_dates):
        signal = signal_cache.get(signal_date)
        if signal is None:
            continue
        selected = score_candidate(signal.scores, candidate.weights, config.top_n)
        if selected.empty:
            continue
        next_date = rebalance_dates[index + 1] if index + 1 < len(rebalance_dates) else None
        future_dates = holding_period_dates(
            pd.DataFrame({"Date": return_dates}),
            signal_date,
            next_date,
        )
        for return_date in future_dates:
            date_returns = returns_by_date.get(return_date)
            if date_returns is None:
                selected_returns = pd.Series(0.0, index=selected.index)
            else:
                selected_returns = date_returns.reindex(selected.index)
                selected_returns = (
                    selected_returns.fillna(0.0)
                    if config.missing_returns_as_cash
                    else selected_returns.dropna()
                )
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
                    "FeatureEligibleCount": signal.feature_eligible_count,
                    "top_score": float(selected.max()),
                    "avg_score": float(selected.mean()),
                    "tenth_score": float(selected.min()),
                    "spread_score": float(selected.max() - selected.min()),
                    "median_ret_horizon_worst_60m": signal.median_worst_60m,
                }
            )
    return pd.DataFrame(rows)


def score_candidate(scores: pd.DataFrame, weights: dict[str, float], top_n: int) -> pd.Series:
    missing = [column for column in weights if column not in scores.columns]
    if missing:
        return pd.Series(dtype=float)
    composite = pd.Series(0.0, index=scores.index, name="Score")
    for column, weight in weights.items():
        composite = composite.add(scores[column].astype(float) * float(weight), fill_value=0.0)
    selected = composite.dropna().sort_values(ascending=False).head(top_n)
    return selected if len(selected) == top_n else pd.Series(dtype=float)


def apply_policy(
    raw: pd.DataFrame,
    policy: GatePolicy,
    *,
    oos_start_year: int,
    oos_end_year: int,
) -> pd.DataFrame:
    if policy.gate is not None:
        gated, _thresholds = apply_expanding_gate(
            raw,
            policy.gate,
            oos_start_year=oos_start_year,
            oos_end_year=oos_end_year,
        )
        return gated
    frame = raw.copy()
    frame.loc[:, "Date"] = pd.to_datetime(frame["Date"])
    frame = frame.loc[frame["Date"].dt.year.between(oos_start_year, oos_end_year)].copy()
    if frame.empty:
        return frame
    frame.loc[:, "GatePassed"] = True
    frame.loc[:, "PortfolioReturn"] = frame["RawPortfolioReturn"]
    frame.loc[:, "GrossExposure"] = 1.0
    frame.loc[:, "NetExposure"] = 1.0
    frame.loc[:, "ActiveNames"] = frame["RawActiveNames"]
    return frame


def all_columns_available(
    candidate: Candidate,
    signal_cache: dict[pd.Timestamp, SignalSlice],
) -> bool:
    if not signal_cache:
        return False
    available = set().union(*(slice_.scores.columns for slice_ in signal_cache.values()))
    return all(column in available for column in candidate.weights)


def write_candidate_catalog(candidates: list[Candidate], output_dir: Path) -> None:
    payload = [candidate.__dict__ for candidate in candidates]
    (output_dir / "candidate_catalog.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def write_tuned_config(
    config: StrategyConfig,
    candidate: Candidate,
    policy: GatePolicy,
    period: str,
    output_dir: Path,
) -> Path:
    raw = yaml.safe_load((ROOT / "configs" / "local.example.yaml").read_text(encoding="utf-8"))
    raw["rebalance_periods"] = [period]
    raw["top_n"] = config.top_n
    raw["score_weights"] = candidate.weights
    raw["output_dir"] = "../outputs/tuned_sp500_alpha_gt15"
    if policy.gate is None:
        raw["gate"] = {
            "enabled": False,
            "train_start_year": 1996,
            "top_score_quantile": 0.85,
            "median_worst_quantile": 0.80,
            "top_score_operator": "<",
            "median_worst_operator": "<",
        }
    else:
        raw["gate"] = policy.gate.__dict__
    tuned_path = output_dir / "tuned_config.example.yaml"
    tuned_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return tuned_path


def write_report(
    output_dir: Path,
    metrics: pd.DataFrame,
    candidate: Candidate,
    policy: GatePolicy,
    tuned_config: Path,
) -> None:
    best = metrics.iloc[0]
    lines = [
        "# PIT S&P 500 Parameter Tuning",
        "",
        "This search intentionally optimized against the full 2001-2025 backtest window.",
        "Treat the winning specification as in-sample tuned, not validated alpha.",
        "",
        "## Best Candidate",
        "",
        f"- Candidate: `{candidate.name}`",
        f"- Family: `{candidate.family}`",
        f"- Gate policy: `{policy.name}`",
        f"- Rebalance period: `{best['RebalancePeriod']}`",
        f"- CAGR: {float(best['CAGR']):.2%}",
        f"- SPY CAGR: {float(best['BenchmarkCAGR']):.2%}",
        f"- Simple alpha: {float(best['SimpleAlphaAnnualized']):.2%}",
        f"- CAPM alpha: {float(best['CAPMAlphaAnnualized']):.2%}",
        f"- RF CAPM alpha: {float(best['CAPMAlphaRiskFreeAnnualized']):.2%}",
        f"- Beta: {float(best['Beta']):.2f}",
        f"- Max drawdown: {float(best['MaxDrawdown']):.2%}",
        f"- Active share: {float(best['ActiveShare']):.2%}",
        "",
        "## Score Weights",
        "",
        "```yaml",
        yaml.safe_dump(candidate.weights, sort_keys=False).strip(),
        "```",
        "",
        "## Outputs",
        "",
        f"- `{tuned_config.name}`",
        "- `candidate_metrics.csv`",
        "- `candidate_catalog.json`",
        "",
    ]
    output_dir.joinpath("tuning_report.md").write_text("\n".join(lines), encoding="utf-8")


def dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    seen: set[tuple[tuple[str, float], ...]] = set()
    out: list[Candidate] = []
    for candidate in candidates:
        key = tuple(sorted((column, round(weight, 8)) for column, weight in candidate.weights.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
