from __future__ import annotations

import math

import numpy as np
import pandas as pd


PERIODS_PER_YEAR = 12


def attach_optional_series(
    returns: pd.DataFrame,
    benchmark: pd.DataFrame,
    risk_free: pd.DataFrame | None,
) -> pd.DataFrame:
    left = datetime_date_frame(returns)
    bench = datetime_date_frame(benchmark)
    out = left.merge(bench, on="Date", how="left")
    if risk_free is not None:
        rf = datetime_date_frame(risk_free)
        out = out.merge(rf, on="Date", how="left")
    excess = (out["PortfolioReturn"] - out["BenchmarkReturn"]).rename("ExcessReturn")
    return pd.concat([out, excess], axis=1)


def datetime_date_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.drop(columns=["Date"]).copy()
    out.insert(0, "Date", pd.to_datetime(frame["Date"]).to_numpy())
    return out


def performance_summary(
    returns: pd.DataFrame,
    *,
    rebalance_period: str,
    frequency: str,
    benchmark_name: str = "SPY",
) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame()
    group = returns.sort_values("Date")
    strategy = group["PortfolioReturn"].astype(float)
    benchmark = group["BenchmarkReturn"].astype(float)
    paired = pd.concat(
        [strategy.rename("strategy"), benchmark.rename("benchmark")], axis=1
    ).dropna()
    metrics = base_metrics(strategy)
    metrics.update(
        {
            "RebalancePeriod": rebalance_period,
            "RebalanceFrequency": frequency,
            "ActiveShare": float(group["GatePassed"].astype(bool).mean())
            if "GatePassed" in group
            else np.nan,
            "ActiveMonths": int(group.get("GatePassed", pd.Series(False, index=group.index)).astype(bool).sum()),
            "InvestedMonths": int((group.get("GrossExposure", pd.Series(0.0, index=group.index)).astype(float) > 0).sum()),
            "StartDate": group["Date"].min(),
            "EndDate": group["Date"].max(),
            "Months": int(strategy.notna().sum()),
            "BenchmarkName": benchmark_name,
        }
    )
    if paired.empty:
        metrics.update(empty_benchmark_metrics())
    else:
        paired_dates = group.loc[paired.index, "Date"]
        metrics.update(base_metrics(paired["benchmark"], prefix="Benchmark"))
        metrics.update(base_metrics(paired["strategy"], prefix="BenchmarkAlignedStrategy"))
        metrics.update(regression_metrics(paired["strategy"], paired["benchmark"]))
        metrics["BenchmarkAlignedStartDate"] = paired_dates.min()
        metrics["BenchmarkAlignedEndDate"] = paired_dates.max()
        metrics["BenchmarkAlignedMonths"] = int(len(paired))
        metrics["ExcessCAGR"] = (
            metrics["BenchmarkAlignedStrategyCAGR"] - metrics["BenchmarkCAGR"]
        )
        metrics["ExcessCumulativeReturn"] = (
            metrics["BenchmarkAlignedStrategyCumulativeReturn"]
            - metrics["BenchmarkCumulativeReturn"]
        )
        metrics["SimpleAlphaAnnualized"] = metrics["ExcessCAGR"]
        metrics["CAPMAlphaAnnualized"] = metrics["AlphaAnnualized"]
        metrics.update(risk_free_capm_metrics(group))
    ordered = [
        "RebalancePeriod",
        "RebalanceFrequency",
        "ActiveShare",
        "ActiveMonths",
        "InvestedMonths",
        "StartDate",
        "EndDate",
        "Months",
        "CumulativeReturn",
        "CAGR",
        "AnnualizedVolatility",
        "Sharpe",
        "MaxDrawdown",
        "Calmar",
        "WinRate",
        "AverageMonthlyReturn",
        "BestMonth",
        "WorstMonth",
        "BenchmarkName",
        "BenchmarkAlignedStartDate",
        "BenchmarkAlignedEndDate",
        "BenchmarkAlignedMonths",
        "BenchmarkAlignedStrategyCumulativeReturn",
        "BenchmarkAlignedStrategyCAGR",
        "BenchmarkAlignedStrategyAnnualizedVolatility",
        "BenchmarkCumulativeReturn",
        "BenchmarkCAGR",
        "BenchmarkAnnualizedVolatility",
        "BenchmarkMaxDrawdown",
        "ExcessCumulativeReturn",
        "ExcessCAGR",
        "SimpleAlphaAnnualized",
        "AlphaAnnualized",
        "CAPMAlphaAnnualized",
        "CAPMAlphaRiskFreeAnnualized",
        "Beta",
        "CAPMBetaRiskFree",
        "Correlation",
        "TrackingError",
        "InformationRatio",
    ]
    frame = pd.DataFrame([metrics])
    return frame[[column for column in ordered if column in frame.columns]]


def yearly_returns(
    returns: pd.DataFrame,
    *,
    rebalance_period: str,
    benchmark_name: str = "SPY",
) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame()
    frame = returns.copy()
    frame.loc[:, "Year"] = pd.to_datetime(frame["Date"]).dt.year
    rows: list[dict[str, object]] = []
    for year, group in frame.groupby("Year"):
        strategy_return = compound(group["PortfolioReturn"])
        benchmark_return = compound(group["BenchmarkReturn"])
        rows.append(
            {
                "RebalancePeriod": rebalance_period,
                "Year": int(year),
                "StrategyReturn": strategy_return,
                f"{benchmark_name}Return": benchmark_return,
                "ExcessReturn": strategy_return - benchmark_return,
                "Months": int(group["PortfolioReturn"].notna().sum()),
                "ActiveMonths": int(group["GatePassed"].astype(bool).sum())
                if "GatePassed" in group
                else 0,
            }
        )
    return pd.DataFrame(rows).sort_values(["RebalancePeriod", "Year"])


def base_metrics(returns: pd.Series, *, prefix: str = "") -> dict[str, float]:
    clean = returns.dropna().astype(float)
    if clean.empty:
        return {
            metric_name(prefix, "CumulativeReturn"): np.nan,
            metric_name(prefix, "CAGR"): np.nan,
            metric_name(prefix, "AnnualizedVolatility"): np.nan,
            metric_name(prefix, "Sharpe"): np.nan,
            metric_name(prefix, "MaxDrawdown"): np.nan,
            metric_name(prefix, "Calmar"): np.nan,
            metric_name(prefix, "WinRate"): np.nan,
            metric_name(prefix, "AverageMonthlyReturn"): np.nan,
            metric_name(prefix, "BestMonth"): np.nan,
            metric_name(prefix, "WorstMonth"): np.nan,
        }
    cumulative = compound(clean)
    years = len(clean) / PERIODS_PER_YEAR
    cagr = (1.0 + cumulative) ** (1.0 / years) - 1.0 if years > 0 else np.nan
    vol = clean.std(ddof=1) * math.sqrt(PERIODS_PER_YEAR) if len(clean) > 1 else np.nan
    monthly_std = clean.std(ddof=1)
    sharpe = (
        clean.mean() / monthly_std * math.sqrt(PERIODS_PER_YEAR)
        if len(clean) > 1 and monthly_std != 0
        else np.nan
    )
    max_dd = max_drawdown(clean)
    return {
        metric_name(prefix, "CumulativeReturn"): cumulative,
        metric_name(prefix, "CAGR"): cagr,
        metric_name(prefix, "AnnualizedVolatility"): vol,
        metric_name(prefix, "Sharpe"): sharpe,
        metric_name(prefix, "MaxDrawdown"): max_dd,
        metric_name(prefix, "Calmar"): cagr / abs(max_dd) if max_dd < 0 else np.nan,
        metric_name(prefix, "WinRate"): float((clean > 0).mean()),
        metric_name(prefix, "AverageMonthlyReturn"): float(clean.mean()),
        metric_name(prefix, "BestMonth"): float(clean.max()),
        metric_name(prefix, "WorstMonth"): float(clean.min()),
    }


def regression_metrics(strategy: pd.Series, benchmark: pd.Series) -> dict[str, float]:
    paired = pd.concat(
        [strategy.rename("strategy"), benchmark.rename("benchmark")], axis=1
    ).dropna()
    if len(paired) < 2:
        return {
            "AlphaAnnualized": np.nan,
            "Beta": np.nan,
            "Correlation": np.nan,
            "TrackingError": np.nan,
            "InformationRatio": np.nan,
        }
    x = paired["benchmark"].to_numpy(dtype=float)
    y = paired["strategy"].to_numpy(dtype=float)
    variance = float(np.var(x, ddof=1))
    beta = float(np.cov(y, x, ddof=1)[0, 1] / variance) if variance else np.nan
    alpha_monthly = float(np.mean(y) - beta * np.mean(x)) if not pd.isna(beta) else np.nan
    active = paired["strategy"] - paired["benchmark"]
    active_std = active.std(ddof=1)
    return {
        "AlphaAnnualized": (1.0 + alpha_monthly) ** PERIODS_PER_YEAR - 1.0
        if not pd.isna(alpha_monthly)
        else np.nan,
        "Beta": beta,
        "Correlation": float(paired["strategy"].corr(paired["benchmark"])),
        "TrackingError": float(active_std * math.sqrt(PERIODS_PER_YEAR)),
        "InformationRatio": (
            float(active.mean() / active_std * math.sqrt(PERIODS_PER_YEAR))
            if active_std != 0
            else np.nan
        ),
    }


def risk_free_capm_metrics(group: pd.DataFrame) -> dict[str, float]:
    if "RiskFreeReturn" not in group:
        return {"CAPMAlphaRiskFreeAnnualized": np.nan, "CAPMBetaRiskFree": np.nan}
    paired = group[["PortfolioReturn", "BenchmarkReturn", "RiskFreeReturn"]].dropna()
    if len(paired) < 2:
        return {"CAPMAlphaRiskFreeAnnualized": np.nan, "CAPMBetaRiskFree": np.nan}
    strategy_excess = paired["PortfolioReturn"].astype(float) - paired["RiskFreeReturn"].astype(float)
    benchmark_excess = paired["BenchmarkReturn"].astype(float) - paired["RiskFreeReturn"].astype(float)
    variance = float(np.var(benchmark_excess, ddof=1))
    beta = (
        float(np.cov(strategy_excess, benchmark_excess, ddof=1)[0, 1] / variance)
        if variance
        else np.nan
    )
    alpha_monthly = (
        float(strategy_excess.mean() - beta * benchmark_excess.mean())
        if not pd.isna(beta)
        else np.nan
    )
    return {
        "CAPMAlphaRiskFreeAnnualized": (1.0 + alpha_monthly) ** PERIODS_PER_YEAR - 1.0
        if not pd.isna(alpha_monthly)
        else np.nan,
        "CAPMBetaRiskFree": beta,
    }


def empty_benchmark_metrics() -> dict[str, float]:
    return {
        "BenchmarkAlignedStartDate": np.nan,
        "BenchmarkAlignedEndDate": np.nan,
        "BenchmarkAlignedMonths": np.nan,
        "BenchmarkAlignedStrategyCumulativeReturn": np.nan,
        "BenchmarkAlignedStrategyCAGR": np.nan,
        "BenchmarkAlignedStrategyAnnualizedVolatility": np.nan,
        "BenchmarkCumulativeReturn": np.nan,
        "BenchmarkCAGR": np.nan,
        "BenchmarkAnnualizedVolatility": np.nan,
        "BenchmarkMaxDrawdown": np.nan,
        "ExcessCumulativeReturn": np.nan,
        "ExcessCAGR": np.nan,
        "SimpleAlphaAnnualized": np.nan,
        "AlphaAnnualized": np.nan,
        "CAPMAlphaAnnualized": np.nan,
        "CAPMAlphaRiskFreeAnnualized": np.nan,
        "Beta": np.nan,
        "CAPMBetaRiskFree": np.nan,
        "Correlation": np.nan,
        "TrackingError": np.nan,
        "InformationRatio": np.nan,
    }


def compound(returns: pd.Series) -> float:
    clean = returns.dropna().astype(float)
    return float((1.0 + clean).prod() - 1.0) if not clean.empty else np.nan


def max_drawdown(returns: pd.Series) -> float:
    clean = returns.dropna().astype(float)
    if clean.empty:
        return np.nan
    wealth = (1.0 + clean).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def metric_name(prefix: str, name: str) -> str:
    return f"{prefix}{name}" if prefix else name
