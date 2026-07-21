from __future__ import annotations

import pandas as pd

from wrds_spectral_sp500_strategy.config import GateConditionConfig, GateConfig
from wrds_spectral_sp500_strategy.gates import apply_expanding_gate
from wrds_spectral_sp500_strategy.portfolio import portfolio_weights
from wrds_spectral_sp500_strategy.rebalance import select_rebalance_dates


def test_rebalance_date_selection_for_3m_6m_1y() -> None:
    dates = pd.to_datetime(
        [
            "2024-01-31",
            "2024-02-29",
            "2024-03-28",
            "2024-04-30",
            "2024-06-28",
            "2024-12-31",
        ]
    )

    assert select_rebalance_dates(dates, "3M") == [
        pd.Timestamp("2024-03-28"),
        pd.Timestamp("2024-06-28"),
        pd.Timestamp("2024-12-31"),
    ]
    assert select_rebalance_dates(dates, "6M") == [
        pd.Timestamp("2024-06-28"),
        pd.Timestamp("2024-12-31"),
    ]
    assert select_rebalance_dates(dates, "1Y") == [pd.Timestamp("2024-12-31")]


def test_expanding_gate_uses_prior_years_only() -> None:
    raw = pd.DataFrame(
        {
            "SignalDate": pd.to_datetime(["2000-12-31", "2001-12-31", "2002-12-31"]),
            "Date": pd.to_datetime(["2001-01-31", "2002-01-31", "2003-01-31"]),
            "RawPortfolioReturn": [0.1, 0.1, 0.1],
            "RawActiveNames": [10, 10, 10],
            "top_score": [1.0, 2.0, 100.0],
            "median_ret_horizon_worst_60m": [1.0, 2.0, 100.0],
        }
    )
    gated, thresholds = apply_expanding_gate(
        raw,
        GateConfig(train_start_year=2000, top_score_quantile=0.85, median_worst_quantile=0.80),
        oos_start_year=2001,
        oos_end_year=2003,
    )

    threshold_2003 = thresholds.set_index("OOSYear").loc[2003]

    assert threshold_2003["TopScoreThreshold"] < 100.0
    assert threshold_2003["MedianWorstThreshold"] < 100.0
    assert not bool(gated.loc[gated["Date"].eq(pd.Timestamp("2003-01-31")), "GatePassed"].item())


def test_disabled_gate_stays_fully_invested_for_oos_rows() -> None:
    raw = pd.DataFrame(
        {
            "SignalDate": pd.to_datetime(["2000-12-31", "2001-12-31"]),
            "Date": pd.to_datetime(["2001-01-31", "2002-01-31"]),
            "RawPortfolioReturn": [0.1, -0.2],
            "RawActiveNames": [3, 3],
            "top_score": [1.0, 2.0],
            "median_ret_horizon_worst_60m": [1.0, 2.0],
        }
    )

    gated, thresholds = apply_expanding_gate(
        raw,
        GateConfig(enabled=False),
        oos_start_year=2001,
        oos_end_year=2002,
    )

    assert gated["GatePassed"].tolist() == [True, True]
    assert gated["PortfolioReturn"].tolist() == [0.1, -0.2]
    assert gated["ActiveNames"].tolist() == [3, 3]
    assert thresholds["GateEnabled"].tolist() == [False, False]


def test_generic_gate_conditions_use_prior_year_quantiles() -> None:
    raw = pd.DataFrame(
        {
            "SignalDate": pd.to_datetime(["2000-12-31", "2001-12-31", "2002-12-31"]),
            "Date": pd.to_datetime(["2001-01-31", "2002-01-31", "2003-01-31"]),
            "RawPortfolioReturn": [0.1, 0.2, 0.3],
            "RawActiveNames": [10, 10, 10],
            "top_score": [1.0, 2.0, 100.0],
            "median_ret_horizon_worst_60m": [1.0, 2.0, 100.0],
            "bench_12m": [0.05, 0.10, -0.50],
        }
    )

    gated, thresholds = apply_expanding_gate(
        raw,
        GateConfig(
            train_start_year=2000,
            conditions=(
                GateConditionConfig(metric="bench_12m", operator=">", quantile=0.10),
            ),
        ),
        oos_start_year=2001,
        oos_end_year=2003,
    )

    assert thresholds.set_index("OOSYear").loc[2003, "Condition1Threshold"] > 0
    assert not bool(gated.loc[gated["Date"].eq(pd.Timestamp("2003-01-31")), "GatePassed"].item())


def test_rank_decay_portfolio_weights_keep_all_selected_names() -> None:
    selected = pd.Series([3.0, 2.0, 1.0], index=[101, 202, 303])

    weights = portfolio_weights(selected, weighting="rank_decay", rank_decay=0.5)

    assert weights.index.tolist() == [101, 202, 303]
    assert weights.gt(0.0).all()
    assert weights.sum() == 1.0
    assert weights.loc[101] > weights.loc[202] > weights.loc[303]
