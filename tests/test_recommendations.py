from __future__ import annotations

from pathlib import Path

import pandas as pd

from wrds_spectral_sp500_strategy.config import SectorControlConfig, StrategyConfig
from wrds_spectral_sp500_strategy.objectives import sort_by_selection_score
from wrds_spectral_sp500_strategy.selection import build_group_labels, select_scores
from wrds_spectral_sp500_strategy.universe import broad_wrds_universe_as_of
from wrds_spectral_sp500_strategy.walk_forward import rolling_year_splits


def test_broad_wrds_universe_uses_prior_current_identifier_rows_only() -> None:
    identifiers = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2024-01-31", "2024-02-29", "2024-04-30", "2023-01-31"]
            ),
            "PERMNO": [101, 101, 202, 303],
            "FeedSymbol": ["AAA", "AAB", "BBB", "CCC"],
            "YFTicker": ["AAA", "AAB", "BBB", "CCC"],
            "Security": ["A", "A new", "B future", "C stale"],
            "SIC": ["3571", "3571", "2834", "6021"],
            "MarketCap": [1.0, 2.0, 3.0, 4.0],
            "Exchange": ["N", "N", "Q", "N"],
        }
    )

    mapped, rows = broad_wrds_universe_as_of(
        identifiers,
        pd.Timestamp("2024-03-31"),
        max_identifier_staleness_days=95,
    )

    assert mapped.permnos == (101,)
    assert rows.set_index("PERMNO").loc[101, "SourceSymbol"] == "AAB"
    assert "SIC" in rows.columns


def test_sector_control_caps_selected_names_per_sic_group() -> None:
    scores = pd.Series([5.0, 4.0, 3.0, 2.0], index=[101, 202, 303, 404], name="Score")
    mapped = pd.DataFrame(
        {
            "PERMNO": [101, 202, 303, 404],
            "SIC": ["3571", "3579", "2834", "2836"],
        }
    )
    control = SectorControlConfig(enabled=True, column="SIC", max_per_group=1)
    groups = build_group_labels(scores.index, control, mapped_frame=mapped)

    selected = select_scores(scores, 2, control, group_labels=groups)

    assert selected.index.tolist() == [101, 303]


def test_robust_objective_can_rank_stable_alpha_above_max_alpha() -> None:
    metrics = pd.DataFrame(
        {
            "Candidate": ["fragile", "stable"],
            "SimpleAlphaAnnualized": [0.20, 0.12],
            "CAGR": [0.25, 0.16],
            "Sharpe": [0.1, 0.9],
            "InformationRatio": [0.1, 0.8],
            "MaxDrawdown": [-0.90, -0.20],
            "WorstExcessYear": [-0.80, -0.05],
            "ExcessYearWinRate": [0.20, 0.80],
            "ActiveShare": [0.30, 0.95],
            "Months": [120, 120],
        }
    )

    ranked = sort_by_selection_score(metrics, "robust_alpha")

    assert ranked.iloc[0]["Candidate"] == "stable"


def test_robust_objective_penalizes_sparse_month_coverage() -> None:
    metrics = pd.DataFrame(
        {
            "Candidate": ["sparse", "covered"],
            "SimpleAlphaAnnualized": [0.95, 0.30],
            "CAGR": [1.10, 0.40],
            "Sharpe": [1.4, 0.9],
            "InformationRatio": [1.2, 0.8],
            "MaxDrawdown": [-0.25, -0.35],
            "WorstExcessYear": [0.05, -0.10],
            "ExcessYearWinRate": [1.00, 0.75],
            "ActiveShare": [1.00, 1.00],
            "Months": [22, 194],
        }
    )

    ranked = sort_by_selection_score(metrics, "robust_alpha")

    assert ranked.iloc[0]["Candidate"] == "covered"


def test_rolling_year_splits_reserve_future_years() -> None:
    splits = rolling_year_splits(
        start_year=2001,
        end_year=2006,
        train_years=3,
        test_years=1,
        step_years=1,
    )

    assert [(s.train_start_year, s.train_end_year, s.test_start_year, s.test_end_year) for s in splits] == [
        (2001, 2003, 2004, 2004),
        (2002, 2004, 2005, 2005),
        (2003, 2005, 2006, 2006),
    ]


def test_strategy_config_parses_universe_and_sector_controls() -> None:
    config = StrategyConfig.from_mapping(
        {
            "pit_universe_repo": "../data/repo",
            "returns_path": "../data/returns.csv",
            "benchmark_path": "../data/spy.csv",
            "universe_mode": "broad_wrds",
            "sector_control": {
                "enabled": True,
                "column": "SIC",
                "max_per_group": 2,
                "min_groups": 5,
                "neutralize_scores": True,
            },
        },
        base_dir=Path("configs"),
    )

    assert config.universe_mode == "broad_wrds"
    assert config.sector_control.max_per_group == 2
    assert config.sector_control.neutralize_scores
