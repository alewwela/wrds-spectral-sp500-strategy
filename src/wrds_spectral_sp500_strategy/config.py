from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from wrds_spectral_sp500_strategy.sp500 import Sp500SourceConfig


DEFAULT_WINDOWS = tuple(range(1, 61))
DEFAULT_SCORE_WEIGHTS = {
    "EarningsYield": -3.0,
    "ValueScore": 1.0,
    "ret_horizon_vol_60m": -1.0,
    "cluster_resid_ret_11m": 2.0,
}


@dataclass(frozen=True)
class GateConfig:
    train_start_year: int = 1996
    top_score_quantile: float = 0.85
    median_worst_quantile: float = 0.80
    top_score_operator: str = "<"
    median_worst_operator: str = "<"


@dataclass(frozen=True)
class MappingConfig:
    max_identifier_staleness_days: int = 95


@dataclass(frozen=True)
class StrategyConfig:
    pit_universe_repo: Path
    returns_path: Path
    benchmark_path: Path
    risk_free_path: Path | None = None
    factor_paths: tuple[Path, ...] = ()
    output_dir: Path = Path("outputs/current_fixed_top10_sp500")
    start_year: int = 1996
    oos_start_year: int = 2001
    oos_end_year: int = 2025
    end_date: str | None = "2025-12-31"
    rebalance_periods: tuple[str, ...] = ("3M", "6M", "1Y")
    top_n: int = 10
    windows: tuple[int, ...] = DEFAULT_WINDOWS
    min_history_months: int = 60
    include_signal_month_return: bool = False
    require_current_return: bool = True
    missing_returns_as_cash: bool = True
    n_clusters: int = 10
    nearest_neighbors: int | None = 25
    random_state: int = 7
    positive_only_affinity: bool = True
    min_cluster_size: int = 8
    score_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SCORE_WEIGHTS))
    gate: GateConfig = GateConfig()
    sp500_source: Sp500SourceConfig = Sp500SourceConfig()
    mapping: MappingConfig = MappingConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StrategyConfig":
        config_path = Path(path)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        return cls.from_mapping(raw, base_dir=config_path.parent)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], *, base_dir: str | Path | None = None) -> "StrategyConfig":
        base = Path(base_dir or ".")

        def required_path(name: str) -> Path:
            value = raw.get(name)
            if not value:
                raise ValueError(f"Missing required config value: {name}")
            return resolve_path(value, base)

        def optional_path(name: str) -> Path | None:
            value = raw.get(name)
            return None if not value else resolve_path(value, base)

        gate_raw = raw.get("gate", {}) or {}
        source_raw = raw.get("sp500_source", {}) or {}
        mapping_raw = raw.get("mapping", {}) or {}
        factor_paths = tuple(resolve_path(value, base) for value in raw.get("factor_paths", []) or [])
        windows = tuple(int(value) for value in raw.get("windows", DEFAULT_WINDOWS))
        if not windows:
            raise ValueError("windows cannot be empty")
        periods = tuple(str(value).upper() for value in raw.get("rebalance_periods", ("3M", "6M", "1Y")))
        return cls(
            pit_universe_repo=required_path("pit_universe_repo"),
            returns_path=required_path("returns_path"),
            benchmark_path=required_path("benchmark_path"),
            risk_free_path=optional_path("risk_free_path"),
            factor_paths=factor_paths,
            output_dir=resolve_path(raw.get("output_dir", "outputs/current_fixed_top10_sp500"), base),
            start_year=int(raw.get("start_year", 1996)),
            oos_start_year=int(raw.get("oos_start_year", 2001)),
            oos_end_year=int(raw.get("oos_end_year", 2025)),
            end_date=raw.get("end_date", "2025-12-31"),
            rebalance_periods=periods,
            top_n=int(raw.get("top_n", 10)),
            windows=windows,
            min_history_months=int(raw.get("min_history_months", max(windows))),
            include_signal_month_return=bool(raw.get("include_signal_month_return", False)),
            require_current_return=bool(raw.get("require_current_return", True)),
            missing_returns_as_cash=bool(raw.get("missing_returns_as_cash", True)),
            n_clusters=int(raw.get("n_clusters", 10)),
            nearest_neighbors=(
                None if raw.get("nearest_neighbors") is None else int(raw.get("nearest_neighbors"))
            ),
            random_state=int(raw.get("random_state", 7)),
            positive_only_affinity=bool(raw.get("positive_only_affinity", True)),
            min_cluster_size=int(raw.get("min_cluster_size", 8)),
            score_weights={
                str(column): float(weight)
                for column, weight in raw.get("score_weights", DEFAULT_SCORE_WEIGHTS).items()
            },
            gate=GateConfig(
                train_start_year=int(gate_raw.get("train_start_year", 1996)),
                top_score_quantile=float(gate_raw.get("top_score_quantile", 0.85)),
                median_worst_quantile=float(gate_raw.get("median_worst_quantile", 0.80)),
                top_score_operator=str(gate_raw.get("top_score_operator", "<")),
                median_worst_operator=str(gate_raw.get("median_worst_operator", "<")),
            ),
            sp500_source=Sp500SourceConfig(
                repo=str(source_raw.get("repo", Sp500SourceConfig.repo)),
                version=str(source_raw.get("version", Sp500SourceConfig.version)),
                file=str(source_raw.get("file", Sp500SourceConfig.file)),
                url=source_raw.get("url"),
            ),
            mapping=MappingConfig(
                max_identifier_staleness_days=int(
                    mapping_raw.get("max_identifier_staleness_days", 95)
                )
            ),
        )


def resolve_path(value: object, base: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else base / path

