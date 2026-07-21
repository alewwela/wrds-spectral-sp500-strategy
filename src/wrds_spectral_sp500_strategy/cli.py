from __future__ import annotations

import argparse
from pathlib import Path

from wrds_spectral_sp500_strategy.backtest import run_backtests
from wrds_spectral_sp500_strategy.config import StrategyConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the fixed spectral strategy on a PIT S&P 500 or broad WRDS universe."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/local.yaml"),
        help="YAML config path. Copy configs/local.example.yaml to start.",
    )
    args = parser.parse_args(argv)
    config = StrategyConfig.from_yaml(args.config)
    outputs = run_backtests(config)
    print(f"wrote outputs to {outputs.output_dir}")
    if not outputs.summary.empty:
        display = outputs.summary[
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
            ]
        ]
        print(display.to_string(index=False))
    return 0
