# 2021-2025 Walk-Forward Reserve Selection

This rerun reserves the final five calendar years, 2021-2025, for walk-forward
testing. Strategy parameters were selected only on monthly returns dated
2001-01-31 through 2020-12-31. The reserve test uses returns dated
2021-02-26 through 2025-12-31, driven by annual signals formed from
2020-12-31 onward.

## Selection Window

Search settings:

- Universe: PIT S&P 500 from pinned `fja05680/sp500`.
- Portfolio: long-only, unlevered top 10.
- Rebalance: yearly.
- Feature set: all configured WRDS factor files plus trailing return features.
- Candidate set: single features, domain composites, and 2,000 seeded random sparse combos.
- Selection metric: `SimpleAlphaAnnualized` over 2001-2020.
- Reserve: no 2021-2025 returns were used during selection.

Best training candidates:

| Search | Selected Candidate | Weighting | Gate | Train CAGR | SPY CAGR | Train Simple Alpha |
| --- | --- | --- | --- | --- | --- | --- |
| Rank decay 0.50 | `random_1765` | rank decay 0.50 | none | 25.90% | 8.59% | 17.30% |
| Rank decay 0.60 | `random_1765` | rank decay 0.60 | none | 24.65% | 8.59% | 16.05% |
| Equal weight | `single_pos_EBIT_EV` | equal | none | 22.00% | 8.59% | 13.41% |
| Focused gates on top 100 | `random_0997` | rank decay 0.50 | `top_score < q0.95` and `median_worst < q0.80` | 29.43% | 8.59% | 20.84% |

Selected strategy:

```yaml
rebalance_periods: ["1Y"]
top_n: 10
portfolio_weighting: rank_decay
rank_decay: 0.5
score_weights:
  ret_34m: -2.0
  ret_32m: -1.0
  BookEquityGrowthYoY: -1.5
  ret_horizon_mean_36m: -8.0
  ret_horizon_hit_9m: -3.0
gate:
  enabled: true
  train_start_year: 1996
  top_score_quantile: 0.95
  median_worst_quantile: 0.80
  top_score_operator: "<"
  median_worst_operator: "<"
```

## Reserve Result

The selected training winner was then evaluated once on the untouched
2021-2025 reserve window:

| Reserve Window | Strategy CAGR | SPY CAGR | Simple Alpha | CAPM Alpha | RF CAPM Alpha | Beta | Max Drawdown | Sharpe | Active Share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2021-2025 | -4.12% | 14.83% | -18.95% | -13.55% | -12.67% | 0.89 | -41.57% | -0.04 | 61.02% |

Year-by-year reserve returns:

| Year | Strategy Return | SPY Return | Excess Return | Active Months |
| --- | --- | --- | --- | --- |
| 2021 | 0.00% | 30.05% | -30.05% | 0 |
| 2022 | -7.90% | -18.18% | 10.28% | 12 |
| 2023 | -10.60% | 26.18% | -36.78% | 12 |
| 2024 | -1.24% | 24.89% | -26.13% | 12 |
| 2025 | 0.00% | 17.72% | -17.72% | 0 |

As a diagnostic, the best no-gate training candidate, `random_1765`, also failed
to beat SPY in reserve: 8.18% CAGR versus 14.83% for SPY, or -6.66% simple
annualized alpha.

The five-year reserve does not validate the tuned alpha. It shows that the
training-selected gated strategy was overfit to the 2001-2020 window.
