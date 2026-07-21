# 2025 Walk-Forward Reserve Selection

This rerun reserves calendar year 2025 for walk-forward testing. Strategy
parameters were selected only on monthly returns dated 2001-01-31 through
2024-12-31. The reserve test uses returns dated 2025-01-31 through 2025-12-31,
driven by the annual signal formed on 2024-12-31.

## Selection Window

Search settings:

- Universe: PIT S&P 500 from pinned `fja05680/sp500`.
- Portfolio: long-only, unlevered top 10.
- Rebalance: yearly.
- Feature set: all configured WRDS factor files plus trailing return features.
- Candidate set: single features, domain composites, and 2,000 seeded random sparse combos.
- Gate set: no gate, then a focused expanding-gate pass on the top 100 rank-decay candidates.
- Selection metric: `SimpleAlphaAnnualized` over 2001-2024.

Best training candidates:

| Search | Selected Candidate | Weighting | Gate | Train CAGR | SPY CAGR | Train Simple Alpha |
| --- | --- | --- | --- | --- | --- | --- |
| Rank decay 0.50 | `earnings_yield_ret13` | rank decay 0.50 | none | 27.34% | 9.49% | 17.84% |
| Rank decay 0.60 | `earnings_yield_ret13` | rank decay 0.60 | none | 25.78% | 9.49% | 16.29% |
| Equal weight | `single_pos_EBIT_EV` | equal | none | 20.90% | 9.49% | 11.41% |
| Focused gates on top 100 | `earnings_yield_ret13` | rank decay 0.50 | none | 27.34% | 9.49% | 17.84% |

Selected strategy:

```yaml
rebalance_periods: ["1Y"]
top_n: 10
portfolio_weighting: rank_decay
rank_decay: 0.5
score_weights:
  EarningsYield: -0.5
  ret_13m: 1.0
gate:
  enabled: false
```

## Reserve Result

The selected strategy was then evaluated once on the untouched 2025 reserve
window:

| Reserve Year | Strategy Return | SPY Return | Simple Alpha | CAPM Alpha | RF CAPM Alpha | Beta | Max Drawdown | Sharpe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025 | 24.48% | 17.72% | 6.76% | -8.18% | 9.01% | 2.04 | -15.27% | 0.86 |

The reserve year is positive and beats SPY on a simple return basis, but it does
not reproduce the 15%+ annualized simple alpha seen in the 2001-2024 tuning
window. Treat the selected strategy as still data-mined until it survives more
unseen periods or a rolling walk-forward protocol.
