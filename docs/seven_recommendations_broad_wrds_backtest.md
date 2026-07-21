# Seven-Recommendations Broad WRDS Backtest

Run date: 2026-07-21.

This run used the private local WRDS paths with the broad WRDS CRSP CIZ PIT
universe, SIC group controls, constrained candidates, no cash gate, equal
weights, and a top-N sweep over 10, 20, 30, and 50 names.

## Training Search

Selection window: 2001-2020.

Command shape:

```powershell
python scripts\tune_sp500_parameters.py `
  --config configs\local.yaml `
  --output-dir outputs\seven_recommendations_train_2001_2020 `
  --selection-objective robust_alpha `
  --candidate-set constrained `
  --feature-set core `
  --top-n-values 10,20,30,50 `
  --gate-set no_gate `
  --periods 6M,1Y `
  --random-candidates 800 `
  --oos-start-year 2001 `
  --oos-end-year 2020 `
  --end-date 2020-12-31
```

The raw best training candidate was rejected as a sparse-signal mirage:

- Candidate: `single_neg_ret_horizon_best_48m`
- Rebalance: 1Y
- Top N: 10
- Months: 22
- Training CAGR: 119.04%
- Training simple alpha: 97.21%

After requiring real coverage with `Months >= 180` and `ActiveShare >= 0.90`,
the selected candidate was:

- Candidate: `random_0746`
- Weights: `ValueScore: -2.0`, `ret_39m: -2.0`
- Rebalance: 6M
- Top N: 10
- Training months: 194
- Training CAGR: 57.90%
- Training SPY CAGR: 7.92%
- Training simple alpha: 49.99%
- Training max drawdown: -67.36%
- Worst excess year: -20.50%

## 2021-2025 Reserve Backtest

The filtered training winner did not validate out of sample.

| Metric | Result |
| --- | ---: |
| Strategy CAGR | -29.93% |
| SPY CAGR | 14.83% |
| Simple alpha | -44.76% |
| CAPM alpha | -28.07% |
| RF CAPM alpha | -26.18% |
| Beta | 0.60 |
| Max drawdown | -84.01% |
| Sharpe | -0.48 |
| Information ratio | -0.79 |

## Read

The broad WRDS universe does not rescue this search as currently specified. It
finds spectacular training-period anti-momentum/value combinations, but the
reserve result is decisively negative. The first raw winner also exposed an
objective-design bug: sparse valid-signal coverage can still dominate unless
sample coverage is a hard filter rather than a small penalty.

After this run, `robust_alpha` was updated to apply a materially larger
coverage penalty below 180 months. Re-ranking the completed training metrics
with the fixed objective keeps `random_0746` as the top candidate, so the
2021-2025 reserve result above remains the relevant holdout test.
