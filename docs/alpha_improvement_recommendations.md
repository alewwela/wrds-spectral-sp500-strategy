# Alpha Improvement Implementation

This repo now implements the seven follow-up recommendations as runnable
research controls rather than as narrative caveats.

## 1. Broader Universe

Set `universe_mode: broad_wrds` to select from the PIT WRDS CRSP CIZ identifier
panel instead of the pinned public S&P 500 snapshots. The broad universe uses
only identifier rows with `Date <= signal date` and the configured staleness
window.

## 2. Stop Selecting On Max Alpha Alone

`scripts/tune_sp500_parameters.py` now ranks candidates with
`--selection-objective`. The default `robust_alpha` includes simple alpha but
penalizes weak active share, large drawdowns, poor worst-year excess returns,
and short samples. The old behavior is still available with
`--selection-objective simple_alpha`.

## 3. Rolling Walk-Forward

Pass `--rolling-train-years`, `--rolling-test-years`, and
`--rolling-step-years` to repeatedly select on a training fold and evaluate the
selected strategy on the next reserve fold. Outputs:

- `rolling_train_candidate_metrics.csv`
- `rolling_selected_folds.csv`
- `rolling_strategy_summary.csv`
- `rolling_walk_forward_report.md`

## 4. Constrained Simple Models

The tuner default is now `--candidate-set constrained`, which keeps random
candidate formulas to two or three inputs with small weights. Use
`--candidate-set all` to restore the broader high-degree random search.

## 5. Relax Fixed Top 10

Use `--top-n-values 10,20,30,50` to evaluate multiple selected-name counts in
one run. The chosen `TopN` is written into `candidate_metrics.csv` and any tuned
config generated from the winner.

## 6. Sector Controls

The `sector_control` YAML block can cap selected names per group, require a
minimum number of groups, and rank group-neutralized scores. With the current
WRDS identifier panel, `column: SIC` uses configurable leading SIC digits as a
sector proxy. If no sector column is present, `bucket_column` can form quantile
buckets from a PIT numeric factor such as `LogMarketCap`.

## 7. Separate Objective Reporting

Tuning outputs now include `SelectionObjective`, `SelectionScore`,
`ExcessYearWinRate`, `MedianExcessYear`, and `WorstExcessYear`, making the
chosen strategy auditable against the objective that selected it.

## Suggested Robust Search

```powershell
.\.venv\Scripts\python scripts\tune_sp500_parameters.py `
  --config configs\seven_recommendations_search.example.yaml `
  --output-dir outputs\seven_recommendations_search `
  --selection-objective robust_alpha `
  --candidate-set constrained `
  --feature-set core `
  --top-n-values 10,20,30,50 `
  --portfolio-weighting equal `
  --gate-set no_gate `
  --periods 6M,1Y `
  --random-candidates 800
```

## Suggested Rolling Walk-Forward

```powershell
.\.venv\Scripts\python scripts\tune_sp500_parameters.py `
  --config configs\seven_recommendations_search.example.yaml `
  --output-dir outputs\seven_recommendations_rolling `
  --selection-objective robust_alpha `
  --candidate-set constrained `
  --feature-set core `
  --top-n-values 10,20,30,50 `
  --gate-set no_gate `
  --periods 6M,1Y `
  --random-candidates 800 `
  --rolling-train-years 10 `
  --rolling-test-years 1 `
  --rolling-step-years 1
```
