# WRDS Spectral S&P 500 Strategy

This repo backtests the current fixed top-10 spectral strategy inside a
point-in-time S&P 500 constituent universe.

It is intentionally data-free. WRDS/CRSP/Compustat files stay local or private;
the code points at existing local exports.

## Strategy

- Long-only and unlevered.
- Top 10 stocks.
- Default weighting is equal-weight; configs can opt into rank-decay weighting
  while still holding all 10 selected names.
- Rebalance periods: 3M, 6M, 1Y.
- Benchmark: SPY monthly adjusted returns.
- Universe: PIT S&P 500 membership from pinned `fja05680/sp500` snapshots.
- Identifier bridge: WRDS CRSP CIZ historical tickers mapped to PERMNO as of
  each signal date.

Fixed score, descending after cross-sectional z-scoring:

```text
-3.0 * EarningsYield
+1.0 * ValueScore
-1.0 * ret_horizon_vol_60m
+2.0 * cluster_resid_ret_11m
```

Cash gate:

```text
top_score < prior expanding 85th percentile
median(ret_horizon_worst_60m) < prior expanding 80th percentile
```

The expanding thresholds use only prior years.

## Reproduce

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[test]"
Copy-Item configs\local.example.yaml configs\local.yaml
.\.venv\Scripts\python scripts\run_current_fixed_top10_sp500.py --config configs\local.yaml
```

Outputs are written under `outputs/current_fixed_top10_sp500/`.

## Walk-Forward Tuned Search

`scripts/tune_sp500_parameters.py` searches PIT-safe score parameters and writes
diagnostics under `outputs/tuned_sp500_alpha_search/`.

The checked-in tuned example is `configs/tuned_alpha_gt15.example.yaml`. The
latest rerun selected parameters on 2001-2020 only, reserving 2021-2025 for a
five-year walk-forward test. It is not the original equal-weight fixed top-10
strategy: it keeps top 10 selected names, uses yearly rebalancing, applies
rank-decay portfolio weights with `rank_decay: 0.50`, and scores names as:

```text
-2.0 * ret_34m
-1.0 * ret_32m
-1.5 * BookEquityGrowthYoY
-8.0 * ret_horizon_mean_36m
-3.0 * ret_horizon_hit_9m
```

The selected gate invests only when `top_score < q0.95` and
`median_worst < q0.80` using prior-year expanding thresholds. On the 2001-2020
selection window, this specification produced 29.43% CAGR versus 8.59% for SPY,
or 20.84% simple annualized alpha. On the untouched 2021-2025 reserve window,
it produced -4.12% CAGR versus 14.83% for SPY, or -18.95% simple alpha.

`configs/walk_forward_5y_holdout.example.yaml` reproduces the 2021-2025 reserve
evaluation. See `docs/walk_forward_2021_2025_selection.md` for the split,
search passes, selected strategy, and reserve result. The five-year reserve did
not validate the tuned alpha.

## PIT Safety

- S&P 500 source snapshots are selected with `source date <= signal date`.
- WRDS ticker/PERMNO identifier rows are selected with identifier `Date <= signal date`.
- Factor rows require `Date <= signal date`.
- Factor rows also require `DataAvailableDate <= signal date` when that column exists.
- Trailing return features use monthly returns before the signal date.
- Forward returns start after the signal date and end at the next rebalance date.
- Gate thresholds are expanding and trained only on prior years.

## Artifact Gap

The WRDS PIT database repo does not yet contain a native PERMNO-keyed PIT S&P
500 constituent artifact. This repo can run from the pinned public constituent
source plus the WRDS identifier bridge, but `docs/wrds_pit_sp500_artifact_request.md`
describes the cleaner database artifact to build next.
