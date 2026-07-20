# WRDS Spectral S&P 500 Strategy

This repo backtests the current fixed top-10 spectral strategy inside a
point-in-time S&P 500 constituent universe.

It is intentionally data-free. WRDS/CRSP/Compustat files stay local or private;
the code points at existing local exports.

## Strategy

- Long-only, unlevered, equal-weight.
- Top 10 stocks.
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

## In-Sample Tuned Alpha Search

`scripts/tune_sp500_parameters.py` searches PIT-safe score parameters and writes
diagnostics under `outputs/tuned_sp500_alpha_search/`.

The checked-in tuned example is `configs/tuned_alpha_gt15.example.yaml`. It is
not the original fixed top-10 strategy: it changes concentration to top 3,
uses yearly rebalancing, disables the cash gate, and scores names as:

```text
-0.5 * EarningsYield
+1.0 * ret_13m
```

On the local 2001-2025 PIT S&P 500 backtest, that in-sample tuned
specification produced 27.25% CAGR versus 9.82% for SPY, or 17.43% simple
annualized alpha. Treat this as data-mined research output until it survives a
separate validation protocol.

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
