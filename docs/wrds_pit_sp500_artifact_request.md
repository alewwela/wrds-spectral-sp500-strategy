# WRDS PIT S&P 500 Constituent Artifact Request

Please add a native PIT S&P 500 constituent layer to
`wrds-pit-market-cap-universe`.

## Source

- Primary source: `fja05680/sp500`
- Pinned commit: `c31ac3cc56f28cf9a02b4e694eff7ceab596a0ff`
- File: `S&P 500 Historical Components & Changes (Updated).csv`
- Source schema: `date,tickers`
- Source caveat: research-grade free source, not official S&P DJI licensed feed.

## Requested Output

Create chunked CSV/zip files plus a manifest:

```text
datasets/wrds_sp500_constituents_1996_1999.zip
datasets/wrds_sp500_constituents_2000_2004.zip
...
datasets/wrds_sp500_constituents_chunks.json
```

Required columns:

```text
AsOfDate
SourceSnapshotDate
SourceSymbol
PERMNO
PERMCO
FeedSymbol
YFTicker
Security
Exchange
IdentifierDate
MappingStatus
MappingNote
SourceRepo
SourceVersion
SourceFile
```

Rules:

- For each WRDS monthly `AsOfDate`, use the latest S&P 500 source row whose
  `date <= AsOfDate`.
- Map source tickers to CRSP PERMNOs using only WRDS identifier rows with
  `IdentifierDate <= AsOfDate`.
- Preserve class-share tickers such as `BRK.B`, `BF.B`, `RDS.A`, and similar
  using CRSP security-class text when WRDS audit tickers collapse the dot.
- Do not use current index membership, current ticker lists, current names, or
  current sector metadata to backfill historical rows.
- Emit unmapped and ambiguous rows to an audit file rather than silently adding
  or dropping them.
- Keep delisted and renamed constituents mapped to their historical PERMNO when
  WRDS can resolve them as of the historical date.

Recommended audit files:

```text
datasets/wrds_sp500_constituent_mapping_audit.csv
datasets/wrds_sp500_constituent_date_counts.csv
```

Audit columns should include date-level source constituent count, mapped PERMNO
count, unmapped count, ambiguous count, and samples of unresolved source symbols.

