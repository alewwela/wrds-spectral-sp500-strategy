from __future__ import annotations

import pandas as pd

from wrds_spectral_sp500_strategy.sp500 import (
    map_snapshot_to_permnos,
    snapshot_as_of,
)


def test_sp500_snapshot_uses_latest_prior_source_row_only() -> None:
    source = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-05"]),
            "tickers": ["AAA,BBB", "AAA,CCC"],
        }
    )

    before_change = snapshot_as_of(source, pd.Timestamp("2024-01-04"))
    on_change = snapshot_as_of(source, pd.Timestamp("2024-01-05"))

    assert before_change.source_symbols == ("AAA", "BBB")
    assert on_change.source_symbols == ("AAA", "CCC")


def test_class_share_mapping_uses_security_class() -> None:
    source = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-31"]),
            "tickers": ["BRK.B,BF.B"],
        }
    )
    identifier_history = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-31"] * 4),
            "PERMNO": [1, 2, 3, 4],
            "FeedSymbol": ["BRK", "BRK", "BF", "BF"],
            "YFTicker": ["BRK", "BRK", "BF", "BF"],
            "Security": [
                "BERKSHIRE HATHAWAY INC; COM A; CONS",
                "BERKSHIRE HATHAWAY INC; COM B; CONS",
                "BROWN FORMAN CORP; COM A; CONS",
                "BROWN FORMAN CORP; COM B; CONS",
            ],
        }
    )

    mapped, rows = map_snapshot_to_permnos(
        snapshot_as_of(source, pd.Timestamp("2024-01-31")),
        identifier_history,
    )

    assert mapped.permnos == (2, 4)
    assert rows.set_index("SourceSymbol")["PERMNO"].to_dict() == {"BRK.B": 2, "BF.B": 4}

