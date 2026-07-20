from __future__ import annotations

import pandas as pd

from wrds_spectral_sp500_strategy.data import select_factor_scores_as_of


def test_select_factor_scores_excludes_future_data_available_date() -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-31", "2024-02-29"]),
            "PERMNO": [10001, 10001],
            "DataAvailableDate": pd.to_datetime(["2024-02-15", "2024-03-15"]),
            "ValueScore": [1.0, 9.0],
        }
    )

    selected = select_factor_scores_as_of(
        [frame],
        pd.Timestamp("2024-02-29"),
        [10001],
        ["ValueScore"],
    )

    assert selected.loc[10001, "ValueScore"] == 1.0

