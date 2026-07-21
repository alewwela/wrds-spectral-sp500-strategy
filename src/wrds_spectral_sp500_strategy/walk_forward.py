from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WalkForwardSplit:
    fold: int
    train_start_year: int
    train_end_year: int
    test_start_year: int
    test_end_year: int


def rolling_year_splits(
    *,
    start_year: int,
    end_year: int,
    train_years: int,
    test_years: int = 1,
    step_years: int = 1,
) -> tuple[WalkForwardSplit, ...]:
    if train_years <= 0:
        raise ValueError("train_years must be positive")
    if test_years <= 0:
        raise ValueError("test_years must be positive")
    if step_years <= 0:
        raise ValueError("step_years must be positive")
    splits: list[WalkForwardSplit] = []
    train_start = start_year
    fold = 1
    while True:
        train_end = train_start + train_years - 1
        test_start = train_end + 1
        test_end = test_start + test_years - 1
        if test_end > end_year:
            break
        splits.append(
            WalkForwardSplit(
                fold=fold,
                train_start_year=train_start,
                train_end_year=train_end,
                test_start_year=test_start,
                test_end_year=test_end,
            )
        )
        fold += 1
        train_start += step_years
    return tuple(splits)
