from __future__ import annotations

import math

import pandas as pd

from wrds_spectral_sp500_strategy.config import SectorControlConfig


UNKNOWN_GROUP = "UNKNOWN"


def build_group_labels(
    permnos: pd.Index,
    sector_control: SectorControlConfig,
    *,
    mapped_frame: pd.DataFrame | None = None,
    score_inputs: pd.DataFrame | None = None,
) -> pd.Series | None:
    if not sector_control.enabled:
        return None
    index = pd.Index(permnos, name="PERMNO").astype(int)
    if sector_control.bucket_column:
        labels = bucket_group_labels(
            index,
            score_inputs,
            column=sector_control.bucket_column,
            bucket_count=sector_control.bucket_count,
        )
        if labels is not None:
            return labels

    source = group_source_series(
        index,
        sector_control.column,
        mapped_frame=mapped_frame,
        score_inputs=score_inputs,
    )
    if source is None:
        return pd.Series(UNKNOWN_GROUP, index=index, name="Group")
    if sector_control.column.upper() == "SIC":
        labels = source.map(lambda value: sic_group(value, sector_control.sic_digits))
    else:
        labels = source.map(normalize_group_label)
    return labels.reindex(index).fillna(UNKNOWN_GROUP).rename("Group")


def bucket_group_labels(
    index: pd.Index,
    score_inputs: pd.DataFrame | None,
    *,
    column: str,
    bucket_count: int,
) -> pd.Series | None:
    if score_inputs is None or column not in score_inputs.columns:
        return None
    values = pd.to_numeric(score_inputs[column].reindex(index), errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return pd.Series(UNKNOWN_GROUP, index=index, name="Group")
    bucket_count = max(1, int(bucket_count))
    ranks = valid.rank(method="first")
    bucket_ids = pd.qcut(
        ranks,
        q=min(bucket_count, len(valid)),
        labels=False,
        duplicates="drop",
    )
    labels = pd.Series(UNKNOWN_GROUP, index=index, name="Group")
    labels.loc[valid.index] = [
        f"{column}_q{int(bucket) + 1:02d}" if not pd.isna(bucket) else UNKNOWN_GROUP
        for bucket in bucket_ids
    ]
    return labels


def group_source_series(
    index: pd.Index,
    column: str,
    *,
    mapped_frame: pd.DataFrame | None,
    score_inputs: pd.DataFrame | None,
) -> pd.Series | None:
    if mapped_frame is not None and not mapped_frame.empty and column in mapped_frame.columns:
        source = (
            mapped_frame.drop_duplicates("PERMNO", keep="last")
            .set_index("PERMNO")[column]
            .reindex(index)
        )
        return source
    if score_inputs is not None and column in score_inputs.columns:
        return score_inputs[column].reindex(index)
    return None


def select_scores(
    scores: pd.Series,
    top_n: int,
    sector_control: SectorControlConfig,
    *,
    group_labels: pd.Series | None = None,
) -> pd.Series:
    clean = scores.dropna().astype(float)
    if clean.empty:
        return pd.Series(dtype=float, name=scores.name or "Score")
    ranked_scores = neutralized_scores(clean, group_labels) if sector_control.neutralize_scores else clean
    ranked = ranked_scores.dropna().sort_values(ascending=False)
    if not sector_control.enabled or group_labels is None:
        selected = ranked.head(top_n)
        return selected if len(selected) == top_n else empty_score_series(scores)

    labels = group_labels.reindex(ranked.index).fillna(UNKNOWN_GROUP).astype(str)
    selected_index: list[int] = []
    group_counts: dict[str, int] = {}
    max_per_group = sector_control.max_per_group
    for permno, _score in ranked.items():
        label = labels.loc[permno]
        current_count = group_counts.get(label, 0)
        if max_per_group is not None and current_count >= max_per_group:
            continue
        selected_index.append(int(permno))
        group_counts[label] = current_count + 1
        if len(selected_index) == top_n:
            break
    if len(selected_index) < top_n:
        return empty_score_series(scores)
    selected = ranked.loc[selected_index]
    if sector_control.min_groups > 0:
        selected_groups = labels.reindex(selected.index).nunique(dropna=True)
        if selected_groups < sector_control.min_groups:
            return empty_score_series(scores)
    return selected.rename(scores.name or "Score")


def neutralized_scores(scores: pd.Series, group_labels: pd.Series | None) -> pd.Series:
    if group_labels is None:
        return scores
    labels = group_labels.reindex(scores.index).fillna(UNKNOWN_GROUP)
    group_sizes = labels.groupby(labels).transform("size")
    group_means = scores.groupby(labels).transform("mean")
    adjusted = scores - group_means
    return adjusted.where(group_sizes > 1, scores)


def empty_score_series(scores: pd.Series) -> pd.Series:
    return pd.Series(dtype=float, name=scores.name or "Score")


def sic_group(value: object, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return UNKNOWN_GROUP
    cleaned = "".join(char for char in str(value).strip() if char.isdigit())
    if not cleaned:
        return UNKNOWN_GROUP
    digits = max(1, min(4, int(digits)))
    return f"SIC{cleaned[:digits].ljust(digits, '0')}"


def normalize_group_label(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return UNKNOWN_GROUP
    label = str(value).strip()
    return label if label else UNKNOWN_GROUP
