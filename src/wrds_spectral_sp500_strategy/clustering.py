from __future__ import annotations

import numpy as np
import pandas as pd

from wrds_spectral_sp500_strategy.features import zscore_columns


def correlation_affinity(
    features: pd.DataFrame,
    *,
    nearest_neighbors: int | None,
    positive_only: bool,
) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame(index=features.index, columns=features.index)
    x = zscore_columns(features).to_numpy(dtype=float)
    corr = np.corrcoef(x)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    affinity = np.clip(corr, 0.0, 1.0) if positive_only else np.clip((corr + 1.0) / 2.0, 0.0, 1.0)
    np.fill_diagonal(affinity, 1.0)
    if nearest_neighbors is not None:
        affinity = knn_sparsify(affinity, nearest_neighbors)
    return pd.DataFrame(affinity, index=features.index, columns=features.index)


def knn_sparsify(affinity: np.ndarray, nearest_neighbors: int) -> np.ndarray:
    n = affinity.shape[0]
    if n <= 2 or nearest_neighbors >= n - 1:
        return affinity
    k = max(1, int(nearest_neighbors))
    sparse = np.zeros_like(affinity)
    for index in range(n):
        row = affinity[index].copy()
        row[index] = -np.inf
        keep = np.argpartition(row, -k)[-k:]
        sparse[index, keep] = affinity[index, keep]
    sparse = np.maximum(sparse, sparse.T)
    np.fill_diagonal(sparse, 1.0)
    return sparse


def cluster_returns(
    features: pd.DataFrame,
    *,
    n_clusters: int,
    nearest_neighbors: int | None,
    random_state: int,
    positive_only: bool,
) -> pd.Series:
    if len(features) < 2:
        return pd.Series(index=features.index, data=0, name="Cluster")
    clusters = min(int(n_clusters), len(features))
    if clusters < 2:
        return pd.Series(index=features.index, data=0, name="Cluster")

    from sklearn.cluster import SpectralClustering

    affinity = correlation_affinity(
        features,
        nearest_neighbors=nearest_neighbors,
        positive_only=positive_only,
    )
    model = SpectralClustering(
        n_clusters=clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=random_state,
    )
    labels = model.fit_predict(affinity.to_numpy(dtype=float))
    return pd.Series(labels, index=features.index, name="Cluster")

