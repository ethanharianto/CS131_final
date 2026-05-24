"""Global LAB k-means + per-track majority-vote team-id baseline."""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.cluster import KMeans

import config


def cluster_embeddings(
    embeddings: np.ndarray,
    track_ids: np.ndarray,
    k: int = config.KMEANS_K,
    seed: int = config.KMEANS_SEED,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """Fit k-means on per-track mean embeddings, then predict per-frame cluster ids.

    Per-track fitting prevents long stationary tracks (e.g., sideline staff) from
    over-weighting the cluster centroids. Each track contributes one point to the fit;
    per-frame predictions use the fitted centroids.

    Returns:
        per_frame_cluster_ids: (N,) cluster id per row of `embeddings`
        centroids: (k, D) cluster centroids in LAB-histogram space
        track_to_cluster: {track_id -> cluster_id} from the per-track fit
    """
    unique_tids, inverse = np.unique(track_ids, return_inverse=True)
    n_tracks = unique_tids.shape[0]
    track_means = np.zeros((n_tracks, embeddings.shape[1]), dtype=np.float32)
    counts = np.zeros(n_tracks, dtype=np.int64)
    for idx, row in zip(inverse, embeddings):
        track_means[idx] += row
        counts[idx] += 1
    track_means /= counts[:, None]

    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    track_clusters = km.fit_predict(track_means)

    per_frame_cluster_ids = km.predict(embeddings).astype(np.int32)
    track_to_cluster = {int(t): int(c) for t, c in zip(unique_tids.tolist(), track_clusters.tolist())}
    return per_frame_cluster_ids, km.cluster_centers_.astype(np.float32), track_to_cluster


def remap_clusters_by_track_count(
    track_to_cluster: dict[int, int], k: int
) -> dict[int, str]:
    """Two largest clusters (by track count) -> A, B; smallest -> other."""
    counts = Counter(track_to_cluster.values())
    ordered = [c for c, _ in counts.most_common()]
    labels = ["A", "B"] + ["other"] * max(0, k - 2)
    mapping = {int(c): labels[i] if i < len(labels) else "other" for i, c in enumerate(ordered)}
    for c in range(k):
        mapping.setdefault(c, "other")
    return mapping


def majority_vote(
    track_ids: np.ndarray,
    cluster_ids: np.ndarray,
    label_map: dict[int, str],
    track_to_cluster: dict[int, int] | None = None,
) -> dict[int, dict]:
    """Per track: team label from the per-track k-means decision (preferred) or per-frame majority.

    If `track_to_cluster` is provided, the team label comes directly from the cluster the
    track's *mean* embedding was assigned to during k-means fitting. This is more robust
    than per-frame voting, which can be diluted by background or skin pixels in individual
    crops. Per-frame cluster IDs are still recorded for the BiLSTM's training signal.
    """
    by_track: dict[int, list[int]] = {}
    for t, c in zip(track_ids.tolist(), cluster_ids.tolist(), strict=True):
        by_track.setdefault(int(t), []).append(int(c))

    out: dict[int, dict] = {}
    for tid, clusters in by_track.items():
        counts = Counter(clusters)
        top_cluster, top_count = counts.most_common(1)[0]
        if track_to_cluster is not None:
            decision_cluster = int(track_to_cluster.get(int(tid), top_cluster))
        else:
            decision_cluster = int(top_cluster)
        out[int(tid)] = {
            "team": label_map[decision_cluster],
            "decision_cluster": decision_cluster,
            "majority_cluster_per_frame": int(top_cluster),
            "vote_confidence": round(top_count / len(clusters), 4),
            "n_frames": len(clusters),
            "per_frame_clusters": clusters,
        }
    return out
