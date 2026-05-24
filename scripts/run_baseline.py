"""Compute the per-track team-id baseline: global k-means + majority vote."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

import config
from src.baseline import cluster_embeddings, majority_vote, remap_clusters_by_track_count


def main() -> None:
    seq = config.SEQUENCE_NAME
    out_dir = config.OUTPUT_ROOT / seq
    emb_path = out_dir / "embeddings.npz"
    if not emb_path.exists():
        raise SystemExit(f"Missing embeddings; run scripts/run_embed.py first ({emb_path})")

    data = np.load(emb_path)
    track_ids = data["track_ids"]
    embeddings = data["embeddings"]

    cluster_ids, centroids, track_to_cluster = cluster_embeddings(
        embeddings, track_ids, k=config.KMEANS_K
    )
    label_map = remap_clusters_by_track_count(track_to_cluster, k=config.KMEANS_K)
    labels = majority_vote(track_ids, cluster_ids, label_map, track_to_cluster=track_to_cluster)

    (out_dir / "baseline_labels.json").write_text(json.dumps(labels, indent=2, sort_keys=True))
    (out_dir / "cluster_to_team.json").write_text(
        json.dumps({str(k): v for k, v in label_map.items()}, indent=2, sort_keys=True)
    )
    np.save(out_dir / "kmeans_centroids.npy", centroids)
    np.save(out_dir / "kmeans_cluster_ids.npy", cluster_ids)

    team_counts = {
        "A": sum(1 for v in labels.values() if v["team"] == "A"),
        "B": sum(1 for v in labels.values() if v["team"] == "B"),
        "other": sum(1 for v in labels.values() if v["team"] == "other"),
    }
    confidences = [v["vote_confidence"] for v in labels.values()]

    print(f"Predicted tracks: {len(labels)}")
    print(f"Cluster -> team map: {label_map}")
    print(f"Team distribution: {team_counts}")
    print(
        f"Vote confidence quartiles: "
        f"min={min(confidences):.2f}, median={float(np.median(confidences)):.2f}, "
        f"max={max(confidences):.2f}"
    )
    print(f"Output: {out_dir / 'baseline_labels.json'}")


if __name__ == "__main__":
    main()
