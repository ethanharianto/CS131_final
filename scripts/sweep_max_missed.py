"""Sweep tracker max_missed and report within-track coherence (label-free)."""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import config
from src.embed import lab_histogram, torso_crop
from src.mot_io import Box, image_dir, read_frame, sequence_dir
from src.track import GreedyTracker
from src.track_metrics import load_detections
from src.yolo_detect import cache_path


def run_tracker_with(max_missed: int, dets: dict[int, list[Box]]) -> pd.DataFrame:
    tracker = GreedyTracker(iou_thresh=0.3, max_missed=max_missed)
    rows = []
    for frame in sorted(dets.keys()):
        for tid, box in tracker.update(dets[frame]):
            rows.append(
                {"frame": frame, "track_id": tid, "left": box.left, "top": box.top,
                 "width": box.width, "height": box.height}
            )
    return pd.DataFrame(rows)


def embed_tracks(tracks: pd.DataFrame, img_dir: Path) -> np.ndarray:
    last_frame = -1
    img = None
    embs = []
    for r in tracks.sort_values(["frame", "track_id"]).itertuples(index=False):
        if r.frame != last_frame:
            img = read_frame(img_dir / f"{r.frame:06d}.jpg")
            last_frame = r.frame
        embs.append(lab_histogram(torso_crop(img, Box(r.left, r.top, r.width, r.height))))
    return np.stack(embs).astype(np.float32)


def within_track_purity(track_ids: np.ndarray, cluster_ids: np.ndarray) -> dict[int, float]:
    by_track: dict[int, list[int]] = {}
    for t, c in zip(track_ids.tolist(), cluster_ids.tolist(), strict=True):
        by_track.setdefault(int(t), []).append(int(c))
    return {t: Counter(cs).most_common(1)[0][1] / len(cs) for t, cs in by_track.items()}


def main() -> None:
    seq = config.SEQUENCE_NAME
    out_dir = config.OUTPUT_ROOT / seq
    det_cache = cache_path(seq)
    dets = load_detections(det_cache)
    img_dir = image_dir(sequence_dir(config.DATA_ROOT, config.SEQUENCE_SPLIT, seq))

    print(f"{'max_missed':>10} | {'n_tracks':>8} | {'median_len':>10} | "
          f"{'top20_mean_purity':>17} | {'top20_min_purity':>17} | {'top20_frac>0.9':>15}")
    print("-" * 100)

    for mm in [1, 2, 3, 5, 10]:
        t0 = time.time()
        tracks = run_tracker_with(mm, dets)
        if len(tracks) == 0:
            print(f"{mm:>10} | (no tracks)")
            continue

        emb = embed_tracks(tracks, img_dir)
        km = KMeans(n_clusters=config.KMEANS_K, random_state=config.KMEANS_SEED, n_init=10).fit(emb)
        clusters = km.labels_

        track_ids = tracks.sort_values(["frame", "track_id"])["track_id"].to_numpy()
        purity = within_track_purity(track_ids, clusters)

        track_lengths = tracks.groupby("track_id").size().sort_values(ascending=False)
        top20 = track_lengths.head(20).index.tolist()
        top20_purities = [purity[t] for t in top20]

        print(
            f"{mm:>10} | {len(track_lengths):>8} | {int(track_lengths.median()):>10} | "
            f"{np.mean(top20_purities):>17.3f} | {min(top20_purities):>17.3f} | "
            f"{sum(p > 0.9 for p in top20_purities) / len(top20):>15.0%}"
            f"  ({time.time() - t0:.1f}s)"
        )


if __name__ == "__main__":
    main()
