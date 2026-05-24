"""Sweep tracker (iou_thresh, max_missed) and report per-track embedding coherence."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import config
from src.embed import lab_histogram, torso_crop
from src.mot_io import Box, image_dir, read_frame, sequence_dir
from src.track import GreedyTracker
from src.track_metrics import load_detections
from src.yolo_detect import cache_path


def run(iou_thresh: float, max_missed: int, dets):
    tracker = GreedyTracker(iou_thresh=iou_thresh, max_missed=max_missed)
    rows = []
    for f in sorted(dets.keys()):
        for tid, b in tracker.update(dets[f]):
            rows.append((f, tid, b.left, b.top, b.width, b.height))
    return pd.DataFrame(rows, columns=["frame", "track_id", "left", "top", "width", "height"])


def per_track_coherence(tracks: pd.DataFrame, img_dir: Path, min_len: int = 50):
    """Mean per-frame L2 distance from each track's mean histogram. Lower = more coherent."""
    by_track = {tid: g for tid, g in tracks.groupby("track_id") if len(g) >= min_len}

    last_frame, img = -1, None
    embs_by_track: dict[int, list[np.ndarray]] = {tid: [] for tid in by_track}

    sorted_rows = pd.concat(by_track.values()).sort_values(["frame", "track_id"])
    for r in sorted_rows.itertuples(index=False):
        if r.frame != last_frame:
            img = read_frame(img_dir / f"{r.frame:06d}.jpg")
            last_frame = r.frame
        embs_by_track[int(r.track_id)].append(
            lab_histogram(torso_crop(img, Box(r.left, r.top, r.width, r.height)))
        )

    coherence = {}
    for tid, es in embs_by_track.items():
        if not es:
            continue
        m = np.stack(es)
        mean = m.mean(axis=0, keepdims=True)
        dists = np.linalg.norm(m - mean, axis=1)
        coherence[tid] = float(dists.mean())
    return coherence, {tid: len(es) for tid, es in embs_by_track.items()}


def main() -> None:
    seq = config.SEQUENCE_NAME
    dets = load_detections(cache_path(seq))
    img_dir = image_dir(sequence_dir(config.DATA_ROOT, config.SEQUENCE_SPLIT, seq))

    print(f"{'iou':>4} | {'mm':>3} | {'n_tracks':>8} | {'n_long(>=50)':>12} | "
          f"{'med_long_len':>12} | {'top20_med_coh':>13} | {'top20_p90_coh':>13}")
    print("-" * 100)

    for iou in [0.3, 0.4, 0.5]:
        for mm in [1, 3, 10]:
            t0 = time.time()
            tracks = run(iou, mm, dets)
            if len(tracks) == 0:
                print(f"{iou:>4} | {mm:>3} | (empty)")
                continue
            coh, lens = per_track_coherence(tracks, img_dir, min_len=50)
            if not coh:
                print(f"{iou:>4} | {mm:>3} | {tracks['track_id'].nunique():>8} | 0 long tracks")
                continue
            long_tracks = sorted(coh.items(), key=lambda kv: -lens[kv[0]])[:20]
            top20_coh = sorted([c for _, c in long_tracks])
            print(
                f"{iou:>4} | {mm:>3} | {tracks['track_id'].nunique():>8} | {len(coh):>12} | "
                f"{int(np.median(list(lens.values()))):>12} | "
                f"{np.median(top20_coh):>13.4f} | {np.percentile(top20_coh, 90):>13.4f}"
                f"  ({time.time()-t0:.1f}s)"
            )


if __name__ == "__main__":
    main()
