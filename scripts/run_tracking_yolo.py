"""Track YOLO detections with greedy IoU and report MOT-style metrics vs GT."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.mot_io import gt_path, read_gt, sequence_dir
from src.track_metrics import (
    fragmentation,
    gt_ids_per_frame,
    id_switches,
    load_detections,
    match_predicted_to_gt,
    run_tracker,
)
from src.yolo_detect import cache_path


def main() -> None:
    seq = config.SEQUENCE_NAME
    det_cache = cache_path(seq)
    if not det_cache.exists():
        raise SystemExit(f"Missing detections cache; run scripts/run_yolo.py first ({det_cache})")

    dets = load_detections(det_cache)
    tracks_df = run_tracker(dets)

    out_dir = config.OUTPUT_ROOT / seq
    tracks_path = out_dir / "tracks_yolo.parquet"
    tracks_df.to_parquet(tracks_path, index=False)

    seq_dir = sequence_dir(config.DATA_ROOT, config.SEQUENCE_SPLIT, seq)
    gt_by_frame = read_gt(gt_path(seq_dir))
    frame_to_gt_id = gt_ids_per_frame(gt_path(seq_dir))

    matches = match_predicted_to_gt(tracks_df, gt_by_frame)
    n_switches = id_switches(matches, frame_to_gt_id)
    frag = fragmentation(matches, frame_to_gt_id)

    n_frames = tracks_df["frame"].nunique()
    n_pred_tracks = tracks_df["track_id"].nunique()
    matched_tracks = len(matches)
    minutes = n_frames / 25.0 / 60.0  # SportsMOT = 25 fps

    metrics = {
        "sequence": seq,
        "n_frames": int(n_frames),
        "n_predicted_tracks": int(n_pred_tracks),
        "n_predicted_tracks_with_gt_match": int(matched_tracks),
        "n_gt_tracks": int(len({frame_to_gt_id[k] for k in frame_to_gt_id})),
        "id_switches": int(n_switches),
        "id_switches_per_minute": round(n_switches / minutes, 2) if minutes > 0 else 0.0,
        "fragmentation_mean": round(
            sum(frag.values()) / max(len(frag), 1), 2
        ),
        "fragmentation_per_gt_track": frag,
    }

    metrics_path = out_dir / "tracking_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Tracks: {tracks_path}")
    print(f"Metrics: {metrics_path}")
    for k, v in metrics.items():
        if k != "fragmentation_per_gt_track":
            print(f"  {k}: {v}")
    print(f"  fragmentation_per_gt_track: {len(frag)} GT ids, "
          f"min={min(frag.values(), default=0)}, max={max(frag.values(), default=0)}")


if __name__ == "__main__":
    main()
