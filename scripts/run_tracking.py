#!/usr/bin/env python3
"""Run MOG2 detections + greedy IoU tracker; export track visualization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.detect import Mog2Detector
from src.mot_io import frame_index_from_name, image_dir, list_frames, read_frame
from src.track import GreedyTracker
def _color_for_id(tid: int) -> tuple[int, int, int]:
    rng = (37 * tid) % 255, (17 * tid) % 255, (97 * tid) % 255
    return int(rng[0]), int(rng[1]), int(rng[2])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default="dev_basketball_synth")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=config.FRAME_END)
    parser.add_argument("--sample-every", type=int, default=30)
    args = parser.parse_args()

    seq_dir = config.DATA_ROOT / "train" / args.sequence
    img_dir = image_dir(seq_dir)
    if not img_dir.is_dir():
        print(f"Missing {img_dir}. Run: python scripts/make_dev_sequence.py")
        return 1

    mog2 = Mog2Detector(
        min_area=config.MIN_BOX_AREA,
        max_area=config.MAX_BOX_AREA,
        min_aspect=config.MIN_ASPECT,
        max_aspect=config.MAX_ASPECT,
        kernel=config.MORPH_KERNEL,
    )
    tracker = GreedyTracker()
    out_dir = config.OUTPUT_ROOT / args.sequence / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    strips: list = []
    id_switches = 0
    prev_ids: set[int] = set()

    for path in list_frames(img_dir):
        idx = frame_index_from_name(path)
        if idx < args.start or idx > args.end:
            continue
        frame = read_frame(path)
        boxes, _ = mog2.detect(frame)
        active = tracker.update(boxes)
        cur_ids = {tid for tid, _ in active}
        if prev_ids and cur_ids:
            # crude: count new ids appearing while old ids vanished same frame
            id_switches += len(cur_ids - prev_ids) + len(prev_ids - cur_ids)
        prev_ids = cur_ids

        if idx % args.sample_every == 0:
            overlay = frame.copy()
            for tid, b in active:
                cv2.rectangle(
                    overlay,
                    (b.left, b.top),
                    (b.right, b.bottom),
                    _color_for_id(tid),
                    2,
                )
                cv2.putText(
                    overlay,
                    str(tid),
                    (b.left, max(b.top - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    _color_for_id(tid),
                    1,
                )
            strips.append(overlay)

    if strips:
        from src.visualize import save_timeline_strip

        save_timeline_strip(
            strips[:5],
            out_dir / "timeline_tracks.png",
            title="Greedy IoU tracks (colored by ID)",
        )

    summary = {
        "sequence": args.sequence,
        "frames": args.end - args.start + 1,
        "crude_id_events": id_switches,
        "figure": str(out_dir / "timeline_tracks.png"),
    }
    path_out = config.OUTPUT_ROOT / args.sequence / "tracking_metrics.json"
    path_out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
