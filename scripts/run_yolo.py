"""Run YOLOv8 person detection on the configured sequence and cache to parquet."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

import config
from src.yolo_detect import detect_sequence


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", default=config.SEQUENCE_NAME)
    ap.add_argument("--split", default=config.SEQUENCE_SPLIT)
    ap.add_argument("--frame-start", type=int, default=config.FRAME_START)
    ap.add_argument("--frame-end", type=int, default=config.FRAME_END)
    args = ap.parse_args()

    t0 = time.time()
    out = detect_sequence(
        sequence_name=args.sequence,
        split=args.split,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
    )
    elapsed = time.time() - t0

    df = pd.read_parquet(out)
    real_dets = df[df["width"] > 0]
    n_frames = df["frame"].nunique()
    print(f"YOLO inference: {elapsed:.1f}s on {n_frames} frames")
    print(f"Cache: {out}")
    print(f"Total detections: {len(real_dets)}, mean per frame: {len(real_dets) / n_frames:.2f}")
    print(f"Confidence quartiles: {real_dets['conf'].quantile([0.25, 0.5, 0.75]).to_dict()}")


if __name__ == "__main__":
    main()
