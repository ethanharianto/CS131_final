#!/usr/bin/env python3
"""Extract frames from any local MP4 into MOT-style img1/ (optional dev data)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mp4", type=Path, help="Path to local video file")
    parser.add_argument("--name", default="dev_local_mp4", help="Sequence folder name")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--fps-sample", type=int, default=1, help="Keep every Nth frame")
    args = parser.parse_args()

    if not args.mp4.is_file():
        print(f"Not found: {args.mp4}")
        return 1

    cap = cv2.VideoCapture(str(args.mp4))
    if not cap.isOpened():
        print(f"Cannot open: {args.mp4}")
        return 1

    seq_dir = config.DATA_ROOT / "train" / args.name
    img_dir = seq_dir / "img1"
    img_dir.mkdir(parents=True, exist_ok=True)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    n_in = 0
    n_out = 0
    while n_out < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        n_in += 1
        if (n_in - 1) % args.fps_sample != 0:
            continue
        n_out += 1
        cv2.imwrite(str(img_dir / f"{n_out:06d}.jpg"), frame)

    cap.release()

    (seq_dir / "seqinfo.ini").write_text(
        f"""[Sequence]
name={args.name}
imDir=img1
frameRate=25
seqLength={n_out}
imWidth={w}
imHeight={h}
imExt=.jpg
"""
    )
    print(f"Wrote {n_out} frames to {img_dir}")
    print(f"Run detection (no GT): python scripts/run_detection.py --sequence {args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
