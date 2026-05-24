#!/usr/bin/env python3
"""Create a tiny MOT-format dev sequence (no download) for pipeline testing."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config

NAME = "dev_basketball_synth"
NUM_FRAMES = 150
FPS = 25
W, H = 1280, 720


def _wood_frame() -> np.ndarray:
    base = np.zeros((H, W, 3), dtype=np.uint8)
    base[:] = (32, 90, 140)
    for y in range(0, H, 40):
        cv2.line(base, (0, y), (W, y), (38, 100, 150), 1)
    cv2.rectangle(base, (80, 80), (W - 80, H - 80), (220, 220, 220), 2)
    return base


def _players(t: int) -> list[tuple[int, int, int, int, int, tuple[int, int, int]]]:
    """Return list of (track_id, x, y, w, h, jersey_bgr)."""
    players = []
    specs = [
        (1, 0.12, 0.35, (40, 40, 200)),
        (2, 0.22, 0.42, (200, 40, 40)),
        (3, 0.35, 0.38, (40, 40, 200)),
        (4, 0.48, 0.45, (200, 40, 40)),
        (5, 0.60, 0.40, (40, 40, 200)),
        (6, 0.72, 0.48, (200, 40, 40)),
        (7, 0.30, 0.55, (40, 40, 200)),
        (8, 0.55, 0.58, (200, 40, 40)),
    ]
    for tid, cx0, cy0, color in specs:
        cx = int((cx0 + 0.04 * math.sin(t / 18.0 + tid)) * W)
        cy = int((cy0 + 0.03 * math.cos(t / 22.0 + tid * 0.7)) * H)
        w, h = 52, 118
        x, y = cx - w // 2, cy - h // 2
        players.append((tid, x, y, w, h, color))
    return players


def main() -> int:
    seq_dir = config.DATA_ROOT / "train" / NAME
    img_dir = seq_dir / "img1"
    gt_dir = seq_dir / "gt"
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    gt_lines: list[str] = []
    for f in range(1, NUM_FRAMES + 1):
        frame = _wood_frame()
        for tid, x, y, w, h, color in _players(f):
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
            cv2.rectangle(frame, (x, y - 18), (x + w, y), (255, 255, 255), -1)
            gt_lines.append(f"{f},{tid},{x},{y},{w},{h},1,1,1\n")
        cv2.imwrite(str(img_dir / f"{f:06d}.jpg"), frame)

    (gt_dir / "gt.txt").write_text("".join(gt_lines))
    (seq_dir / "seqinfo.ini").write_text(
        f"""[Sequence]
name={NAME}
imDir=img1
frameRate={FPS}
seqLength={NUM_FRAMES}
imWidth={W}
imHeight={H}
imExt=.jpg
"""
    )
    print(f"Created dev sequence: {seq_dir}")
    print(f"  frames: {NUM_FRAMES}  size: {W}x{H}")
    print("Run:  python scripts/run_detection.py --sequence dev_basketball_synth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
