"""Prepare a hand-labeling worksheet for predicted tracks.

For the top-N longest predicted tracks, samples 3 frames evenly across the track's
lifespan, stitches the player crops into a single montage image per track, and writes
a starter JSON the user fills in by inspecting the montages.

Workflow:
  1. python scripts/label_teams.py        # generates montages + empty JSON template
  2. Open each PNG in outputs/<seq>/labeling/, read the team off the jersey
  3. Edit outputs/<seq>/team_gt.json: replace each null with "A" | "B" | "other"
  4. python scripts/eval_baseline.py      # computes team purity vs baseline
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd

import config
from src.mot_io import Box, image_dir, read_frame, sequence_dir

PER_TEAM = 10  # top-N longest tracks from each baseline-predicted team (A, B, other)
N_SAMPLES = 3
MONTAGE_HEIGHT = 220


def crop_with_context(frame_bgr: np.ndarray, box: Box, pad: float = 0.25) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    px = int(box.width * pad)
    py = int(box.height * pad)
    x0 = max(0, box.left - px)
    x1 = min(w, box.right + px)
    y0 = max(0, box.top - py)
    y1 = min(h, box.bottom + py)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((MONTAGE_HEIGHT, MONTAGE_HEIGHT // 2, 3), dtype=np.uint8)
    crop = frame_bgr[y0:y1, x0:x1].copy()
    cv2.rectangle(
        crop,
        (box.left - x0, box.top - y0),
        (box.right - x0, box.bottom - y0),
        (0, 255, 255),
        2,
    )
    return crop


def resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0:
        return img
    scale = target_h / h
    return cv2.resize(img, (max(1, int(w * scale)), target_h), interpolation=cv2.INTER_AREA)


def make_track_montage(
    frames_and_boxes: list[tuple[int, np.ndarray, Box]], track_id: int
) -> np.ndarray:
    panels: list[np.ndarray] = []
    for fid, img, box in frames_and_boxes:
        crop = crop_with_context(img, box)
        crop = resize_to_height(crop, MONTAGE_HEIGHT)
        cv2.putText(
            crop,
            f"f={fid}",
            (4, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(crop)
        panels.append(np.full((MONTAGE_HEIGHT, 6, 3), 32, dtype=np.uint8))
    if panels:
        panels.pop()
    montage = np.concatenate(panels, axis=1) if panels else np.zeros(
        (MONTAGE_HEIGHT, 100, 3), dtype=np.uint8
    )
    header = np.full((36, montage.shape[1], 3), 24, dtype=np.uint8)
    cv2.putText(
        header,
        f"track_id = {track_id}",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return np.concatenate([header, montage], axis=0)


def main() -> None:
    seq = config.SEQUENCE_NAME
    out_dir = config.OUTPUT_ROOT / seq
    tracks_path = out_dir / "tracks_yolo.parquet"
    if not tracks_path.exists():
        raise SystemExit(f"Missing tracks; run scripts/run_tracking_yolo.py first ({tracks_path})")

    tracks = pd.read_parquet(tracks_path)
    track_lengths = tracks.groupby("track_id").size().sort_values(ascending=False)

    baseline_path = out_dir / "baseline_labels.json"
    if not baseline_path.exists():
        raise SystemExit(f"Missing baseline; run scripts/run_baseline.py first ({baseline_path})")
    baseline = json.loads(baseline_path.read_text())
    track_team = {int(k): v["team"] for k, v in baseline.items()}

    selected: list[int] = []
    for team in ("A", "B", "other"):
        team_tracks = [t for t in track_lengths.index if track_team.get(int(t)) == team]
        picked = team_tracks[:PER_TEAM]
        selected.extend(picked)
        print(f"  predicted {team}: {len(picked)} longest tracks selected "
              f"({picked[:3]}... lengths {[int(track_lengths[t]) for t in picked[:3]]})")

    img_dir = image_dir(sequence_dir(config.DATA_ROOT, config.SEQUENCE_SPLIT, seq))
    label_dir = out_dir / "labeling"
    label_dir.mkdir(parents=True, exist_ok=True)

    template: dict[int, dict] = {}

    print(f"\nGenerating montages for {len(selected)} stratified tracks:")

    for tid in selected:
        rows = tracks[tracks["track_id"] == tid].sort_values("frame")
        if len(rows) == 0:
            continue
        idxs = np.linspace(0, len(rows) - 1, N_SAMPLES, dtype=int)
        sampled = rows.iloc[idxs]
        frames_data: list[tuple[int, np.ndarray, Box]] = []
        for r in sampled.itertuples(index=False):
            img = read_frame(img_dir / f"{r.frame:06d}.jpg")
            frames_data.append((int(r.frame), img, Box(r.left, r.top, r.width, r.height)))

        montage = make_track_montage(frames_data, int(tid))
        out_png = label_dir / f"track_{int(tid):04d}.png"
        cv2.imwrite(str(out_png), montage)
        template[int(tid)] = {"team": None, "n_frames": int(len(rows))}
        print(f"  track {int(tid):4d}: {len(rows):4d} frames -> {out_png.name}")

    gt_path = out_dir / "team_gt.json"
    if gt_path.exists():
        print(f"\nNote: {gt_path} already exists; not overwriting.")
    else:
        gt_path.write_text(json.dumps(template, indent=2, sort_keys=True))
        print(f"\nWrote starter labels: {gt_path}")

    print(f"\nNext steps:")
    print(f"  1. Open {label_dir} in Finder; view the PNGs")
    print(f"  2. For each track, edit {gt_path} and replace null with \"A\" | \"B\" | \"other\"")
    print(f"  3. Run: python scripts/eval_baseline.py")


if __name__ == "__main__":
    main()
