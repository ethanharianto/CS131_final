"""Compute LAB torso-histogram embeddings for every (track_id, frame) pair."""

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


def main() -> None:
    seq = config.SEQUENCE_NAME
    out_dir = config.OUTPUT_ROOT / seq
    tracks_path = out_dir / "tracks_yolo.parquet"
    if not tracks_path.exists():
        raise SystemExit(f"Missing tracks; run scripts/run_tracking_yolo.py first ({tracks_path})")

    tracks = pd.read_parquet(tracks_path).sort_values(["frame", "track_id"])

    img_dir = image_dir(sequence_dir(config.DATA_ROOT, config.SEQUENCE_SPLIT, seq))

    track_ids: list[int] = []
    frame_ids: list[int] = []
    embeddings: list[np.ndarray] = []

    t0 = time.time()
    last_frame = -1
    frame_img: np.ndarray | None = None
    for row in tracks.itertuples(index=False):
        if row.frame != last_frame:
            frame_path = img_dir / f"{row.frame:06d}.jpg"
            frame_img = read_frame(frame_path)
            last_frame = row.frame
        assert frame_img is not None
        crop = torso_crop(frame_img, Box(row.left, row.top, row.width, row.height))
        hist = lab_histogram(crop)
        track_ids.append(int(row.track_id))
        frame_ids.append(int(row.frame))
        embeddings.append(hist)

    emb = np.stack(embeddings, axis=0).astype(np.float32)
    np.savez_compressed(
        out_dir / "embeddings.npz",
        track_ids=np.array(track_ids, dtype=np.int32),
        frame_ids=np.array(frame_ids, dtype=np.int32),
        embeddings=emb,
    )
    elapsed = time.time() - t0
    print(f"Embedded {len(track_ids)} (track, frame) pairs in {elapsed:.1f}s")
    print(f"Embedding shape: {emb.shape}")
    print(f"Output: {out_dir / 'embeddings.npz'}")


if __name__ == "__main__":
    main()
