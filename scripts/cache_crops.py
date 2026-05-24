"""Cache resized torso crops per (track, frame) to a single .npz for fast training."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd

import config
from src.embed import torso_crop
from src.mot_io import Box, image_dir, read_frame, sequence_dir


def main() -> None:
    seq = config.SEQUENCE_NAME
    out_dir = config.OUTPUT_ROOT / seq
    tracks = pd.read_parquet(out_dir / "tracks_yolo.parquet").sort_values(["frame", "track_id"])
    img_dir = image_dir(sequence_dir(config.DATA_ROOT, config.SEQUENCE_SPLIT, seq))

    H, W = config.CROP_H, config.CROP_W
    crops = np.zeros((len(tracks), H, W, 3), dtype=np.uint8)
    track_ids = np.zeros(len(tracks), dtype=np.int32)
    frame_ids = np.zeros(len(tracks), dtype=np.int32)

    t0 = time.time()
    last_frame = -1
    img = None
    for i, r in enumerate(tracks.itertuples(index=False)):
        if r.frame != last_frame:
            img = read_frame(img_dir / f"{r.frame:06d}.jpg")
            last_frame = r.frame
        crop = torso_crop(img, Box(r.left, r.top, r.width, r.height))
        if crop.shape[0] < 2 or crop.shape[1] < 2:
            resized = np.zeros((H, W, 3), dtype=np.uint8)
        else:
            resized = cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA)
        crops[i] = resized
        track_ids[i] = int(r.track_id)
        frame_ids[i] = int(r.frame)

    out_path = out_dir / "crops.npz"
    np.savez_compressed(out_path, crops=crops, track_ids=track_ids, frame_ids=frame_ids)

    print(f"Cached {len(crops)} crops ({H}x{W}) in {time.time() - t0:.1f}s")
    print(f"Memory footprint: {crops.nbytes / 1e6:.1f} MB")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
