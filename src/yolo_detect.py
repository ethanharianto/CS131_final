"""YOLOv8 person detection with on-disk parquet cache.

Detections are cached per sequence + per (model, conf) so config tweaks don't
silently reuse stale results.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from ultralytics import YOLO

import config
from src.mot_io import image_dir, list_frames, sequence_dir


def cache_path(sequence_name: str) -> Path:
    out_dir = config.OUTPUT_ROOT / sequence_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = Path(config.YOLO_MODEL).stem
    return out_dir / f"detections_{model_tag}_conf{config.YOLO_CONF:.2f}.parquet"


def detect_sequence(
    sequence_name: str = config.SEQUENCE_NAME,
    split: str = config.SEQUENCE_SPLIT,
    frame_start: int = config.FRAME_START,
    frame_end: int = config.FRAME_END,
    out_path: Path | None = None,
) -> Path:
    seq_dir = sequence_dir(config.DATA_ROOT, split, sequence_name)
    img_dir = image_dir(seq_dir)
    frames = list_frames(img_dir)
    if not frames:
        raise FileNotFoundError(f"No frames in {img_dir}")

    model = YOLO(config.YOLO_MODEL)
    rows: list[dict] = []

    for frame_path in frames:
        frame_id = int(frame_path.stem)
        if frame_id < frame_start or frame_id > frame_end:
            continue

        result = model.predict(
            str(frame_path),
            imgsz=config.YOLO_IMG_SIZE,
            conf=config.YOLO_CONF,
            iou=config.YOLO_IOU,
            classes=[config.YOLO_PERSON_CLASS],
            verbose=False,
        )[0]

        if len(result.boxes) == 0:
            rows.append(
                {"frame": frame_id, "left": -1, "top": -1, "width": 0, "height": 0, "conf": 0.0}
            )
            continue

        for box, conf in zip(
            result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), strict=True
        ):
            x1, y1, x2, y2 = box
            rows.append(
                {
                    "frame": frame_id,
                    "left": int(round(x1)),
                    "top": int(round(y1)),
                    "width": int(round(x2 - x1)),
                    "height": int(round(y2 - y1)),
                    "conf": float(conf),
                }
            )

    df = pd.DataFrame(rows, columns=["frame", "left", "top", "width", "height", "conf"])
    df = df[(df["width"] > 0) | (df["frame"].isin(df["frame"].unique()))]

    target = out_path or cache_path(sequence_name)
    df.to_parquet(target, index=False)
    return target
