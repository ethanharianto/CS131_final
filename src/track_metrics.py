"""Run the greedy IoU tracker on cached YOLO detections + compute ID-switch metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from src.detect import iou
from src.mot_io import Box
from src.track import GreedyTracker


def load_detections(parquet_path: Path) -> dict[int, list[Box]]:
    df = pd.read_parquet(parquet_path)
    by_frame: dict[int, list[Box]] = defaultdict(list)
    for row in df.itertuples(index=False):
        if row.width <= 0 or row.height <= 0:
            continue
        by_frame[int(row.frame)].append(
            Box(int(row.left), int(row.top), int(row.width), int(row.height))
        )
    return by_frame


def run_tracker(
    detections_by_frame: dict[int, list[Box]],
    iou_thresh: float | None = None,
    max_missed: int | None = None,
) -> pd.DataFrame:
    import config

    if iou_thresh is None:
        iou_thresh = config.TRACK_IOU_THRESH
    if max_missed is None:
        max_missed = config.TRACK_MAX_MISSED
    tracker = GreedyTracker(iou_thresh=iou_thresh, max_missed=max_missed)
    rows: list[dict] = []
    for frame in sorted(detections_by_frame.keys()):
        for track_id, box in tracker.update(detections_by_frame[frame]):
            rows.append(
                {
                    "frame": frame,
                    "track_id": track_id,
                    "left": box.left,
                    "top": box.top,
                    "width": box.width,
                    "height": box.height,
                }
            )
    return pd.DataFrame(rows, columns=["frame", "track_id", "left", "top", "width", "height"])


def match_predicted_to_gt(
    tracks_df: pd.DataFrame, gt_by_frame: dict[int, list[Box]]
) -> dict[int, list[tuple[int, int]]]:
    """For each frame, greedy IoU-match predicted track ids to GT box indices.

    Returns: predicted_track_id -> list of (frame, gt_index_in_frame) hits.
    """
    matches: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for frame, group in tracks_df.groupby("frame"):
        gt_boxes = gt_by_frame.get(int(frame), [])
        if not gt_boxes:
            continue
        pred_rows = list(group.itertuples(index=False))
        pred_boxes = [Box(r.left, r.top, r.width, r.height) for r in pred_rows]

        candidates: list[tuple[float, int, int]] = []
        for pi, pb in enumerate(pred_boxes):
            for gi, gb in enumerate(gt_boxes):
                v = iou(pb, gb)
                if v >= 0.3:
                    candidates.append((v, pi, gi))
        candidates.sort(reverse=True)
        used_p: set[int] = set()
        used_g: set[int] = set()
        for v, pi, gi in candidates:
            if pi in used_p or gi in used_g:
                continue
            used_p.add(pi)
            used_g.add(gi)
            matches[pred_rows[pi].track_id].append((int(frame), gi))
    return matches


def gt_ids_per_frame(gt_raw_path: Path) -> dict[tuple[int, int], int]:
    """Map (frame, gt_index_in_frame) -> gt track id, preserving order in gt.txt."""
    mapping: dict[tuple[int, int], int] = {}
    counts: dict[int, int] = defaultdict(int)
    with gt_raw_path.open() as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            frame = int(parts[0])
            gt_id = int(parts[1])
            cls = int(float(parts[8])) if len(parts) > 8 else 1
            if cls != 1:
                continue
            mapping[(frame, counts[frame])] = gt_id
            counts[frame] += 1
    return mapping


def id_switches(
    matches: dict[int, list[tuple[int, int]]],
    frame_to_gt_id: dict[tuple[int, int], int],
) -> int:
    """For each predicted track, count how often the matched GT-id changes between consecutive matched frames."""
    switches = 0
    for track_id, hits in matches.items():
        ordered = sorted(hits)
        gt_ids = [frame_to_gt_id.get((frame, gi)) for frame, gi in ordered]
        gt_ids = [g for g in gt_ids if g is not None]
        for a, b in zip(gt_ids, gt_ids[1:], strict=False):
            if a != b:
                switches += 1
    return switches


def fragmentation(
    matches: dict[int, list[tuple[int, int]]],
    frame_to_gt_id: dict[tuple[int, int], int],
) -> dict[int, int]:
    """Per-GT-id count of distinct predicted tracks that ever matched it (1 = perfectly tracked)."""
    by_gt: dict[int, set[int]] = defaultdict(set)
    for track_id, hits in matches.items():
        seen: set[int] = set()
        for frame, gi in hits:
            gid = frame_to_gt_id.get((frame, gi))
            if gid is not None:
                seen.add(gid)
        for gid in seen:
            by_gt[gid].add(track_id)
    return {gid: len(tracks) for gid, tracks in by_gt.items()}


def majority_track_class(matches: dict[int, list[tuple[int, int]]]) -> dict[int, int | None]:
    """For each predicted track id, return the most common GT match index (used for diagnostics)."""
    out: dict[int, int | None] = {}
    for track_id, hits in matches.items():
        if not hits:
            out[track_id] = None
            continue
        counts = Counter(hits)
        out[track_id] = counts.most_common(1)[0][0][1]
    return out
