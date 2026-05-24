"""Generate milestone figures: MOG2 vs YOLO panning, team-colored overlay strip."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

import config
from src.detect import Mog2Detector
from src.mot_io import Box, gt_path, image_dir, list_frames, read_frame, read_gt, sequence_dir
from src.eval_team import team_purity
from src.visualize import (
    draw_team_overlay,
    save_baseline_vs_e2e,
    save_confusion_pair,
    save_mog2_vs_yolo,
    save_purity_bar,
    save_timeline_strip,
    save_training_curve,
)


def main() -> None:
    seq = config.SEQUENCE_NAME
    out_dir = config.OUTPUT_ROOT / seq
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    seq_dir = sequence_dir(config.DATA_ROOT, config.SEQUENCE_SPLIT, seq)
    img_dir = image_dir(seq_dir)
    frames = list_frames(img_dir)
    gt_by_frame = read_gt(gt_path(seq_dir))

    # 1) MOG2 vs YOLO on a panning frame. We run MOG2 for warm-up then sample.
    yolo_df = pd.read_parquet(out_dir / "detections_yolov8n_conf0.25.parquet")
    sample_frame_id = 400  # mid-clip; broadcast camera will have moved
    sample_path = img_dir / f"{sample_frame_id:06d}.jpg"

    mog2 = Mog2Detector()
    for fp in frames[: sample_frame_id - 1]:
        mog2.detect(read_frame(fp))
    sample_img = read_frame(sample_path)
    mog2_boxes, _ = mog2.detect(sample_img)
    yolo_boxes = [
        Box(int(r.left), int(r.top), int(r.width), int(r.height))
        for r in yolo_df[yolo_df["frame"] == sample_frame_id].itertuples(index=False)
        if r.width > 0
    ]
    gt_boxes_sample = gt_by_frame.get(sample_frame_id, [])
    save_mog2_vs_yolo(
        sample_img,
        mog2_boxes,
        yolo_boxes,
        gt_boxes_sample,
        fig_dir / "mog2_vs_yolo_panning.png",
        title=f"{seq} frame {sample_frame_id}: MOG2 fails on panning broadcast, YOLO succeeds",
    )

    # 2) Team-colored overlay strip across the clip
    tracks = pd.read_parquet(out_dir / "tracks_yolo.parquet")
    baseline = json.loads((out_dir / "baseline_labels.json").read_text())
    team_labels = {int(k): v["team"] for k, v in baseline.items()}

    sample_frames = [100, 300, 500, 700]
    overlays = []
    for fid in sample_frames:
        img = read_frame(img_dir / f"{fid:06d}.jpg")
        rows = tracks[tracks["frame"] == fid]
        track_boxes = [
            (int(r.track_id), Box(int(r.left), int(r.top), int(r.width), int(r.height)))
            for r in rows.itertuples(index=False)
        ]
        overlays.append(draw_team_overlay(img, track_boxes, team_labels))

    save_timeline_strip(
        overlays,
        fig_dir / "team_overlay_baseline.png",
        title=f"{seq}: YOLO+IoU tracks colored by k-means majority-vote team",
    )

    # 3) Baseline vs end-to-end (M3) side-by-side on same frames
    e2e_path = out_dir / "learned_labels_e2e.json"
    if e2e_path.exists():
        e2e = json.loads(e2e_path.read_text())
        e2e_labels = {int(k): v["team"] for k, v in e2e.items()}
        compare_frames = [200, 400, 600, 800]
        base_panels, e2e_panels = [], []
        for fid in compare_frames:
            img = read_frame(img_dir / f"{fid:06d}.jpg")
            rows = tracks[tracks["frame"] == fid]
            track_boxes = [
                (int(r.track_id), Box(int(r.left), int(r.top), int(r.width), int(r.height)))
                for r in rows.itertuples(index=False)
            ]
            base_panels.append(draw_team_overlay(img, track_boxes, team_labels))
            e2e_panels.append(draw_team_overlay(img, track_boxes, e2e_labels))
        save_baseline_vs_e2e(
            base_panels, e2e_panels, compare_frames,
            fig_dir / "baseline_vs_e2e.png",
            title=f"{seq}: per-track team predictions, baseline vs M3 end-to-end",
        )

        # 4) Purity bar + confusion-matrix pair
        gt_path_ = out_dir / "team_gt.json"
        if gt_path_.exists():
            gt = json.loads(gt_path_.read_text())
            gt_clean = {
                int(k): v["team"] for k, v in gt.items()
                if v.get("team") in ("A", "B", "other")
            }
            base_r = team_purity(team_labels, gt_clean)
            e2e_r = team_purity(e2e_labels, gt_clean)
            save_purity_bar(
                {"baseline": base_r, "M3 end-to-end": e2e_r},
                fig_dir / "purity_bar.png",
            )
            save_confusion_pair(
                base_r["confusion_matrix"], e2e_r["confusion_matrix"],
                fig_dir / "confusion_pair.png",
            )

        # 5) Training curve
        metrics_path = out_dir / "team_model_e2e_metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            save_training_curve(metrics["history"], fig_dir / "training_curve_e2e.png")

    for p in sorted(fig_dir.glob("*.png")):
        print(f"Wrote: {p}")


if __name__ == "__main__":
    main()
