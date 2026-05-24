#!/usr/bin/env python3
"""Run MOG2 + frame-diff baselines on a SportsMOT basketball sequence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.detect import FrameDiffDetector, Mog2Detector, match_greedy
from src.mot_io import frame_index_from_name, gt_path, image_dir, list_frames, read_frame, read_gt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train")
    parser.add_argument("--sequence", default=config.SEQUENCE_NAME)
    parser.add_argument("--start", type=int, default=config.FRAME_START)
    parser.add_argument("--end", type=int, default=config.FRAME_END)
    parser.add_argument("--sample-every", type=int, default=config.SAMPLE_EVERY)
    args = parser.parse_args()

    seq_dir = config.DATA_ROOT / args.split / args.sequence
    img_dir = image_dir(seq_dir)
    if not img_dir.is_dir():
        print(
            f"Missing sequence at {img_dir}\n"
            "Download SportsMOT train split (Codalab) and unzip to data/dataset/train/<name>/img1/\n"
            "See README.md for steps."
        )
        return 1

    frames = list_frames(img_dir)
    gt = read_gt(gt_path(seq_dir))
    mog2 = Mog2Detector(
        min_area=config.MIN_BOX_AREA,
        max_area=config.MAX_BOX_AREA,
        min_aspect=config.MIN_ASPECT,
        max_aspect=config.MAX_ASPECT,
        kernel=config.MORPH_KERNEL,
    )
    fd = FrameDiffDetector(
        min_area=config.MIN_BOX_AREA,
        max_area=config.MAX_BOX_AREA,
        min_aspect=config.MIN_ASPECT,
        max_aspect=config.MAX_ASPECT,
        kernel=config.MORPH_KERNEL,
    )

    out_dir = config.OUTPUT_ROOT / args.sequence
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    mog2_stats = {"tp": 0, "fp": 0, "frames": 0, "mean_iou": []}
    fd_stats = {"tp": 0, "fp": 0, "frames": 0, "mean_iou": []}
    panel_saved = False
    timeline: list = []

    for path in frames:
        idx = frame_index_from_name(path)
        if idx < args.start or idx > args.end:
            continue

        frame = read_frame(path)
        mog_boxes, mog_mask = mog2.detect(frame)
        fd_boxes, _ = fd.detect(frame)
        gts = gt.get(idx, [])

        if gts:
            tp, fp, miou = match_greedy(mog_boxes, gts)
            mog2_stats["tp"] += tp
            mog2_stats["fp"] += fp
            mog2_stats["frames"] += 1
            mog2_stats["mean_iou"].append(miou)
            tp2, fp2, miou2 = match_greedy(fd_boxes, gts)
            fd_stats["tp"] += tp2
            fd_stats["fp"] += fp2
            fd_stats["frames"] += 1
            fd_stats["mean_iou"].append(miou2)

        if idx % args.sample_every == 0:
            from src.visualize import draw_boxes, save_panel

            if not panel_saved:
                save_panel(
                    frame,
                    mog_mask,
                    mog_boxes,
                    gts,
                    f"{args.sequence} frame {idx}",
                    fig_dir / "panel_mog2_vs_gt.png",
                )
                panel_saved = True
            timeline.append(draw_boxes(frame, mog_boxes))

    if timeline:
        from src.visualize import save_timeline_strip

        save_timeline_strip(
            timeline[:5],
            fig_dir / "timeline_mog2.png",
            title=f"MOG2 proposals every {args.sample_every} frames",
        )

    def summarize(s: dict) -> dict:
        prec = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) else 0.0
        return {
            "frames_evaluated": s["frames"],
            "tp": s["tp"],
            "fp": s["fp"],
            "precision_at_iou_0.3": round(prec, 4),
            "mean_match_iou": round(float(sum(s["mean_iou"]) / len(s["mean_iou"])), 4)
            if s["mean_iou"]
            else 0.0,
        }

    summary = {
        "sequence": args.sequence,
        "frame_range": [args.start, args.end],
        "mog2": summarize(mog2_stats),
        "frame_diff": summarize(fd_stats),
        "figures": {
            "panel": str(fig_dir / "panel_mog2_vs_gt.png"),
            "timeline": str(fig_dir / "timeline_mog2.png"),
        },
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nWrote figures to {fig_dir}")
    print(f"Metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
