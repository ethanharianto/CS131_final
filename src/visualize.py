"""Figures for milestone and debugging."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.mot_io import Box


def draw_boxes(
    frame_bgr: np.ndarray,
    boxes: list[Box],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    out = frame_bgr.copy()
    for b in boxes:
        cv2.rectangle(
            out,
            (b.left, b.top),
            (b.right, b.bottom),
            color,
            thickness,
        )
    return out


def save_panel(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    det_boxes: list[Box],
    gt_boxes: list[Box],
    title: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    overlay_det = draw_boxes(frame_bgr, det_boxes, (0, 255, 0))
    overlay_both = draw_boxes(overlay_det, gt_boxes, (255, 80, 0))

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("RGB frame")
    axes[0, 1].imshow(mask, cmap="gray")
    axes[0, 1].set_title("Foreground mask")
    axes[1, 0].imshow(cv2.cvtColor(overlay_det, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f"MOG2 detections ({len(det_boxes)})")
    axes[1, 1].imshow(cv2.cvtColor(overlay_both, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f"+ GT players ({len(gt_boxes)})")

    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


TEAM_COLORS_BGR = {
    "A": (50, 50, 220),    # red
    "B": (220, 120, 30),   # blue
    "other": (160, 160, 160),  # grey
}


def draw_team_overlay(
    frame_bgr: np.ndarray,
    track_boxes: list[tuple[int, Box]],
    team_labels: dict[int, str],
    thickness: int = 3,
) -> np.ndarray:
    out = frame_bgr.copy()
    for track_id, b in track_boxes:
        team = team_labels.get(track_id, "other")
        color = TEAM_COLORS_BGR.get(team, (160, 160, 160))
        cv2.rectangle(out, (b.left, b.top), (b.right, b.bottom), color, thickness)
        cv2.putText(
            out,
            f"{team}:{track_id}",
            (b.left, max(b.top - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def save_mog2_vs_yolo(
    frame_bgr: np.ndarray,
    mog2_boxes: list[Box],
    yolo_boxes: list[Box],
    gt_boxes: list[Box],
    out_path: Path,
    title: str = "MOG2 vs YOLO on broadcast basketball",
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    mog_overlay = draw_boxes(frame_bgr, mog2_boxes, (60, 60, 220), 2)
    yolo_overlay = draw_boxes(frame_bgr, yolo_boxes, (50, 200, 50), 2)
    gt_overlay = draw_boxes(frame_bgr, gt_boxes, (255, 120, 0), 2)

    axes[0].imshow(cv2.cvtColor(mog_overlay, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"MOG2 background subtraction ({len(mog2_boxes)} boxes)")
    axes[1].imshow(cv2.cvtColor(yolo_overlay, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"YOLOv8n person detector ({len(yolo_boxes)} boxes)")
    axes[2].imshow(cv2.cvtColor(gt_overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"Ground truth players ({len(gt_boxes)} boxes)")

    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_baseline_vs_e2e(
    frames_baseline: list[np.ndarray],
    frames_e2e: list[np.ndarray],
    frame_ids: list[int],
    out_path: Path,
    title: str = "Baseline (k-means on LAB hist) vs end-to-end CNN+BiLSTM (M3)",
) -> None:
    n = len(frames_baseline)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6))
    if n == 1:
        axes = axes.reshape(2, 1)
    for j, fid in enumerate(frame_ids):
        axes[0, j].imshow(cv2.cvtColor(frames_baseline[j], cv2.COLOR_BGR2RGB))
        axes[0, j].set_title(f"baseline, frame {fid}", fontsize=9)
        axes[1, j].imshow(cv2.cvtColor(frames_e2e[j], cv2.COLOR_BGR2RGB))
        axes[1, j].set_title(f"M3 end-to-end, frame {fid}", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_purity_bar(
    results: dict[str, dict],
    out_path: Path,
    title: str = "Team-purity on 30 hand-labeled tracks",
) -> None:
    names = list(results.keys())
    purities = [results[n]["overall_purity"] for n in names]
    correct = [results[n]["n_correct"] for n in names]
    total = results[names[0]]["n_labeled_tracks"]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#888888", "#3a7dbb", "#d35f00"][: len(names)]
    bars = ax.bar(names, purities, color=colors)
    for bar, p, c in zip(bars, purities, correct):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            p + 0.01,
            f"{p:.3f}\n({c}/{total})",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Overall purity")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_confusion_pair(
    cm_baseline: dict[str, dict[str, int]],
    cm_e2e: dict[str, dict[str, int]],
    out_path: Path,
    classes: tuple[str, ...] = ("A", "B", "other"),
    title: str = "Confusion matrices (rows = GT, cols = pred)",
) -> None:
    def to_array(cm):
        return np.array(
            [[cm.get(r, {}).get(c, 0) for c in classes] for r in classes], dtype=float
        )

    a, b = to_array(cm_baseline), to_array(cm_e2e)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, mat, name in zip(axes, (a, b), ("baseline", "M3 end-to-end")):
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(a.max(), b.max(), 1))
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        ax.set_xlabel("predicted")
        ax.set_ylabel("ground truth")
        ax.set_title(name)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(
                    j, i, int(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black",
                    fontsize=11,
                )
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_training_curve(
    history: list[dict],
    out_path: Path,
    title: str = "End-to-end CNN+BiLSTM training",
) -> None:
    epochs = [h["epoch"] for h in history]
    loss = [h["loss"] for h in history]
    acc = [h.get("train_acc_vs_pseudo") for h in history]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(epochs, loss, color="#d35f00", marker="o", label="loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss (class-weighted CE)", color="#d35f00")
    ax1.tick_params(axis="y", labelcolor="#d35f00")
    ax2 = ax1.twinx()
    ax2.plot(epochs, acc, color="#3a7dbb", marker="s", label="train acc vs pseudo")
    ax2.set_ylabel("train acc vs pseudo-labels", color="#3a7dbb")
    ax2.tick_params(axis="y", labelcolor="#3a7dbb")
    ax2.set_ylim(0, 1.0)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_timeline_strip(
    frames_and_overlays: list[np.ndarray],
    out_path: Path,
    title: str = "Detection timeline",
) -> None:
    n = len(frames_and_overlays)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, img in zip(axes, frames_and_overlays):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
