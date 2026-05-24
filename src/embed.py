"""LAB color-histogram embeddings of torso crops per (track, frame)."""

from __future__ import annotations

import cv2
import numpy as np

import config
from src.mot_io import Box


def torso_crop(frame_bgr: np.ndarray, box: Box) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    x0 = max(0, box.left + int(config.TORSO_LEFT * box.width))
    x1 = min(w, box.left + int(config.TORSO_RIGHT * box.width))
    y0 = max(0, box.top + int(config.TORSO_TOP * box.height))
    y1 = min(h, box.top + int(config.TORSO_BOT * box.height))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return frame_bgr[y0:y1, x0:x1]


def lab_histogram(
    crop_bgr: np.ndarray,
    bins: tuple[int, int, int] = config.HIST_BINS,
    min_chroma: float | None = None,
) -> np.ndarray:
    if crop_bgr.size == 0 or crop_bgr.shape[0] < 2 or crop_bgr.shape[1] < 2:
        size = bins[0] * bins[1] * bins[2]
        return np.full(size, 1.0 / size, dtype=np.float32)

    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)

    mask = None
    threshold = config.MIN_CHROMA if min_chroma is None else min_chroma
    if threshold and threshold > 0:
        a = lab[:, :, 1].astype(np.float32) - 128.0
        b = lab[:, :, 2].astype(np.float32) - 128.0
        chroma = np.sqrt(a * a + b * b)
        mask = (chroma >= threshold).astype(np.uint8) * 255
        if mask.sum() == 0:
            mask = None  # fall back to all pixels if mask kills everything

    hist = cv2.calcHist(
        [lab],
        channels=[0, 1, 2],
        mask=mask,
        histSize=list(bins),
        ranges=[0, 256, 0, 256, 0, 256],
    ).astype(np.float32).flatten()
    total = hist.sum()
    if total > 0:
        hist /= total
    else:
        size = bins[0] * bins[1] * bins[2]
        hist = np.full(size, 1.0 / size, dtype=np.float32)
    return hist
