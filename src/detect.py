"""Classical foreground-based player proposals."""

from __future__ import annotations

import cv2
import numpy as np

from src.mot_io import Box


def _filter_boxes(
    boxes: list[Box],
    min_area: int,
    max_area: int,
    min_aspect: float,
    max_aspect: float,
) -> list[Box]:
    out: list[Box] = []
    for b in boxes:
        if b.area < min_area or b.area > max_area:
            continue
        if b.aspect < min_aspect or b.aspect > max_aspect:
            continue
        out.append(b)
    return out


def boxes_from_mask(
    mask: np.ndarray,
    min_area: int,
    max_area: int,
    min_aspect: float,
    max_aspect: float,
    kernel: tuple[int, int],
) -> list[Box]:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel)
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
    boxes: list[Box] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        boxes.append(Box(int(x), int(y), int(w), int(h)))
    return _filter_boxes(boxes, min_area, max_area, min_aspect, max_aspect)


class Mog2Detector:
    def __init__(
        self,
        history: int = 300,
        var_threshold: float = 16.0,
        detect_shadows: bool = True,
        min_area: int = 800,
        max_area: int = 80_000,
        min_aspect: float = 0.15,
        max_aspect: float = 1.2,
        kernel: tuple[int, int] = (5, 5),
    ):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.kernel = kernel

    def detect(self, frame_bgr: np.ndarray) -> tuple[list[Box], np.ndarray]:
        fg = self.subtractor.apply(frame_bgr)
        # MOG2 shadow label = 127
        _, binary = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        boxes = boxes_from_mask(
            binary,
            self.min_area,
            self.max_area,
            self.min_aspect,
            self.max_aspect,
            self.kernel,
        )
        return boxes, binary


class FrameDiffDetector:
    """Naive per-frame baseline (no temporal model beyond previous frame)."""

    def __init__(
        self,
        min_area: int = 800,
        max_area: int = 80_000,
        min_aspect: float = 0.15,
        max_aspect: float = 1.2,
        kernel: tuple[int, int] = (5, 5),
        diff_thresh: int = 25,
    ):
        self.prev_gray: np.ndarray | None = None
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.kernel = kernel
        self.diff_thresh = diff_thresh

    def detect(self, frame_bgr: np.ndarray) -> tuple[list[Box], np.ndarray]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.prev_gray is None:
            self.prev_gray = gray
            return [], np.zeros_like(gray)
        diff = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray
        _, binary = cv2.threshold(diff, self.diff_thresh, 255, cv2.THRESH_BINARY)
        boxes = boxes_from_mask(
            binary,
            self.min_area,
            self.max_area,
            self.min_aspect,
            self.max_aspect,
            self.kernel,
        )
        return boxes, binary


def iou(a: Box, b: Box) -> float:
    x1 = max(a.left, b.left)
    y1 = max(a.top, b.top)
    x2 = min(a.right, b.right)
    y2 = min(a.bottom, b.bottom)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union


def match_greedy(
    dets: list[Box], gts: list[Box], iou_thresh: float = 0.3
) -> tuple[int, int, float]:
    """Return (tp, fp, mean_iou_of_matches)."""
    used_gt: set[int] = set()
    tp = 0
    ious: list[float] = []
    for d in dets:
        best_i, best_v = -1, 0.0
        for j, g in enumerate(gts):
            if j in used_gt:
                continue
            v = iou(d, g)
            if v > best_v:
                best_v, best_i = v, j
        if best_i >= 0 and best_v >= iou_thresh:
            tp += 1
            used_gt.add(best_i)
            ious.append(best_v)
    fp = len(dets) - tp
    mean_iou = float(np.mean(ious)) if ious else 0.0
    return tp, fp, mean_iou
