"""MOT Challenge format I/O for SportsMOT."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class Box:
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect(self) -> float:
        return self.width / max(self.height, 1)


def sequence_dir(data_root: Path, split: str, name: str) -> Path:
    return data_root / split / name


def image_dir(seq_dir: Path) -> Path:
    return seq_dir / "img1"


def gt_path(seq_dir: Path) -> Path:
    return seq_dir / "gt" / "gt.txt"


def list_frames(img_dir: Path) -> list[Path]:
    return sorted(img_dir.glob("*.jpg"))


def read_gt(path: Path) -> dict[int, list[Box]]:
    """Parse MOT gt.txt -> frame_id -> list of player boxes (class 1 only)."""
    by_frame: dict[int, list[Box]] = defaultdict(list)
    if not path.exists():
        return by_frame

    with path.open() as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            frame = int(parts[0])
            # SportsMOT: last field is class id; 1 = player on court
            cls = int(float(parts[8])) if len(parts) > 8 else 1
            if cls != 1:
                continue
            left, top, w, h = map(int, map(float, parts[2:6]))
            if w <= 0 or h <= 0:
                continue
            by_frame[frame].append(Box(left, top, w, h))
    return by_frame


def read_frame(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return img


def frame_index_from_name(path: Path) -> int:
    return int(path.stem)
