"""Greedy IoU tracker (v0) for detection boxes."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.detect import iou
from src.mot_io import Box


@dataclass
class Track:
    track_id: int
    box: Box
    age: int = 0
    missed: int = 0


@dataclass
class GreedyTracker:
    iou_thresh: float = 0.3
    max_missed: int = 10
    next_id: int = 1
    tracks: list[Track] = field(default_factory=list)

    def update(self, detections: list[Box]) -> list[tuple[int, Box]]:
        for t in self.tracks:
            t.missed += 1

        unmatched_dets = list(range(len(detections)))
        matched_pairs: list[tuple[int, int, float]] = []

        for ti, tr in enumerate(self.tracks):
            best_di, best_iou = -1, 0.0
            for di in unmatched_dets:
                v = iou(tr.box, detections[di])
                if v > best_iou:
                    best_iou, best_di = v, di
            if best_di >= 0 and best_iou >= self.iou_thresh:
                matched_pairs.append((ti, best_di, best_iou))
                unmatched_dets.remove(best_di)

        matched_pairs.sort(key=lambda x: -x[2])
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for ti, di, _ in matched_pairs:
            if ti in used_tracks or di in used_dets:
                continue
            self.tracks[ti].box = detections[di]
            self.tracks[ti].missed = 0
            self.tracks[ti].age += 1
            used_tracks.add(ti)
            used_dets.add(di)

        for di in unmatched_dets:
            self.tracks.append(
                Track(track_id=self.next_id, box=detections[di], age=1, missed=0)
            )
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return [(t.track_id, t.box) for t in self.tracks if t.missed == 0]
