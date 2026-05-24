"""Compute team-purity of the k-means majority-vote baseline against hand-labeled GT."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.eval_team import gt_team_distribution, team_purity


def main() -> None:
    seq = config.SEQUENCE_NAME
    out_dir = config.OUTPUT_ROOT / seq

    gt_path = out_dir / "team_gt.json"
    base_path = out_dir / "baseline_labels.json"
    if not gt_path.exists():
        raise SystemExit(f"Missing GT labels: {gt_path}; run scripts/label_teams.py first")
    if not base_path.exists():
        raise SystemExit(f"Missing baseline: {base_path}; run scripts/run_baseline.py first")

    gt_raw = json.loads(gt_path.read_text())
    base_raw = json.loads(base_path.read_text())

    gt = {int(k): v["team"] for k, v in gt_raw.items() if v.get("team") in ("A", "B", "other")}
    predicted = {int(k): v["team"] for k, v in base_raw.items()}

    unlabeled = sum(1 for v in gt_raw.values() if v.get("team") is None)
    if unlabeled:
        print(f"Warning: {unlabeled} tracks in team_gt.json still null; skipping those.")

    result = team_purity(predicted, gt)
    result["gt_distribution"] = gt_team_distribution(gt)

    out_path = out_dir / "baseline_purity.json"
    out_path.write_text(json.dumps(result, indent=2))

    print(f"Baseline team purity vs {len(gt)} hand-labeled tracks:")
    print(f"  overall_purity: {result['overall_purity']}")
    print(f"  n_correct: {result['n_correct']} / {result['n_labeled_tracks']}")
    print(f"  per_class: {json.dumps(result['per_class'], indent=2)}")
    print(f"  confusion: {json.dumps(result['confusion_matrix'], indent=2)}")
    print(f"  gt_distribution: {result['gt_distribution']}")
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
