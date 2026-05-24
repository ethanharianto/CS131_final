"""Evaluate the learned BiLSTM team-id predictions against hand-labeled GT."""

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
    learned_path = out_dir / "learned_labels.json"
    base_path = out_dir / "baseline_labels.json"
    if not learned_path.exists():
        raise SystemExit(f"Missing learned labels: {learned_path}; run scripts/train_team_model.py first")

    gt_raw = json.loads(gt_path.read_text())
    learned_raw = json.loads(learned_path.read_text())
    base_raw = json.loads(base_path.read_text())

    gt = {int(k): v["team"] for k, v in gt_raw.items() if v.get("team") in ("A", "B", "other")}
    learned = {int(k): v["team"] for k, v in learned_raw.items()}
    baseline = {int(k): v["team"] for k, v in base_raw.items()}

    learned_result = team_purity(learned, gt)
    baseline_result = team_purity(baseline, gt)

    learned_result["gt_distribution"] = gt_team_distribution(gt)
    (out_dir / "learned_purity.json").write_text(json.dumps(learned_result, indent=2))

    print(f"Team-purity on {len(gt)} hand-labeled tracks:")
    print(f"  baseline (k-means + per-track decision): "
          f"{baseline_result['overall_purity']:.4f} "
          f"({baseline_result['n_correct']}/{baseline_result['n_labeled_tracks']})")
    print(f"  learned  (BiLSTM):                       "
          f"{learned_result['overall_purity']:.4f} "
          f"({learned_result['n_correct']}/{learned_result['n_labeled_tracks']})")
    delta = learned_result["overall_purity"] - baseline_result["overall_purity"]
    print(f"  delta:                                   {delta:+.4f}")
    print()
    print("Per-class (learned):")
    print(json.dumps(learned_result["per_class"], indent=2))
    print("Confusion (learned):")
    print(json.dumps(learned_result["confusion_matrix"], indent=2))


if __name__ == "__main__":
    main()
