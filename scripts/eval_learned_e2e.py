"""Evaluate the end-to-end model against hand-labeled GT, side-by-side with baseline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src.eval_team import team_purity


def main() -> None:
    seq = config.SEQUENCE_NAME
    out_dir = config.OUTPUT_ROOT / seq

    gt = json.loads((out_dir / "team_gt.json").read_text())
    gt_clean = {int(k): v["team"] for k, v in gt.items() if v.get("team") in ("A", "B", "other")}

    base = json.loads((out_dir / "baseline_labels.json").read_text())
    base_pred = {int(k): v["team"] for k, v in base.items()}

    e2e = json.loads((out_dir / "learned_labels_e2e.json").read_text())
    e2e_pred = {int(k): v["team"] for k, v in e2e.items()}

    base_r = team_purity(base_pred, gt_clean)
    e2e_r = team_purity(e2e_pred, gt_clean)

    (out_dir / "learned_purity_e2e.json").write_text(json.dumps(e2e_r, indent=2))

    print(f"Team-purity on {len(gt_clean)} hand-labeled tracks:")
    print(f"  baseline (k-means + per-track):     "
          f"{base_r['overall_purity']:.4f} ({base_r['n_correct']}/{base_r['n_labeled_tracks']})")
    print(f"  end-to-end CNN+BiLSTM (M3):         "
          f"{e2e_r['overall_purity']:.4f} ({e2e_r['n_correct']}/{e2e_r['n_labeled_tracks']})")
    print(f"  delta vs baseline:                  "
          f"{e2e_r['overall_purity'] - base_r['overall_purity']:+.4f}")
    print()
    print("Per-class (e2e):")
    print(json.dumps(e2e_r["per_class"], indent=2))
    print("Confusion (e2e):")
    print(json.dumps(e2e_r["confusion_matrix"], indent=2))


if __name__ == "__main__":
    main()
