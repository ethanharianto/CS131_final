"""Train end-to-end CNN+BiLSTM team-id model and save checkpoint + predictions."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

import config
from src.team_train_e2e import predict, train_model


def main() -> None:
    seq = config.SEQUENCE_NAME
    out_dir = config.OUTPUT_ROOT / seq

    if not (out_dir / "crops.npz").exists():
        raise SystemExit("Missing crops.npz; run scripts/cache_crops.py first.")

    gt_path = out_dir / "team_gt.json"
    holdout = set()
    if gt_path.exists():
        gt = json.loads(gt_path.read_text())
        holdout = {int(k) for k, v in gt.items() if v.get("team") in ("A", "B", "other")}
        print(f"Holding out {len(holdout)} hand-labeled tracks from training.")

    t0 = time.time()
    model, metrics = train_model(out_dir, holdout_track_ids=holdout)
    metrics["train_seconds"] = round(time.time() - t0, 1)

    ckpt = out_dir / "team_model_e2e.pt"
    torch.save({"state_dict": model.state_dict()}, ckpt)

    preds = predict(model, out_dir)
    (out_dir / "learned_labels_e2e.json").write_text(
        json.dumps(preds, indent=2, sort_keys=True)
    )
    (out_dir / "team_model_e2e_metrics.json").write_text(json.dumps(metrics, indent=2))

    last = metrics["history"][-1]
    print(f"Training done in {metrics['train_seconds']}s on {metrics['n_train_tracks']} tracks.")
    print(f"Final epoch: loss={last['loss']}, train_acc_vs_pseudo={last['train_acc_vs_pseudo']}")
    print(f"Class weights (A, B, other): {metrics['class_weights']}")
    print(f"Checkpoint: {ckpt}")
    print(f"Predictions: {out_dir / 'learned_labels_e2e.json'}")


if __name__ == "__main__":
    main()
