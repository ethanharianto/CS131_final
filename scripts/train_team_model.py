"""Train the BiLSTM team-id denoiser and save checkpoint + predictions."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

import config
from src.team_train import predict, train_model


def main() -> None:
    seq = config.SEQUENCE_NAME
    out_dir = config.OUTPUT_ROOT / seq

    # Hold out hand-labeled tracks from training so the model doesn't fit to
    # pseudo-labels that overlap with the eval set.
    gt_path = out_dir / "team_gt.json"
    holdout: set[int] = set()
    if gt_path.exists():
        gt = json.loads(gt_path.read_text())
        holdout = {int(k) for k, v in gt.items() if v.get("team") in ("A", "B", "other")}
        print(f"Holding out {len(holdout)} hand-labeled tracks from training.")

    t0 = time.time()
    model, train_metrics = train_model(out_dir, holdout_track_ids=holdout)
    train_time = time.time() - t0
    train_metrics["train_seconds"] = round(train_time, 1)

    ckpt = out_dir / "team_model.pt"
    torch.save({"state_dict": model.state_dict(), "config": {
        "input_dim": 256, "hidden_dim": config.LSTM_HIDDEN,
    }}, ckpt)

    preds = predict(model, out_dir)
    (out_dir / "learned_labels.json").write_text(
        json.dumps(preds, indent=2, sort_keys=True)
    )
    (out_dir / "team_model_metrics.json").write_text(json.dumps(train_metrics, indent=2))

    print(f"Training done in {train_time:.1f}s on {train_metrics['n_train_tracks']} tracks.")
    last = train_metrics["history"][-1]
    acc_key = next((k for k in last if k.startswith(("frame_acc", "train_acc"))), None)
    acc_str = f", {acc_key}={last[acc_key]}" if acc_key else ""
    print(f"Final epoch: loss={last['loss']}{acc_str}")
    print(f"Checkpoint: {ckpt}")
    print(f"Predictions: {out_dir / 'learned_labels.json'}")


if __name__ == "__main__":
    main()
