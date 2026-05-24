"""Train end-to-end CNN encoder + BiLSTM with per-track team supervision."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import config
from src.team_model_e2e import TeamE2E

TEAM_TO_IDX = {"A": 0, "B": 1, "other": 2}
IDX_TO_TEAM = {v: k for k, v in TEAM_TO_IDX.items()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_e2e_data(seq_dir: Path):
    crops_npz = np.load(seq_dir / "crops.npz")
    crops = crops_npz["crops"]  # (N, H, W, 3) uint8
    track_ids = crops_npz["track_ids"]
    frame_ids = crops_npz["frame_ids"]

    baseline = json.loads((seq_dir / "baseline_labels.json").read_text())
    tracks_df = pd.read_parquet(seq_dir / "tracks_yolo.parquet")
    W, H = float(config.SEQ_WIDTH), float(config.SEQ_HEIGHT)
    pos_lookup = {
        (int(r.track_id), int(r.frame)): np.array(
            [(r.left + r.width / 2) / W, (r.top + r.height / 2) / H, r.width / W, r.height / H],
            dtype=np.float32,
        )
        for r in tracks_df.itertuples(index=False)
    }

    by_track = defaultdict(list)
    for i, (tid, fid) in enumerate(zip(track_ids.tolist(), frame_ids.tolist())):
        by_track[int(tid)].append((int(fid), i))

    seqs, positions, labels, lengths, out_tids = [], [], [], [], []
    for tid, entries in by_track.items():
        team = baseline.get(str(tid), {}).get("team")
        if team not in TEAM_TO_IDX:
            continue
        entries.sort(key=lambda x: x[0])
        if len(entries) > config.LSTM_MAX_SEQ_LEN:
            idxs = np.linspace(0, len(entries) - 1, config.LSTM_MAX_SEQ_LEN, dtype=int)
            entries = [entries[i] for i in idxs]
        crop_idxs = [e[1] for e in entries]
        pos = np.stack([pos_lookup[(tid, e[0])] for e in entries], axis=0)
        seqs.append(crops[crop_idxs])  # (T, H, W, 3) uint8
        positions.append(pos)
        labels.append(TEAM_TO_IDX[team])
        lengths.append(len(entries))
        out_tids.append(tid)

    return seqs, positions, labels, lengths, out_tids


class TrackDataset(Dataset):
    def __init__(self, seqs, positions, labels, lengths, track_ids):
        self.seqs = seqs
        self.positions = positions
        self.labels = labels
        self.lengths = lengths
        self.track_ids = track_ids

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        crops = torch.from_numpy(self.seqs[i]).float() / 255.0  # (T, H, W, 3)
        crops = crops.permute(0, 3, 1, 2).contiguous()  # (T, 3, H, W)
        return crops, torch.from_numpy(self.positions[i]), self.lengths[i], self.labels[i], self.track_ids[i]


def collate(batch):
    crops_list = [b[0] for b in batch]
    pos_list = [b[1] for b in batch]
    lengths = torch.tensor([b[2] for b in batch], dtype=torch.long)
    labels = torch.tensor([b[3] for b in batch], dtype=torch.long)
    track_ids = torch.tensor([b[4] for b in batch], dtype=torch.long)

    T_max = int(lengths.max())
    _, C, H, W = crops_list[0].shape
    D = pos_list[0].shape[1]

    padded_crops = torch.zeros(len(batch), T_max, C, H, W, dtype=torch.float32)
    padded_pos = torch.zeros(len(batch), T_max, D, dtype=torch.float32)
    for i, (c, p) in enumerate(zip(crops_list, pos_list)):
        padded_crops[i, : c.size(0)] = c
        padded_pos[i, : p.size(0)] = p
    return padded_crops, padded_pos, lengths, labels, track_ids


def train_model(
    seq_dir: Path,
    holdout_track_ids: set[int] | None = None,
    epochs: int = config.E2E_EPOCHS,
    batch_size: int = 16,
) -> tuple[TeamE2E, dict]:
    set_seed(config.LSTM_SEED)
    seqs, positions, labels, lengths, tids = load_e2e_data(seq_dir)
    if holdout_track_ids:
        keep = [i for i, t in enumerate(tids) if t not in holdout_track_ids]
        seqs = [seqs[i] for i in keep]
        positions = [positions[i] for i in keep]
        labels = [labels[i] for i in keep]
        lengths = [lengths[i] for i in keep]
        tids = [tids[i] for i in keep]

    # Class weights inverse-proportional to support to fight A-dominance
    label_counts = [labels.count(i) for i in range(3)]
    total = sum(label_counts)
    class_weights = torch.tensor(
        [total / (3 * max(c, 1)) for c in label_counts], dtype=torch.float32
    )

    dataset = TrackDataset(seqs, positions, labels, lengths, tids)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

    model = TeamE2E(
        feature_dim=config.CNN_FEATURE_DIM,
        hidden_dim=config.LSTM_HIDDEN,
        dropout=config.E2E_DROPOUT,
    )
    opt = torch.optim.Adam(
        model.parameters(), lr=config.E2E_LR, weight_decay=config.E2E_WEIGHT_DECAY
    )
    ce = nn.CrossEntropyLoss(weight=class_weights)

    history = []
    for ep in range(epochs):
        model.train()
        total_loss, total_correct, total = 0.0, 0, 0
        for crops_b, pos_b, lens_b, labs_b, _ in loader:
            logits = model(crops_b, pos_b, lens_b)
            loss = ce(logits, labs_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.detach()) * labs_b.size(0)
            total_correct += int((logits.argmax(-1) == labs_b).sum())
            total += labs_b.size(0)
        history.append({
            "epoch": ep + 1,
            "loss": round(total_loss / max(total, 1), 4),
            "train_acc_vs_pseudo": round(total_correct / max(total, 1), 4),
        })

    return model, {
        "epochs": epochs,
        "n_train_tracks": len(seqs),
        "supervision": "per-track baseline label (class-weighted CE)",
        "class_weights": class_weights.tolist(),
        "history": history,
    }


def predict(model: TeamE2E, seq_dir: Path) -> dict[int, dict]:
    seqs, positions, labels, lengths, tids = load_e2e_data(seq_dir)
    dataset = TrackDataset(seqs, positions, labels, lengths, tids)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate)

    model.eval()
    out = {}
    with torch.no_grad():
        for crops_b, pos_b, lens_b, labs_b, tids_b in loader:
            logits = model(crops_b, pos_b, lens_b)
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(-1)
            for i, tid in enumerate(tids_b.tolist()):
                out[int(tid)] = {
                    "team": IDX_TO_TEAM[int(preds[i])],
                    "probabilities": {
                        "A": round(float(probs[i, 0]), 4),
                        "B": round(float(probs[i, 1]), 4),
                        "other": round(float(probs[i, 2]), 4),
                    },
                    "n_frames": int(lens_b[i]),
                    "pseudo_label": IDX_TO_TEAM[int(labs_b[i])],
                }
    return out
