"""Train the BiLSTM team-id denoiser with per-frame supervision.

Pseudo-label: for each frame in a track, the per-frame k-means cluster ID mapped
through the cluster->team table. These per-frame labels are noisy (a blue jersey frame
with hardwood background may land in cluster A). The BiLSTM's job is to use temporal
context to predict each frame's label more cleanly than per-frame k-means can, and the
per-track team label at inference is the argmax of mean-softmax across frames.
"""

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
from src.team_model import TeamBiLSTM


def _load_position_features(seq_dir: Path) -> dict[tuple[int, int], np.ndarray]:
    """Per (track_id, frame) -> normalized (cx, cy, w, h)."""
    tracks = pd.read_parquet(seq_dir / "tracks_yolo.parquet")
    pos: dict[tuple[int, int], np.ndarray] = {}
    W = float(config.SEQ_WIDTH)
    H = float(config.SEQ_HEIGHT)
    for r in tracks.itertuples(index=False):
        cx = (r.left + r.width / 2.0) / W
        cy = (r.top + r.height / 2.0) / H
        w = r.width / W
        h = r.height / H
        pos[(int(r.track_id), int(r.frame))] = np.array([cx, cy, w, h], dtype=np.float32)
    return pos

TEAM_TO_IDX = {"A": 0, "B": 1, "other": 2}
IDX_TO_TEAM = {v: k for k, v in TEAM_TO_IDX.items()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_per_frame_data(
    seq_dir: Path, max_seq_len: int = config.LSTM_MAX_SEQ_LEN
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], list[int]]:
    """Returns (sequences, per_frame_labels, track_lengths, track_ids)."""
    data = np.load(seq_dir / "embeddings.npz")
    track_ids = data["track_ids"]
    frame_ids = data["frame_ids"]
    embeddings = data["embeddings"]
    cluster_ids = np.load(seq_dir / "kmeans_cluster_ids.npy")

    cluster_to_team_raw = json.loads((seq_dir / "cluster_to_team.json").read_text())
    cluster_to_team_idx = {
        int(k): TEAM_TO_IDX[v] for k, v in cluster_to_team_raw.items() if v in TEAM_TO_IDX
    }

    pos_features = _load_position_features(seq_dir) if config.LSTM_USE_POSITION else {}

    by_track: dict[int, list[tuple[int, np.ndarray, int]]] = defaultdict(list)
    for tid, fid, emb, cid in zip(
        track_ids.tolist(), frame_ids.tolist(), embeddings, cluster_ids.tolist()
    ):
        team_idx = cluster_to_team_idx.get(int(cid))
        if team_idx is None:
            continue
        if config.LSTM_USE_POSITION:
            p = pos_features.get((int(tid), int(fid)))
            if p is None:
                continue
            emb = np.concatenate([emb, p], axis=0)
        by_track[int(tid)].append((int(fid), emb, int(team_idx)))

    sequences: list[np.ndarray] = []
    per_frame_labels: list[np.ndarray] = []
    lengths: list[int] = []
    out_tids: list[int] = []

    for tid, entries in by_track.items():
        entries.sort(key=lambda x: x[0])
        seq = np.stack([e for _, e, _ in entries], axis=0).astype(np.float32)
        labs = np.array([l for _, _, l in entries], dtype=np.int64)

        if seq.shape[0] > max_seq_len:
            idxs = np.linspace(0, seq.shape[0] - 1, max_seq_len, dtype=int)
            seq = seq[idxs]
            labs = labs[idxs]

        sequences.append(seq)
        per_frame_labels.append(labs)
        lengths.append(seq.shape[0])
        out_tids.append(tid)

    return sequences, per_frame_labels, lengths, out_tids


class TrackDataset(Dataset):
    def __init__(self, sequences, frame_labels, lengths, track_ids):
        self.sequences = sequences
        self.frame_labels = frame_labels
        self.lengths = lengths
        self.track_ids = track_ids

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.sequences[idx]),
            torch.from_numpy(self.frame_labels[idx]),
            self.lengths[idx],
            self.track_ids[idx],
        )


def collate(batch):
    seqs = [b[0] for b in batch]
    flabs = [b[1] for b in batch]
    lengths = torch.tensor([b[2] for b in batch], dtype=torch.long)
    track_ids = torch.tensor([b[3] for b in batch], dtype=torch.long)

    max_len = int(lengths.max())
    dim = seqs[0].shape[1]
    padded = torch.zeros(len(seqs), max_len, dim, dtype=torch.float32)
    pad_labels = torch.full((len(seqs), max_len), -100, dtype=torch.long)  # -100 = CE ignore
    for i, (s, l) in enumerate(zip(seqs, flabs)):
        padded[i, : s.shape[0]] = s
        pad_labels[i, : l.shape[0]] = l
    return padded, pad_labels, lengths, track_ids


def train_model(
    seq_dir: Path,
    epochs: int = config.LSTM_EPOCHS,
    lr: float = config.LSTM_LR,
    hidden: int = config.LSTM_HIDDEN,
    seed: int = config.LSTM_SEED,
    batch_size: int = 32,
    holdout_track_ids: set[int] | None = None,
) -> tuple[TeamBiLSTM, dict]:
    set_seed(seed)
    sequences, frame_labels, lengths, tids = load_per_frame_data(seq_dir)

    if holdout_track_ids:
        kept = [
            (s, fl, ln, t)
            for s, fl, ln, t in zip(sequences, frame_labels, lengths, tids)
            if t not in holdout_track_ids
        ]
        sequences = [k[0] for k in kept]
        frame_labels = [k[1] for k in kept]
        lengths = [k[2] for k in kept]
        tids = [k[3] for k in kept]

    dataset = TrackDataset(sequences, frame_labels, lengths, tids)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

    model = TeamBiLSTM(input_dim=sequences[0].shape[1], hidden_dim=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss(ignore_index=-100)  # padding mask

    history: list[dict] = []
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_valid = 0
        for padded, pad_labels, lens, _ in loader:
            logits = model(padded, lens)  # (B, T, 3)
            # truncate labels to logits' T (which might be shorter than max in batch
            # after pack/unpack); align by min length
            T = min(logits.size(1), pad_labels.size(1))
            loss = ce(
                logits[:, :T].reshape(-1, 3),
                pad_labels[:, :T].reshape(-1),
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

            preds = logits[:, :T].argmax(dim=-1)
            mask = pad_labels[:, :T] != -100
            total_correct += int(((preds == pad_labels[:, :T]) & mask).sum())
            total_valid += int(mask.sum())
            total_loss += float(loss.detach()) * int(mask.sum())

        avg_loss = total_loss / max(total_valid, 1)
        frame_acc = total_correct / max(total_valid, 1)
        history.append(
            {"epoch": ep + 1, "loss": round(avg_loss, 4),
             "frame_acc_vs_pseudo": round(frame_acc, 4)}
        )

    return model, {
        "epochs": epochs,
        "n_train_tracks": len(sequences),
        "supervision": "per-frame cluster id (mapped to team via cluster_to_team.json)",
        "history": history,
    }


def predict(model: TeamBiLSTM, seq_dir: Path) -> dict[int, dict]:
    """Per-track team via mean-softmax over per-frame logits."""
    sequences, frame_labels, lengths, tids = load_per_frame_data(seq_dir)
    dataset = TrackDataset(sequences, frame_labels, lengths, tids)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate)

    model.eval()
    out: dict[int, dict] = {}
    with torch.no_grad():
        for padded, pad_labels, lens, batch_tids in loader:
            logits = model(padded, lens)  # (B, T, 3)
            probs = torch.softmax(logits, dim=-1)
            T = min(probs.size(1), pad_labels.size(1))
            mask = (pad_labels[:, :T] != -100).float().unsqueeze(-1)
            track_probs = (probs[:, :T] * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            preds = track_probs.argmax(dim=-1)
            for i, tid in enumerate(batch_tids.tolist()):
                # majority of per-frame pseudo-label (just for diagnostic)
                pl = pad_labels[i, : int(lens[i])].tolist()
                pseudo_majority = max(set(pl), key=pl.count) if pl else -1
                out[int(tid)] = {
                    "team": IDX_TO_TEAM[int(preds[i])],
                    "probabilities": {
                        "A": round(float(track_probs[i, 0]), 4),
                        "B": round(float(track_probs[i, 1]), 4),
                        "other": round(float(track_probs[i, 2]), 4),
                    },
                    "n_frames": int(lens[i]),
                    "per_frame_pseudo_majority": (
                        IDX_TO_TEAM[pseudo_majority] if pseudo_majority >= 0 else None
                    ),
                }
    return out
