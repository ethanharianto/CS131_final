"""End-to-end CNN encoder + BiLSTM for team-id from raw torso crops."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CropEncoder(nn.Module):
    """Tiny CNN: crop (3, 48, 24) -> feature (CNN_FEATURE_DIM,).

    Three conv blocks with BatchNorm, ReLU, MaxPool, ending in adaptive avg pool.
    Designed small (~30k params) for CPU training.
    """

    def __init__(self, feature_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 3, H, W) -- N can be batch * time flattened
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.gap(x).flatten(1)  # (N, 64)
        return self.fc(self.dropout(x))  # (N, feature_dim)


class TeamE2E(nn.Module):
    """CNN encoder per frame -> BiLSTM over time -> mean-pool -> MLP -> 3 classes.

    Per-track supervision: one team label per track. The CNN has freedom to learn
    discriminative features (jersey patterns, numbers, color saturation) beyond LAB
    histograms; the LSTM aggregates temporally.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 64,
        position_dim: int = 4,
        n_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = CropEncoder(feature_dim=feature_dim, dropout=dropout)
        self.lstm = nn.LSTM(
            input_size=feature_dim + position_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def forward(
        self,
        crops: torch.Tensor,  # (B, T, 3, H, W) padded
        positions: torch.Tensor,  # (B, T, position_dim)
        lengths: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        B, T = crops.size(0), crops.size(1)
        flat_crops = crops.reshape(B * T, *crops.shape[2:])
        feats = self.encoder(flat_crops).reshape(B, T, -1)
        x = torch.cat([feats, positions], dim=-1)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)

        max_t = out.size(1)
        idx = torch.arange(max_t, device=out.device).unsqueeze(0).expand(B, -1)
        mask = (idx < lengths.unsqueeze(1)).float().unsqueeze(-1)
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        h = F.relu(self.fc1(self.dropout(pooled)))
        return self.fc2(h)
