"""Small BiLSTM that maps a per-track LAB-histogram embedding sequence to a team label."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TeamBiLSTM(nn.Module):
    """1-layer BiLSTM -> masked mean-pool -> MLP -> 3-class softmax."""

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 64,
        n_classes: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x:       (B, T, input_dim)  padded
        lengths: (B,)              true track lengths

        Returns per-frame logits of shape (B, T, n_classes). Aggregation across timesteps
        (mean of softmax for the per-track decision) happens in inference code.
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        # out: (B, T_max, 2 * hidden_dim)

        h = F.relu(self.fc1(self.dropout(out)))   # (B, T_max, hidden_dim)
        return self.fc2(h)                        # (B, T_max, n_classes)
