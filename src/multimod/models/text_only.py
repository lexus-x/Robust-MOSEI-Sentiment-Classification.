"""Text-only baseline for sentiment classification."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .common import MLPProjection, SequenceClassifierHead, masked_sequence_mean


class TextOnlyClassifier(nn.Module):
    def __init__(self, text_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.text_proj = MLPProjection(text_dim, hidden_dim, dropout)
        self.encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            dropout=0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = SequenceClassifierHead(hidden_dim, hidden_dim, num_classes, dropout)

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        vision: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del audio, vision
        projected = self.text_proj(text)
        lengths = mask.sum(dim=1).cpu()
        packed = pack_padded_sequence(projected, lengths, batch_first=True, enforce_sorted=False)
        packed_outputs, _ = self.encoder(packed)
        encoded, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=projected.size(1),
        )
        pooled = masked_sequence_mean(encoded, mask)
        logits = self.classifier(pooled)
        return logits, {}
