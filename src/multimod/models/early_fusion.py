"""Early-fusion baseline for multimodal sentiment classification."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .common import MLPProjection, SequenceClassifierHead, masked_sequence_mean


class EarlyFusionClassifier(nn.Module):
    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        vision_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.text_proj = MLPProjection(text_dim, hidden_dim, dropout)
        self.audio_proj = MLPProjection(audio_dim, hidden_dim, dropout)
        self.vision_proj = MLPProjection(vision_dim, hidden_dim, dropout)
        self.encoder = nn.GRU(
            input_size=hidden_dim * 3,
            hidden_size=hidden_dim,
            num_layers=1,
            dropout=0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = SequenceClassifierHead(hidden_dim * 2, hidden_dim, num_classes, dropout)

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        vision: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        fused = torch.cat(
            [
                self.text_proj(text),
                self.audio_proj(audio),
                self.vision_proj(vision),
            ],
            dim=-1,
        )
        lengths = mask.sum(dim=1).cpu()
        packed = pack_padded_sequence(fused, lengths, batch_first=True, enforce_sorted=False)
        packed_outputs, _ = self.encoder(packed)
        encoded, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=fused.size(1),
        )
        pooled = masked_sequence_mean(encoded, mask)
        logits = self.classifier(pooled)
        return logits, {}
