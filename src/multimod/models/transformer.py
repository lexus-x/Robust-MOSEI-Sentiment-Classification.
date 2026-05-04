"""Cross-modal transformer with optional modality gating."""

from __future__ import annotations

import copy

import torch
from torch import nn

from .common import MLPProjection, SequenceClassifierHead, SinusoidalPositionalEncoding, masked_sequence_mean


class CrossModalTransformer(nn.Module):
    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        vision_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        num_classes: int,
        dropout: float,
        use_gating: bool = False,
    ) -> None:
        super().__init__()
        self.use_gating = use_gating
        self.text_proj = MLPProjection(text_dim, hidden_dim, dropout)
        self.audio_proj = MLPProjection(audio_dim, hidden_dim, dropout)
        self.vision_proj = MLPProjection(vision_dim, hidden_dim, dropout)
        self.position = SinusoidalPositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Pre-norm encoder layers are incompatible with PyTorch's nested-tensor fast path.
        self.text_encoder = nn.TransformerEncoder(
            copy.deepcopy(encoder_layer),
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.audio_encoder = nn.TransformerEncoder(
            copy.deepcopy(encoder_layer),
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.vision_encoder = nn.TransformerEncoder(
            copy.deepcopy(encoder_layer),
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.audio_cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.vision_cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.cross_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.cross_ffn_norm = nn.LayerNorm(hidden_dim)

        if use_gating:
            self.gating = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 3),
            )
        else:
            self.gating = None

        self.classifier = SequenceClassifierHead(hidden_dim * 3, hidden_dim, num_classes, dropout)

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        vision: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        padding_mask = ~mask
        text_encoded = self.text_encoder(
            self.position(self.text_proj(text)),
            src_key_padding_mask=padding_mask,
        )
        audio_encoded = self.audio_encoder(
            self.position(self.audio_proj(audio)),
            src_key_padding_mask=padding_mask,
        )
        vision_encoded = self.vision_encoder(
            self.position(self.vision_proj(vision)),
            src_key_padding_mask=padding_mask,
        )

        audio_context, audio_attention = self.audio_cross_attention(
            query=text_encoded,
            key=audio_encoded,
            value=audio_encoded,
            key_padding_mask=padding_mask,
            need_weights=True,
        )
        vision_context, vision_attention = self.vision_cross_attention(
            query=text_encoded,
            key=vision_encoded,
            value=vision_encoded,
            key_padding_mask=padding_mask,
            need_weights=True,
        )

        fused_tokens = self.cross_norm(text_encoded + audio_context + vision_context)
        fused_tokens = self.cross_ffn_norm(fused_tokens + self.cross_ffn(fused_tokens))

        fused_summary = masked_sequence_mean(fused_tokens, mask)
        audio_summary = masked_sequence_mean(audio_encoded, mask)
        vision_summary = masked_sequence_mean(vision_encoded, mask)

        stacked = torch.stack([fused_summary, audio_summary, vision_summary], dim=1)
        if self.gating is None:
            gates = torch.ones(
                stacked.size(0),
                stacked.size(1),
                device=stacked.device,
                dtype=stacked.dtype,
            )
        else:
            gates = torch.sigmoid(self.gating(torch.cat([fused_summary, audio_summary, vision_summary], dim=-1)))
        weighted = stacked * gates.unsqueeze(-1)
        logits = self.classifier(weighted.flatten(start_dim=1))
        diagnostics = {
            "gates": gates,
            "audio_attention_mean": audio_attention.mean(dim=1),
            "vision_attention_mean": vision_attention.mean(dim=1),
        }
        return logits, diagnostics
