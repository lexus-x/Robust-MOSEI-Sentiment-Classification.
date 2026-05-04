"""Mamba-based sequence encoder for multimodal sentiment.

Replaces Transformer self-attention with Mamba SSM blocks for O(n)
linear-complexity sequence modeling. Falls back to a lightweight
1D-conv + GRU encoder if the mamba-ssm package is not installed.

References:
  - Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective
    State Spaces", ICLR 2024
  - MSAmba (AAAI 2025): Mamba for multimodal sentiment analysis
  - TF-Mamba: Text-enhanced Fusion Mamba for missing modalities

The fallback ensures the codebase always runs, even without CUDA
or the mamba-ssm wheel.
"""

from __future__ import annotations

import warnings

import torch
from torch import nn

try:
    from mamba_ssm import Mamba

    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    warnings.warn(
        "mamba-ssm is not installed. MambaEncoder will use a conv+GRU fallback. "
        "Results from 'eidmsa_mamba' experiments do NOT use real Mamba SSM. "
        "Install with: pip install mamba-ssm",
        RuntimeWarning,
        stacklevel=2,
    )


class MambaBlock(nn.Module):
    """Single Mamba SSM block with pre-norm and residual connection.

    If mamba-ssm is not installed, falls back to a 1D-conv + GRU block
    that has similar inductive bias (local + sequential).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if MAMBA_AVAILABLE:
            self.ssm = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.use_mamba = True
        else:
            # Fallback: 1D causal conv + bidirectional GRU
            self.conv = nn.Conv1d(
                d_model, d_model, kernel_size=d_conv,
                padding=d_conv - 1, groups=d_model,
            )
            self.gru = nn.GRU(
                d_model, d_model // 2,
                batch_first=True, bidirectional=True,
            )
            self.proj = nn.Linear(d_model, d_model)
            self.use_mamba = False

    @staticmethod
    def _left_to_right_padded(
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Move valid tokens to the front so RNN packing matches left-padded inputs."""
        batch_size, seq_len, hidden_dim = x.shape
        right_padded = x.new_zeros(batch_size, seq_len, hidden_dim)
        lengths = mask.sum(dim=1, dtype=torch.long)
        for batch_index, length in enumerate(lengths.tolist()):
            if length <= 0:
                continue
            right_padded[batch_index, :length] = x[batch_index, mask[batch_index]]
        return right_padded, lengths

    @staticmethod
    def _right_to_left_padded(
        x: torch.Tensor,
        mask: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Restore the original left-padded alignment after packed RNN processing."""
        left_padded = x.new_zeros(x.shape)
        for batch_index, length in enumerate(lengths.tolist()):
            if length <= 0:
                continue
            left_padded[batch_index, -length:] = x[batch_index, :length]
        return left_padded

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: [batch, seq_len] boolean (unused by Mamba, used by fallback)

        Returns:
            [batch, seq_len, d_model]
        """
        residual = x
        x = self.norm(x)

        if self.use_mamba:
            x = self.ssm(x)
        else:
            if mask is not None:
                x, lengths = self._left_to_right_padded(x, mask)
            else:
                lengths = None

            # Fallback path
            # 1D conv (causal)
            conv_in = x.transpose(1, 2)  # [batch, d_model, seq_len]
            conv_out = self.conv(conv_in)[:, :, :x.size(1)]  # trim to original length
            x = conv_out.transpose(1, 2)  # [batch, seq_len, d_model]

            if mask is not None:
                packed = nn.utils.rnn.pack_padded_sequence(
                    x,
                    lengths.cpu(),
                    batch_first=True,
                    enforce_sorted=False,
                )
                packed_out, _ = self.gru(packed)
                x, _ = nn.utils.rnn.pad_packed_sequence(
                    packed_out,
                    batch_first=True,
                    total_length=x.size(1),
                )
            else:
                x, _ = self.gru(x)
            x = self.proj(x)

            if mask is not None:
                x = self._right_to_left_padded(x, mask, lengths)

        return residual + self.dropout(x)


class MambaEncoder(nn.Module):
    """Stacked Mamba blocks as a sequence encoder.

    Drop-in replacement for nn.TransformerEncoder — same input/output
    shapes but with O(n) complexity instead of O(n²).
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            src_key_padding_mask: [batch, seq_len] — True for padding positions

        Returns:
            [batch, seq_len, d_model]
        """
        # Convert padding mask to boolean mask (True = valid)
        mask = None
        if src_key_padding_mask is not None:
            mask = ~src_key_padding_mask

        for layer in self.layers:
            x = layer(x, mask=mask)

        return self.final_norm(x)


def is_mamba_available() -> bool:
    """Check if the mamba-ssm package is installed."""
    return MAMBA_AVAILABLE
