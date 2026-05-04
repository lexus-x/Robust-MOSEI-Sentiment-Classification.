"""Information Bottleneck encoders for modality compression.

Each modality is encoded into a stochastic latent representation Z_m that
retains only task-relevant information, following the IB principle:

    min I(Z_m; X_m) - β · I(Z_m; Y)

We approximate I(Z_m; X_m) via a variational upper bound using the
reparameterization trick (Kingma & Welling, 2014).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .common import SinusoidalPositionalEncoding


class IBEncoder(nn.Module):
    """Variational Information Bottleneck encoder for a single modality.

    Projects input features to a stochastic latent space where the
    KL divergence against a unit Gaussian serves as a tractable upper
    bound on the mutual information I(Z; X).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Deterministic feature extractor
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Stochastic bottleneck: predict mean and log-variance
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        self.position = SinusoidalPositionalEncoding(latent_dim)

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Sample z ~ N(mu, sigma^2) using the reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu  # deterministic at test time by default

    def kl_divergence(
        self, mu: torch.Tensor, logvar: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """KL(q(z|x) || p(z)) where p(z) = N(0, I).

        Returns scalar KL averaged over valid (non-padding) positions.
        """
        # Per-position KL: 0.5 * sum_d(-1 - logvar_d + mu_d^2 + exp(logvar_d))
        kl_per_position = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)
        # Mask out padding positions
        weights = mask.float()
        denom = weights.sum().clamp_min(1.0)
        return (kl_per_position * weights).sum() / denom

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode modality features through the information bottleneck.

        Args:
            features: [batch, seq_len, input_dim]
            mask: [batch, seq_len] boolean

        Returns:
            z: [batch, seq_len, latent_dim] — compressed representation
            kl: scalar — KL divergence (upper bound on I(Z; X))
        """
        h = self.projection(features)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)

        z = self.reparameterize(mu, logvar)
        z = self.position(z)

        kl = self.kl_divergence(mu, logvar, mask)
        return z, kl


class MultiModalIBEncoder(nn.Module):
    """Parallel IB encoders for text, audio, and vision modalities."""

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        vision_dim: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.text_encoder = IBEncoder(text_dim, hidden_dim, latent_dim, dropout)
        self.audio_encoder = IBEncoder(audio_dim, hidden_dim, latent_dim, dropout)
        self.vision_encoder = IBEncoder(vision_dim, hidden_dim, latent_dim, dropout)

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        vision: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode all three modalities.

        Returns:
            z_text, z_audio, z_vision: compressed latent sequences
            total_kl: sum of per-modality KL divergences
        """
        z_text, kl_text = self.text_encoder(text, mask)
        z_audio, kl_audio = self.audio_encoder(audio, mask)
        z_vision, kl_vision = self.vision_encoder(vision, mask)

        total_kl = kl_text + kl_audio + kl_vision
        return z_text, z_audio, z_vision, total_kl
