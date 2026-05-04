"""KAN-enhanced PID Decomposition module.

Replaces the MLP projection heads in PIDDecomposition with KANProjection
heads that use learnable B-spline activations for interpretable nonlinear
feature extraction.

Reference: Liu et al., "KAN: Kolmogorov-Arnold Networks", ICLR 2025
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ..utils import masked_mean
from .kan_layers import KANProjection
from .pid_decomposition import InfoNCEEstimator


class PIDDecompositionKAN(nn.Module):
    """PID Decomposition with KAN (B-spline) projection heads.

    Same interface as PIDDecomposition but uses KAN layers for:
      - Unique component extraction (per modality)
      - Redundancy extraction
      - Synergy extraction

    This provides inherent interpretability: the learned B-spline functions
    on each edge can be inspected to understand what nonlinear transforms
    the model applies to extract sentiment-relevant information.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        grid_size: int = 5,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # KAN-based unique extraction heads
        self.unique_text = KANProjection(
            latent_dim, hidden_dim, latent_dim, grid_size=grid_size, dropout=dropout
        )
        self.unique_audio = KANProjection(
            latent_dim, hidden_dim, latent_dim, grid_size=grid_size, dropout=dropout
        )
        self.unique_vision = KANProjection(
            latent_dim, hidden_dim, latent_dim, grid_size=grid_size, dropout=dropout
        )

        # KAN-based redundancy head
        self.redundancy = KANProjection(
            latent_dim * 3, hidden_dim, latent_dim, grid_size=grid_size, dropout=dropout
        )

        # Synergy: gate + KAN transform
        self.synergy_gate = nn.Sequential(
            nn.Linear(latent_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Sigmoid(),
        )
        self.synergy_transform = KANProjection(
            latent_dim, hidden_dim, latent_dim, grid_size=grid_size, dropout=dropout
        )

        # MI estimators (same as MLP version)
        self.mi_unique_text = InfoNCEEstimator(latent_dim, num_classes, hidden_dim)
        self.mi_unique_audio = InfoNCEEstimator(latent_dim, num_classes, hidden_dim)
        self.mi_unique_vision = InfoNCEEstimator(latent_dim, num_classes, hidden_dim)
        self.mi_redundancy = InfoNCEEstimator(latent_dim, num_classes, hidden_dim)
        self.mi_synergy = InfoNCEEstimator(latent_dim, num_classes, hidden_dim)

    def forward(
        self,
        z_text: torch.Tensor,
        z_audio: torch.Tensor,
        z_vision: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Decompose modality representations into PID components using KAN heads."""
        # Pool sequences
        t_pool = masked_mean(z_text, mask)
        a_pool = masked_mean(z_audio, mask)
        v_pool = masked_mean(z_vision, mask)

        # KAN-based unique extraction
        u_text = self.unique_text(t_pool)
        u_audio = self.unique_audio(a_pool)
        u_vision = self.unique_vision(v_pool)

        # KAN-based redundancy
        concat_all = torch.cat([t_pool, a_pool, v_pool], dim=-1)
        redundant = self.redundancy(concat_all)

        # Synergy: gated triple interaction → KAN transform
        interaction = t_pool * a_pool * v_pool
        gate = self.synergy_gate(concat_all)
        synergistic = self.synergy_transform(gate * interaction)

        return {
            "unique_text": u_text,
            "unique_audio": u_audio,
            "unique_vision": u_vision,
            "redundant": redundant,
            "synergistic": synergistic,
        }

    def pid_consistency_loss(
        self,
        components: dict[str, torch.Tensor],
        labels: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        """Orthogonality constraint on PID components (same as MLP version)."""
        u_t = components["unique_text"]
        u_a = components["unique_audio"]
        u_v = components["unique_vision"]
        red = components["redundant"]

        def cosine_penalty(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return F.cosine_similarity(a, b, dim=-1).abs().mean()

        ortho_loss = (
            cosine_penalty(u_t, u_a) + cosine_penalty(u_t, u_v)
            + cosine_penalty(u_a, u_v)
            + cosine_penalty(u_t, red) + cosine_penalty(u_a, red)
            + cosine_penalty(u_v, red)
        ) / 6.0

        return ortho_loss

    def kan_regularization_loss(self) -> torch.Tensor:
        """Aggregate KAN regularization across all heads."""
        total = torch.tensor(0.0)
        for module in [self.unique_text, self.unique_audio, self.unique_vision,
                       self.redundancy, self.synergy_transform]:
            if hasattr(module, "regularization_loss"):
                loss = module.regularization_loss()
                total = total.to(loss.device) + loss
        return total
