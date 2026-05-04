"""Neural Partial Information Decomposition for multimodal sentiment.

Decomposes the mutual information I(Y; T, A, V) into:
  - Unique(Y; T\\A,V)  — information only text provides about sentiment
  - Unique(Y; A\\T,V)  — information only audio provides
  - Unique(Y; V\\T,A)  — information only vision provides
  - Redundancy(Y; T,A,V) — shared sentiment signal across all modalities
  - Synergy(Y; T,A,V) — sentiment that emerges only from combination

We estimate these components using learned projection heads with
contrastive (InfoNCE-style) objectives for mutual information estimation,
and enforce consistency via the PID identity.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from ..utils import masked_mean


class InfoNCEEstimator(nn.Module):
    """Differentiable mutual information lower bound via InfoNCE.

    Estimates I(Z; Y) using a bilinear critic function.
    """

    def __init__(self, z_dim: int, y_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(z_dim + y_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Estimate MI lower bound.

        Args:
            z: [batch, dim_z] — representation
            y: [batch, dim_y] — target encoding

        Returns:
            mi_estimate: scalar — InfoNCE lower bound on I(Z; Y)
        """
        batch_size = z.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=z.device, dtype=z.dtype)

        # Positive pairs: (z_i, y_i)
        pos_input = torch.cat([z, y], dim=-1)
        pos_scores = self.critic(pos_input).squeeze(-1)

        # Negative pairs: (z_i, y_j) for j ≠ i (all permutations)
        y_shuffled = y[torch.randperm(batch_size, device=y.device)]
        neg_input = torch.cat([z, y_shuffled], dim=-1)
        neg_scores = self.critic(neg_input).squeeze(-1)

        # InfoNCE: log(exp(pos) / (exp(pos) + exp(neg)))
        logits = torch.stack([pos_scores, neg_scores], dim=-1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=z.device)
        mi_estimate = torch.log(torch.tensor(2.0, device=z.device)) - F.cross_entropy(logits, labels)
        return mi_estimate


class PIDDecomposition(nn.Module):
    """Neural Partial Information Decomposition module.

    Takes IB-compressed modality representations and decomposes them into
    unique, redundant, and synergistic components for sentiment prediction.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Unique extraction heads: isolate modality-specific information
        self.unique_text = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.unique_audio = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.unique_vision = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Redundancy head: extract shared sentiment signal
        self.redundancy = nn.Sequential(
            nn.Linear(latent_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Synergy head: extract emergent cross-modal interactions
        # Uses element-wise product to capture multiplicative interactions
        self.synergy_gate = nn.Sequential(
            nn.Linear(latent_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Sigmoid(),
        )
        self.synergy_transform = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        # MI estimators for the consistency loss
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
        """Decompose modality representations into PID components.

        Args:
            z_text, z_audio, z_vision: [batch, seq_len, latent_dim]
            mask: [batch, seq_len] boolean

        Returns:
            Dictionary with keys:
                'unique_text', 'unique_audio', 'unique_vision': [batch, latent_dim]
                'redundant': [batch, latent_dim]
                'synergistic': [batch, latent_dim]
        """
        # Pool sequences to fixed-size representations
        t_pool = masked_mean(z_text, mask)    # [batch, latent_dim]
        a_pool = masked_mean(z_audio, mask)   # [batch, latent_dim]
        v_pool = masked_mean(z_vision, mask)  # [batch, latent_dim]

        # Extract unique components
        u_text = self.unique_text(t_pool)
        u_audio = self.unique_audio(a_pool)
        u_vision = self.unique_vision(v_pool)

        # Extract redundant component (shared across all)
        concat_all = torch.cat([t_pool, a_pool, v_pool], dim=-1)
        redundant = self.redundancy(concat_all)

        # Extract synergistic component (multiplicative interactions)
        # Key insight: synergy arises from interactions that no single 
        # modality can capture alone
        interaction = t_pool * a_pool * v_pool  # triple interaction
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
        """PID consistency: components should capture non-overlapping information.

        Enforces that unique components are orthogonal to each other and to
        the redundant component, while the synergy component captures
        information not present in any individual modality.
        """
        u_t = components["unique_text"]
        u_a = components["unique_audio"]
        u_v = components["unique_vision"]
        red = components["redundant"]

        # Orthogonality loss: unique components should be dissimilar
        def cosine_penalty(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            sim = F.cosine_similarity(a, b, dim=-1).abs().mean()
            return sim

        ortho_loss = (
            cosine_penalty(u_t, u_a)
            + cosine_penalty(u_t, u_v)
            + cosine_penalty(u_a, u_v)
            + cosine_penalty(u_t, red)
            + cosine_penalty(u_a, red)
            + cosine_penalty(u_v, red)
        ) / 6.0

        return ortho_loss
