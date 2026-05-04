"""Evidential Deep Learning fusion with Dempster-Shafer theory.

Each PID component produces evidence for a Dirichlet distribution over
class probabilities. These are combined via Discounted Belief Fusion,
which automatically detects and handles inter-modal conflict.

Key concepts:
  - Evidence e_k → concentration α_k = e_k + 1
  - Dirichlet Dir(p | α) naturally represents both prediction and uncertainty
  - Total evidence S = sum(α) measures confidence; low S = high uncertainty
  - Conflict between components is detected and used for discounting
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class EvidentialHead(nn.Module):
    """Maps a representation to Dirichlet evidence (non-negative)."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
            nn.Softplus(),  # Ensure non-negative evidence
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Produce evidence vector e >= 0 for each sample.

        Args:
            features: [batch, input_dim]

        Returns:
            evidence: [batch, num_classes] — non-negative evidence
        """
        return self.net(features)


class EvidentialFusion(nn.Module):
    """Fuses PID components via Discounted Belief Fusion.

    Each of the 5 PID components (unique_t, unique_a, unique_v, redundant,
    synergistic) produces independent evidence. These are combined using
    a conflict-aware fusion rule inspired by Dempster-Shafer theory.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # One evidential head per PID component
        self.evidence_heads = nn.ModuleDict({
            "unique_text": EvidentialHead(latent_dim, hidden_dim, num_classes, dropout),
            "unique_audio": EvidentialHead(latent_dim, hidden_dim, num_classes, dropout),
            "unique_vision": EvidentialHead(latent_dim, hidden_dim, num_classes, dropout),
            "redundant": EvidentialHead(latent_dim, hidden_dim, num_classes, dropout),
            "synergistic": EvidentialHead(latent_dim, hidden_dim, num_classes, dropout),
        })

    def _belief_from_evidence(self, evidence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert evidence to belief mass and uncertainty.

        Args:
            evidence: [batch, K] non-negative evidence

        Returns:
            belief: [batch, K] — belief mass per class
            uncertainty: [batch, 1] — uncertainty mass
        """
        alpha = evidence + 1.0
        S = alpha.sum(dim=-1, keepdim=True)
        belief = evidence / S
        uncertainty = self.num_classes / S
        return belief, uncertainty

    def _pairwise_conflict(self, b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
        """Compute conflict between two belief functions.

        Conflict = sum_{i≠j} b1_i * b2_j (mass assigned to contradictory classes).

        Args:
            b1, b2: [batch, K] belief masses

        Returns:
            conflict: [batch, 1]
        """
        # Total product of masses minus agreement
        total = (b1.sum(dim=-1, keepdim=True) * b2.sum(dim=-1, keepdim=True))
        agreement = (b1 * b2).sum(dim=-1, keepdim=True)
        conflict = (total - agreement).clamp_min(0.0)
        return conflict

    def forward(
        self,
        components: dict[str, torch.Tensor],
        reliability: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Fuse PID components via discounted belief fusion.

        Args:
            components: dict with keys matching evidence_heads
            reliability: optional per-component trust weights in [0, 1] with shape [batch, 1]

        Returns:
            Dictionary with:
                'alpha': [batch, K] — fused Dirichlet concentration
                'evidence': [batch, K] — fused evidence
                'uncertainty': [batch, 1] — fused uncertainty mass
                'conflict': [batch, 1] — total inter-component conflict
                'per_component_evidence': dict of per-component evidence
                'per_component_uncertainty': dict of per-component uncertainty
        """
        all_evidence = {}
        all_beliefs = {}
        all_uncertainties = {}

        for name, head in self.evidence_heads.items():
            if name not in components:
                continue
            ev = head(components[name])
            if reliability is not None and name in reliability:
                ev = ev * reliability[name].clamp(0.0, 1.0)
            b, u = self._belief_from_evidence(ev)
            all_evidence[name] = ev
            all_beliefs[name] = b
            all_uncertainties[name] = u

        # Discounted Belief Fusion: weight each source by (1 - conflict_with_others)
        component_names = list(all_beliefs.keys())
        n_components = len(component_names)

        # Compute pairwise conflicts
        total_conflict = torch.zeros(
            all_beliefs[component_names[0]].shape[0], 1,
            device=all_beliefs[component_names[0]].device,
            dtype=all_beliefs[component_names[0]].dtype,
        )
        discount_factors = {}

        for i, name_i in enumerate(component_names):
            conflict_i = torch.zeros_like(total_conflict)
            for j, name_j in enumerate(component_names):
                if i == j:
                    continue
                c = self._pairwise_conflict(all_beliefs[name_i], all_beliefs[name_j])
                conflict_i = conflict_i + c
            # Average conflict for this component
            if n_components > 1:
                conflict_i = conflict_i / (n_components - 1)
            # Discount factor: less trust when high conflict
            discount_factors[name_i] = 1.0 - conflict_i.clamp(0.0, 0.999)
            total_conflict = total_conflict + conflict_i

        if n_components > 0:
            total_conflict = total_conflict / n_components

        # Fuse: weighted sum of evidence, discounted by conflict
        fused_evidence = torch.zeros_like(all_evidence[component_names[0]])
        for name in component_names:
            fused_evidence = fused_evidence + discount_factors[name] * all_evidence[name]

        fused_alpha = fused_evidence + 1.0
        fused_S = fused_alpha.sum(dim=-1, keepdim=True)
        fused_uncertainty = self.num_classes / fused_S

        return {
            "alpha": fused_alpha,
            "evidence": fused_evidence,
            "uncertainty": fused_uncertainty,
            "conflict": total_conflict,
            "per_component_evidence": all_evidence,
            "per_component_uncertainty": all_uncertainties,
        }


def evidential_loss(
    alpha: torch.Tensor,
    labels: torch.Tensor,
    epoch: int = 0,
    total_epochs: int = 50,
    annealing_start: int = 5,
) -> torch.Tensor:
    """Evidential Deep Learning loss (Type-II Maximum Likelihood).

    Combines:
      1. Negative log-likelihood of the expected Dirichlet distribution
      2. KL divergence regularizer (annealed) to prevent evidence inflation

    Args:
        alpha: [batch, K] — Dirichlet concentration parameters
        labels: [batch] — ground truth class indices
        epoch: current training epoch
        total_epochs: total number of training epochs
        annealing_start: epoch to start annealing the KL regularizer

    Returns:
        loss: scalar
    """
    num_classes = alpha.shape[-1]
    batch_size = alpha.shape[0]

    # One-hot encode labels
    one_hot = F.one_hot(labels, num_classes=num_classes).float()

    S = alpha.sum(dim=-1, keepdim=True)

    # Type-II MLE: E[log p(y|θ)] where θ ~ Dir(α)
    # = ψ(α_y) - ψ(S) ≈ log(α_y / S) for large α
    nll = (one_hot * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=-1).mean()

    # KL regularizer: KL(Dir(α_tilde) || Dir(1, ..., 1))
    # where α_tilde removes evidence for the correct class
    alpha_tilde = one_hot + (1.0 - one_hot) * alpha

    S_tilde = alpha_tilde.sum(dim=-1, keepdim=True)
    kl = (
        torch.lgamma(S_tilde)
        - torch.lgamma(torch.tensor(num_classes, dtype=alpha.dtype, device=alpha.device))
        - torch.lgamma(alpha_tilde).sum(dim=-1, keepdim=True)
        + ((alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))).sum(
            dim=-1, keepdim=True
        )
    ).mean()

    # Annealing coefficient: gradually increase KL weight
    if epoch < annealing_start:
        annealing_coeff = 0.0
    else:
        annealing_coeff = min(1.0, (epoch - annealing_start) / max(total_epochs - annealing_start, 1))

    return nll + annealing_coeff * kl
