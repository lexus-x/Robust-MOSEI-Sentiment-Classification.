"""EIDMSA: Evidential Information-Decomposed Multimodal Sentiment Analysis.

The full unified architecture that integrates:
  1. Information Bottleneck encoders (variational compression)
  2. Neural Partial Information Decomposition (unique/redundant/synergistic)
  3. Evidential Fusion via Dempster-Shafer (uncertainty-aware prediction)
  4. Test-Time Adaptation support (entropy minimization at inference)

Optional novel paper integrations:
  5. KAN projection heads (efficient-kan, ICLR 2025) — interpretable nonlinear fusion
  6. Mamba SSM encoder (MSAmba, AAAI 2025) — O(n) sequence modeling

This model replaces the CrossModalTransformer for PhD-level research.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .common import SinusoidalPositionalEncoding
from .ib_encoder import MultiModalIBEncoder
from .pid_decomposition import PIDDecomposition
from .evidential_fusion import EvidentialFusion, evidential_loss


class EIDMSA(nn.Module):
    """Evidential Information-Decomposed Multimodal Sentiment Analysis.

    Architecture flow:
        Raw modalities → IB Encoders → Context Encoder → PID Decomposition → Evidential Fusion → Output

    The model outputs both predictions AND uncertainty estimates, and
    can decompose its predictions into interpretable PID components.
    """

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        vision_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_classes: int = 3,
        dropout: float = 0.1,
        ib_beta: float = 1e-3,
        use_kan: bool = False,
        use_mamba: bool = False,
        kan_grid_size: int = 5,
        kan_reg_weight: float = 1e-4,
    ) -> None:
        """
        Args:
            text_dim: Dimensionality of text features.
            audio_dim: Dimensionality of audio features.
            vision_dim: Dimensionality of vision features.
            hidden_dim: Hidden layer size throughout the model.
            latent_dim: Information Bottleneck latent dimension.
            num_classes: Number of output classes.
            dropout: Dropout probability.
            ib_beta: Weight for the Information Bottleneck KL term.
            use_kan: Replace MLP heads with KAN (B-spline) projections.
            use_mamba: Replace Transformer encoder with Mamba SSM.
            kan_grid_size: B-spline grid resolution for KAN layers.
            kan_reg_weight: Weight for KAN spline regularization loss.
        """
        super().__init__()
        self.num_classes = num_classes
        self.ib_beta = ib_beta
        self.latent_dim = latent_dim
        self.use_kan = use_kan
        self.use_mamba = use_mamba
        self.kan_reg_weight = kan_reg_weight

        # Stage 1: Information Bottleneck Encoders
        self.ib_encoders = MultiModalIBEncoder(
            text_dim=text_dim,
            audio_dim=audio_dim,
            vision_dim=vision_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )

        # Contextual refinement: Mamba SSM or Transformer
        if use_mamba:
            from .mamba_encoder import MambaEncoder

            self.context_encoder = MambaEncoder(
                d_model=latent_dim,
                num_layers=2,
                dropout=dropout,
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=max(1, latent_dim // 16),
                dim_feedforward=latent_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.context_encoder = nn.TransformerEncoder(
                copy.deepcopy(encoder_layer),
                num_layers=2,
                enable_nested_tensor=False,
            )

        # Cross-modal attention for synergy detection
        self.audio_cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=max(1, latent_dim // 16),
            dropout=dropout,
            batch_first=True,
        )
        self.vision_cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=max(1, latent_dim // 16),
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(latent_dim)

        # Stage 2: PID Decomposition — optionally with KAN heads
        if use_kan:
            from .pid_kan import PIDDecompositionKAN

            self.pid = PIDDecompositionKAN(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                dropout=dropout,
                grid_size=kan_grid_size,
            )
        else:
            self.pid = PIDDecomposition(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                dropout=dropout,
            )

        # Stage 3: Evidential Fusion
        self.evidential_fusion = EvidentialFusion(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        vision: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass through EIDMSA.

        Args:
            text: [batch, seq_len, text_dim]
            audio: [batch, seq_len, audio_dim]
            vision: [batch, seq_len, vision_dim]
            mask: [batch, seq_len] boolean

        Returns:
            Dictionary with:
                'logits': [batch, num_classes] — expected class probabilities
                'alpha': [batch, num_classes] — Dirichlet concentrations
                'evidence': [batch, num_classes] — fused evidence
                'uncertainty': [batch, 1] — predictive uncertainty
                'conflict': [batch, 1] — inter-component conflict
                'ib_kl': scalar — IB regularization term
                'pid_components': dict of PID component vectors
                'per_component_evidence': dict of per-component evidence
                'per_component_uncertainty': dict of per-component uncertainty
                'modality_reliability': dict of per-modality reliability weights
                'component_reliability': dict of per-component reliability weights
        """
        padding_mask = ~mask
        modality_reliability = self._estimate_modality_reliability(text, audio, vision, mask)
        component_reliability = self._component_reliability(modality_reliability)

        # Stage 1: Information Bottleneck compression
        z_text, z_audio, z_vision, ib_kl = self.ib_encoders(
            text, audio, vision, mask
        )

        # Contextual refinement
        z_text = self.context_encoder(z_text, src_key_padding_mask=padding_mask)
        z_audio_ctx, _ = self.audio_cross_attn(
            query=z_text, key=z_audio, value=z_audio,
            key_padding_mask=padding_mask,
        )
        z_vision_ctx, _ = self.vision_cross_attn(
            query=z_text, key=z_vision, value=z_vision,
            key_padding_mask=padding_mask,
        )
        # Integrate cross-modal context back into modality representations
        z_text_final = self.cross_norm(z_text + z_audio_ctx + z_vision_ctx)
        z_audio_final = z_audio  # Keep original for unique extraction
        z_vision_final = z_vision

        # Stage 2: PID Decomposition
        pid_components = self.pid(z_text_final, z_audio_final, z_vision_final, mask)

        # Stage 3: Evidential Fusion
        fusion_output = self.evidential_fusion(
            pid_components,
            reliability=component_reliability,
        )

        # Compute expected probabilities from Dirichlet
        alpha = fusion_output["alpha"]
        S = alpha.sum(dim=-1, keepdim=True)
        logits = alpha / S  # Expected probabilities E[p_k] = α_k / S

        return {
            "logits": logits,
            "alpha": alpha,
            "evidence": fusion_output["evidence"],
            "uncertainty": fusion_output["uncertainty"],
            "conflict": fusion_output["conflict"],
            "ib_kl": ib_kl,
            "pid_components": pid_components,
            "per_component_evidence": fusion_output["per_component_evidence"],
            "per_component_uncertainty": fusion_output["per_component_uncertainty"],
            "modality_reliability": modality_reliability,
            "component_reliability": component_reliability,
        }

    def _estimate_modality_reliability(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        vision: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Estimate how much valid signal survives in each modality.

        A missing or heavily zeroed modality should contribute less evidence.
        This is deliberately parameter-free so old checkpoints remain loadable.
        """

        def coverage(features: torch.Tensor) -> torch.Tensor:
            token_energy = features.abs().sum(dim=-1)
            valid = mask.float()
            observed = (token_energy > 1e-8).float() * valid
            denom = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
            return (observed.sum(dim=-1, keepdim=True) / denom).clamp(0.0, 1.0)

        return {
            "text": coverage(text),
            "audio": coverage(audio),
            "vision": coverage(vision),
        }

    def _component_reliability(
        self,
        modality_reliability: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        text_rel = modality_reliability["text"]
        audio_rel = modality_reliability["audio"]
        vision_rel = modality_reliability["vision"]
        shared_rel = (text_rel + audio_rel + vision_rel) / 3.0
        synergy_rel = torch.minimum(torch.minimum(text_rel, audio_rel), vision_rel)
        return {
            "unique_text": text_rel,
            "unique_audio": audio_rel,
            "unique_vision": vision_rel,
            "redundant": shared_rel,
            "synergistic": synergy_rel,
        }

    def compute_loss(
        self,
        output: dict[str, torch.Tensor],
        labels: torch.Tensor,
        epoch: int = 0,
        total_epochs: int = 50,
        pid_weight: float = 0.1,
        use_evidential_loss: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Compute the composite EIDMSA loss.

        Combines:
          1. Prediction loss: evidential objective or standard NLL ablation
          2. IB compression loss (β * KL divergence)
          3. PID consistency loss (orthogonality of components)
          4. KAN regularization loss (if use_kan is active)

        Args:
            output: forward() output dict
            labels: [batch] ground truth class indices
            epoch: current epoch for annealing
            total_epochs: total epochs for annealing schedule
            pid_weight: weight for PID consistency loss
            use_evidential_loss: when False, use standard NLL on expected probs

        Returns:
            Dictionary with 'total', 'evidential', 'ib', 'pid' loss terms
        """
        # 1. Prediction loss
        if use_evidential_loss:
            ev_loss = evidential_loss(
                alpha=output["alpha"],
                labels=labels,
                epoch=epoch,
                total_epochs=total_epochs,
            )
        else:
            ev_loss = F.nll_loss(output["logits"].clamp_min(1e-8).log(), labels)

        # 2. IB compression loss
        ib_loss = self.ib_beta * output["ib_kl"]

        # 3. PID consistency loss
        pid_loss = pid_weight * self.pid.pid_consistency_loss(
            components=output["pid_components"],
            labels=labels,
            num_classes=self.num_classes,
        )

        total = ev_loss + ib_loss + pid_loss

        # 4. KAN regularization (spline L1 + entropy)
        if self.use_kan and hasattr(self.pid, "kan_regularization_loss"):
            kan_loss = self.kan_reg_weight * self.pid.kan_regularization_loss()
            total = total + kan_loss

        return {
            "total": total,
            "evidential": ev_loss,
            "ib": ib_loss,
            "pid": pid_loss,
        }
