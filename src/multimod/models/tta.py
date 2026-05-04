"""Test-Time Adaptation via entropy minimization for EIDMSA.

At test time, given an unlabeled batch, we minimize the entropy of the
Dirichlet predictive distribution by adapting only the IB compression
parameters (mu_head and logvar_head). This allows the model to adapt
to new speakers, recording conditions, or cultural contexts without
retraining.

Reference: Wang et al., "Tent: Fully Test-Time Adaptation by Entropy
Minimization", ICLR 2021 — extended here for evidential multimodal models.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn


def dirichlet_entropy(alpha: torch.Tensor) -> torch.Tensor:
    """Compute the entropy of the expected categorical distribution
    under a Dirichlet(alpha).

    H[E[p]] where E[p_k] = α_k / S, S = sum(α).

    Args:
        alpha: [batch, K] Dirichlet concentration parameters

    Returns:
        entropy: [batch] — per-sample entropy
    """
    S = alpha.sum(dim=-1, keepdim=True)
    probs = alpha / S
    # Avoid log(0)
    log_probs = torch.log(probs + 1e-10)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def collect_ib_params(model: nn.Module) -> list[nn.Parameter]:
    """Collect only the IB encoder parameters (mu_head, logvar_head)
    that should be adapted at test time.

    These control the compression rate and are the minimal set of
    parameters needed to shift the information bottleneck.
    """
    params = []
    for name, param in model.named_parameters():
        if "mu_head" in name or "logvar_head" in name:
            params.append(param)
    return params


class TestTimeAdapter:
    """Entropy-minimization test-time adaptation for EIDMSA.

    Adapts only IB compression parameters on each test batch to minimize
    the entropy of the Dirichlet predictive distribution.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        num_steps: int = 3,
        min_entropy_threshold: float = 0.1,
    ) -> None:
        """
        Args:
            model: The EIDMSA model to adapt.
            lr: Learning rate for adaptation steps.
            num_steps: Number of gradient steps per test batch.
            min_entropy_threshold: Stop adapting if entropy drops below this.
        """
        self.num_steps = num_steps
        self.lr = lr
        self.min_entropy_threshold = min_entropy_threshold

        # Save original state to reset after each batch
        self._original_state = copy.deepcopy(model.state_dict())
        self.model = model

    def reset(self) -> None:
        """Reset model to original state (before any TTA)."""
        self.model.load_state_dict(self._original_state)

    def adapt_and_predict(
        self,
        batch: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Adapt the model on this batch and return predictions.

        The model is temporarily put into training mode for the IB
        parameters only, adapted, then predictions are collected.
        After prediction, the model is reset to its original state.

        Args:
            batch: dict with 'text', 'audio', 'vision', 'mask' tensors

        Returns:
            output: The model's forward output dict after adaptation
        """
        # Reset to original trained weights
        self.model.load_state_dict(copy.deepcopy(self._original_state))

        # Freeze everything except IB params
        for param in self.model.parameters():
            param.requires_grad_(False)

        ib_params = collect_ib_params(self.model)
        for param in ib_params:
            param.requires_grad_(True)

        if not ib_params:
            # No IB params found — just do regular forward
            self.model.eval()
            with torch.no_grad():
                return self.model(
                    batch["text"], batch["audio"],
                    batch["vision"], batch["mask"],
                )

        optimizer = torch.optim.Adam(ib_params, lr=self.lr)

        # Adaptation loop
        with torch.enable_grad():
            self.model.train()
            for step in range(self.num_steps):
                optimizer.zero_grad(set_to_none=True)

                output = self.model(
                    batch["text"], batch["audio"],
                    batch["vision"], batch["mask"],
                )

                alpha = output.get("alpha")
                if alpha is None:
                    break

                entropy = dirichlet_entropy(alpha).mean()

                if entropy.item() < self.min_entropy_threshold:
                    break

                entropy.backward()
                optimizer.step()

        # Final prediction after adaptation
        self.model.eval()
        with torch.no_grad():
            output = self.model(
                batch["text"], batch["audio"],
                batch["vision"], batch["mask"],
            )

        # Reset requires_grad state
        for param in self.model.parameters():
            param.requires_grad_(True)

        return output
