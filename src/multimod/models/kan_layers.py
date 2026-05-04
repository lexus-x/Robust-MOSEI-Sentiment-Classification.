"""KAN (Kolmogorov-Arnold Network) layers for interpretable fusion.

Integrates the efficient-kan implementation (Blealtan, MIT License) as
drop-in replacements for MLP layers in the PID decomposition and
evidential heads.

Reference:
  - Liu et al., "KAN: Kolmogorov-Arnold Networks", ICLR 2025
  - github.com/Blealtan/efficient-kan (MIT License)

Key advantage over MLP: learnable activation functions on edges
(B-splines) provide inherent interpretability — you can inspect
exactly what nonlinear transform each feature undergoes.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class KANLinear(nn.Module):
    """Efficient KAN linear layer with B-spline activations on edges.

    Adapted from github.com/Blealtan/efficient-kan (MIT License).
    Reformulated as matrix multiplications for GPU efficiency.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation: type = nn.SiLU,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                self._curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def _b_splines(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        bases = ((x >= self.grid[:, :-1]) & (x < self.grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - self.grid[:, : -(k + 1)])
                / (self.grid[:, k:-1] - self.grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (self.grid[:, k + 1 :] - x)
                / (self.grid[:, k + 1 :] - self.grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
        return bases.contiguous()

    def _curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        A = self._b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1).contiguous()

    @property
    def _scaled_spline_weight(self) -> torch.Tensor:
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self._b_splines(x).view(x.size(0), -1),
            self._scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        return output.reshape(*original_shape[:-1], self.out_features)

    def regularization_loss(self) -> torch.Tensor:
        """L1 + entropy regularization on spline weights."""
        l1 = self.spline_weight.abs().mean(-1)
        reg_l1 = l1.sum()
        p = l1 / reg_l1.clamp_min(1e-8)
        reg_entropy = -torch.sum(p * (p + 1e-8).log())
        return reg_l1 + reg_entropy


class KANLayer(nn.Module):
    """Drop-in KAN replacement for nn.Linear + activation.

    Use this anywhere you'd use nn.Sequential(nn.Linear, nn.GELU).
    Provides learnable nonlinear activation via B-splines instead of
    fixed GELU/ReLU.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.kan = KANLinear(in_features, out_features, grid_size=grid_size)
        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.norm(self.kan(x)))

    def regularization_loss(self) -> torch.Tensor:
        return self.kan.regularization_loss()


class KANProjection(nn.Module):
    """KAN-based projection head — replaces MLPProjection.

    Two-layer KAN: input_dim → hidden_dim → output_dim.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        grid_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        output_dim = output_dim or hidden_dim
        self.layer1 = KANLayer(input_dim, hidden_dim, grid_size=grid_size, dropout=dropout)
        self.layer2 = KANLayer(hidden_dim, output_dim, grid_size=grid_size, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.layer1(x))

    def regularization_loss(self) -> torch.Tensor:
        return self.layer1.regularization_loss() + self.layer2.regularization_loss()
