"""Model factory for MOSEI robustness experiments."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from ..config import ModelConfig
from .early_fusion import EarlyFusionClassifier
from .eidmsa import EIDMSA
from .text_only import TextOnlyClassifier
from .transformer import CrossModalTransformer


@dataclass
class InputDims:
    text: int
    audio: int
    vision: int


def build_model(config: ModelConfig, input_dims: InputDims) -> nn.Module:
    if config.name == "text_only":
        return TextOnlyClassifier(
            text_dim=input_dims.text,
            hidden_dim=config.hidden_dim,
            num_classes=config.num_classes,
            dropout=config.dropout,
        )
    if config.name == "early_fusion":
        return EarlyFusionClassifier(
            text_dim=input_dims.text,
            audio_dim=input_dims.audio,
            vision_dim=input_dims.vision,
            hidden_dim=config.hidden_dim,
            num_classes=config.num_classes,
            dropout=config.dropout,
        )
    if config.name == "xmodal_transformer":
        return CrossModalTransformer(
            text_dim=input_dims.text,
            audio_dim=input_dims.audio,
            vision_dim=input_dims.vision,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            num_classes=config.num_classes,
            dropout=config.dropout,
            use_gating=config.use_gating,
        )
    if config.name == "eidmsa":
        return EIDMSA(
            text_dim=input_dims.text,
            audio_dim=input_dims.audio,
            vision_dim=input_dims.vision,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            num_classes=config.num_classes,
            dropout=config.dropout,
            ib_beta=config.ib_beta,
            use_kan=config.use_kan,
            use_mamba=config.use_mamba,
            kan_grid_size=config.kan_grid_size,
            kan_reg_weight=config.kan_reg_weight,
        )
    raise ValueError(f"Unsupported model '{config.name}'.")
