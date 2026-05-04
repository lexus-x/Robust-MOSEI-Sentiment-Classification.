"""Configuration helpers for MOSEI robustness experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    data_path: str
    batch_size: int = 32
    num_workers: int = 0
    max_seq_len: int | None = None
    lower_threshold: float = -0.5
    upper_threshold: float = 0.5
    label_mode: str = "3class"
    conditions: tuple[str, ...] = (
        "clean",
        "missing_audio",
        "missing_vision",
        "missing_audio_vision",
        "mild_jitter",
    )


@dataclass
class ModelConfig:
    name: str
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    num_classes: int = 3
    use_gating: bool = False
    modality_dropout_p: float = 0.0
    audio_dropout_p: float | None = None
    vision_dropout_p: float | None = None
    jitter_prob: float = 0.0
    realistic_corruption_p: float = 0.0
    # EIDMSA-specific parameters
    latent_dim: int = 64
    ib_beta: float = 1e-3
    pid_weight: float = 0.1
    use_evidential_loss: bool = True
    evidential_warmup_epochs: int = 0
    alignment_weight: float = 0.0
    use_tta: bool = False
    tta_lr: float = 1e-4
    tta_steps: int = 3
    # Novel paper integrations
    use_kan: bool = False   # KAN projection heads (efficient-kan, ICLR 2025)
    use_mamba: bool = False  # Mamba SSM encoder (MSAmba, AAAI 2025)
    kan_grid_size: int = 5
    kan_reg_weight: float = 1e-4


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    max_epochs: int = 20
    patience: int = 3
    gradient_clip: float = 1.0
    seeds: tuple[int, ...] = (13, 17, 23)


@dataclass
class ExperimentConfig:
    experiment_name: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: str = "outputs"
    save_predictions: bool = True
    diagnostics_examples: int = 2
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _base_model(name: str) -> ModelConfig:
    return ModelConfig(name=name)


def available_experiments() -> tuple[str, ...]:
    return (
        "text_only",
        "early_fusion",
        "xmodal_transformer",
        "xmodal_transformer_robust",
        "minus_gating",
        "minus_modality_dropout",
        "minus_jitter_augmentation",
        # EIDMSA experiments
        "eidmsa",
        "eidmsa_7class",
        "eidmsa_no_ib",
        "eidmsa_no_pid",
        "eidmsa_no_evidential",
        "eidmsa_tta",
        "eidmsa_realistic_retry",
        "eidmsa_robust_v2",
        # Novel paper integrations
        "eidmsa_kan",
        "eidmsa_mamba",
        "eidmsa_kan_mamba",
    )


def make_experiment_config(
    experiment_name: str,
    data_path: str,
    output_dir: str = "outputs",
) -> ExperimentConfig:
    if experiment_name not in available_experiments():
        raise ValueError(f"Unknown experiment '{experiment_name}'.")

    data = DataConfig(data_path=data_path)
    model = _base_model(experiment_name)
    training = TrainingConfig()
    notes = ""

    if experiment_name == "text_only":
        training.seeds = (13,)
    elif experiment_name == "early_fusion":
        training.seeds = (13,)
    elif experiment_name == "xmodal_transformer":
        model.name = "xmodal_transformer"
    elif experiment_name == "xmodal_transformer_robust":
        model.name = "xmodal_transformer"
        model.use_gating = True
        model.modality_dropout_p = 0.2
        model.jitter_prob = 0.3
        notes = "Full robust model with gating, modality dropout, and jitter augmentation."
    elif experiment_name == "minus_gating":
        model.name = "xmodal_transformer"
        model.use_gating = False
        model.modality_dropout_p = 0.2
        model.jitter_prob = 0.3
        training.seeds = (13,)
        notes = "Robust training without gating."
    elif experiment_name == "minus_modality_dropout":
        model.name = "xmodal_transformer"
        model.use_gating = True
        model.modality_dropout_p = 0.0
        model.jitter_prob = 0.3
        training.seeds = (13,)
        notes = "Gating plus jitter augmentation only."
    elif experiment_name == "minus_jitter_augmentation":
        model.name = "xmodal_transformer"
        model.use_gating = True
        model.modality_dropout_p = 0.2
        model.jitter_prob = 0.0
        training.seeds = (13,)
        notes = "Gating plus modality dropout only."
    # ---- EIDMSA experiments ----
    elif experiment_name == "eidmsa":
        model.name = "eidmsa"
        model.hidden_dim = 128
        model.latent_dim = 64
        model.ib_beta = 1e-3
        model.pid_weight = 0.1
        model.modality_dropout_p = 0.2
        model.jitter_prob = 0.3
        training.max_epochs = 30
        training.patience = 5
        notes = "Full EIDMSA: IB + PID + Evidential Fusion."
    elif experiment_name == "eidmsa_7class":
        model.name = "eidmsa"
        model.hidden_dim = 128
        model.latent_dim = 64
        model.num_classes = 7
        model.ib_beta = 1e-3
        model.pid_weight = 0.1
        model.modality_dropout_p = 0.2
        model.jitter_prob = 0.3
        data.label_mode = "7class"
        training.max_epochs = 30
        training.patience = 5
        notes = "EIDMSA on 7-class ordinal sentiment."
    elif experiment_name == "eidmsa_no_ib":
        model.name = "eidmsa"
        model.hidden_dim = 128
        model.latent_dim = 64
        model.ib_beta = 0.0  # Disable IB regularization
        model.pid_weight = 0.1
        model.modality_dropout_p = 0.2
        model.jitter_prob = 0.3
        training.max_epochs = 30
        training.patience = 5
        training.seeds = (13,)
        notes = "EIDMSA ablation: no Information Bottleneck."
    elif experiment_name == "eidmsa_no_pid":
        model.name = "eidmsa"
        model.hidden_dim = 128
        model.latent_dim = 64
        model.ib_beta = 1e-3
        model.pid_weight = 0.0  # Disable PID consistency
        model.modality_dropout_p = 0.2
        model.jitter_prob = 0.3
        training.max_epochs = 30
        training.patience = 5
        training.seeds = (13,)
        notes = "EIDMSA ablation: no PID consistency loss."
    elif experiment_name == "eidmsa_no_evidential":
        model.name = "eidmsa"
        model.hidden_dim = 128
        model.latent_dim = 64
        model.ib_beta = 1e-3
        model.pid_weight = 0.1
        model.use_evidential_loss = False
        model.modality_dropout_p = 0.2
        model.jitter_prob = 0.3
        training.max_epochs = 30
        training.patience = 5
        training.seeds = (13,)
        notes = "EIDMSA ablation: standard CE loss instead of evidential."
    elif experiment_name == "eidmsa_tta":
        model.name = "eidmsa"
        model.hidden_dim = 128
        model.latent_dim = 64
        model.ib_beta = 1e-3
        model.pid_weight = 0.1
        model.use_tta = True
        model.tta_lr = 1e-4
        model.tta_steps = 3
        model.modality_dropout_p = 0.2
        model.jitter_prob = 0.3
        training.max_epochs = 30
        training.patience = 5
        notes = "EIDMSA with Test-Time Adaptation enabled."
    elif experiment_name == "eidmsa_realistic_retry":
        model.name = "eidmsa"
        model.hidden_dim = 128
        model.latent_dim = 64
        model.ib_beta = 1e-3
        model.pid_weight = 0.1
        model.use_evidential_loss = True
        model.evidential_warmup_epochs = 8
        model.alignment_weight = 0.15
        model.audio_dropout_p = 0.15
        model.vision_dropout_p = 0.35
        model.jitter_prob = 0.15
        model.realistic_corruption_p = 0.60
        training.max_epochs = 24
        training.patience = 6
        notes = (
            "EIDMSA retry: stronger realistic corruption training, asymmetric vision dropout, "
            "clean-to-corrupted consistency, and CE-to-EDL warmup."
        )
    # ---- Novel paper integrations ----
    elif experiment_name == "eidmsa_robust_v2":
        model.name = "eidmsa"
        model.hidden_dim = 128
        model.latent_dim = 64
        model.ib_beta = 1e-3
        model.pid_weight = 0.1
        model.use_evidential_loss = True
        model.evidential_warmup_epochs = 5
        model.alignment_weight = 0.12
        model.audio_dropout_p = 0.15
        model.vision_dropout_p = 0.45
        model.jitter_prob = 0.3
        model.realistic_corruption_p = 0.55
        training.max_epochs = 40
        training.patience = 7
        notes = (
            "EIDMSA robust v2: aggressive vision dropout (0.45), high realistic corruption "
            "probability (0.55), clean-to-corrupted alignment (0.12), evidential warmup (5 epochs). "
            "Designed to close the block_missing_vision::severe gap."
        )
    # ---- Novel paper integrations (original) ----
    elif experiment_name == "eidmsa_kan":
        model.name = "eidmsa"
        model.hidden_dim = 128
        model.latent_dim = 64
        model.ib_beta = 1e-3
        model.pid_weight = 0.1
        model.use_kan = True
        model.kan_grid_size = 5
        model.kan_reg_weight = 1e-4
        model.modality_dropout_p = 0.2
        model.jitter_prob = 0.3
        training.max_epochs = 30
        training.patience = 5
        notes = "EIDMSA + KAN projection heads (efficient-kan, ICLR 2025)."
    elif experiment_name == "eidmsa_mamba":
        model.name = "eidmsa"
        model.hidden_dim = 128
        model.latent_dim = 64
        model.ib_beta = 1e-3
        model.pid_weight = 0.1
        model.use_mamba = True
        model.modality_dropout_p = 0.2
        model.jitter_prob = 0.3
        training.max_epochs = 30
        training.patience = 5
        notes = "EIDMSA + Mamba SSM encoder (MSAmba, AAAI 2025)."
    elif experiment_name == "eidmsa_kan_mamba":
        model.name = "eidmsa"
        model.hidden_dim = 128
        model.latent_dim = 64
        model.ib_beta = 1e-3
        model.pid_weight = 0.1
        model.use_kan = True
        model.use_mamba = True
        model.kan_grid_size = 5
        model.kan_reg_weight = 1e-4
        model.modality_dropout_p = 0.2
        model.jitter_prob = 0.3
        training.max_epochs = 30
        training.patience = 5
        notes = "EIDMSA + KAN + Mamba (full novel stack)."

    return ExperimentConfig(
        experiment_name=experiment_name,
        data=data,
        model=model,
        training=training,
        output_dir=output_dir,
        notes=notes,
    )


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def experiment_config_from_dict(payload: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name=payload["experiment_name"],
        data=DataConfig(**payload["data"]),
        model=ModelConfig(**payload["model"]),
        training=TrainingConfig(**payload["training"]),
        output_dir=payload.get("output_dir", "outputs"),
        save_predictions=payload.get("save_predictions", True),
        diagnostics_examples=payload.get("diagnostics_examples", 2),
        notes=payload.get("notes", ""),
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    return experiment_config_from_dict(load_config(path))
