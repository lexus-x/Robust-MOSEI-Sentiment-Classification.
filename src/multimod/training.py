"""Training loop and experiment orchestration."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from .config import ExperimentConfig, save_config
from .data import apply_training_robustness, build_dataloaders, describe_dataset
from .evaluation import evaluate, evaluate_eidmsa
from .models import InputDims, build_model
from .models.eidmsa import EIDMSA
from .utils import count_parameters, ensure_dir, move_batch_to_device, resolve_device, save_json, set_seed


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    modality_dropout_p: float,
    jitter_prob: float,
    gradient_clip: float,
    rng: random.Random,
    realistic_corruption_p: float = 0.0,
    audio_dropout_p: float | None = None,
    vision_dropout_p: float | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        batch = apply_training_robustness(
            batch=batch,
            modality_dropout_p=modality_dropout_p,
            jitter_prob=jitter_prob,
            rng=rng,
            realistic_corruption_p=realistic_corruption_p,
            audio_dropout_p=audio_dropout_p,
            vision_dropout_p=vision_dropout_p,
        )
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(batch["text"], batch["audio"], batch["vision"], batch["mask"])
        loss = criterion(logits, batch["label"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        batch_size = batch["label"].shape[0]
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
    return total_loss / max(total_items, 1)


def train_one_epoch_eidmsa(
    model: EIDMSA,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    modality_dropout_p: float,
    jitter_prob: float,
    gradient_clip: float,
    rng: random.Random,
    epoch: int = 0,
    total_epochs: int = 30,
    pid_weight: float = 0.1,
    use_evidential_loss: bool = True,
    realistic_corruption_p: float = 0.0,
    audio_dropout_p: float | None = None,
    vision_dropout_p: float | None = None,
    alignment_weight: float = 0.0,
) -> dict[str, float]:
    """Training loop for EIDMSA with composite loss."""
    model.train()
    total_losses = {"total": 0.0, "evidential": 0.0, "ib": 0.0, "pid": 0.0, "alignment": 0.0}
    total_items = 0

    for batch in dataloader:
        clean_batch = move_batch_to_device(batch, device)
        batch = apply_training_robustness(
            batch=clean_batch,
            modality_dropout_p=modality_dropout_p,
            jitter_prob=jitter_prob,
            rng=rng,
            realistic_corruption_p=realistic_corruption_p,
            audio_dropout_p=audio_dropout_p,
            vision_dropout_p=vision_dropout_p,
        )
        optimizer.zero_grad(set_to_none=True)

        output = model(batch["text"], batch["audio"], batch["vision"], batch["mask"])
        losses = model.compute_loss(
            output=output,
            labels=batch["label"],
            epoch=epoch,
            total_epochs=total_epochs,
            pid_weight=pid_weight,
            use_evidential_loss=use_evidential_loss,
        )

        alignment_loss = torch.zeros((), device=device, dtype=output["logits"].dtype)
        if alignment_weight > 0.0:
            model.eval()
            with torch.no_grad():
                teacher_output = model(
                    clean_batch["text"],
                    clean_batch["audio"],
                    clean_batch["vision"],
                    clean_batch["mask"],
                )
            model.train()
            alignment_loss = F.kl_div(
                output["logits"].clamp_min(1e-8).log(),
                teacher_output["logits"].clamp_min(1e-8),
                reduction="batchmean",
            )
            losses["total"] = losses["total"] + alignment_weight * alignment_loss

        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        batch_size = batch["label"].shape[0]
        for key in total_losses:
            if key == "alignment":
                total_losses[key] += float(alignment_loss.item()) * batch_size
            else:
                total_losses[key] += float(losses[key].item()) * batch_size
        total_items += batch_size

    return {key: val / max(total_items, 1) for key, val in total_losses.items()}


def _validation_score(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    metrics = evaluate(
        model=model,
        dataloader=dataloader,
        conditions=("clean",),
        device=device,
        output_dir=None,
        diagnostics_examples=0,
    )
    return float(metrics["conditions"][0]["weighted_f1"])


def _validation_score_eidmsa(
    model: EIDMSA,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Validation score for EIDMSA using the clean condition."""
    metrics = evaluate_eidmsa(
        model=model,
        dataloader=dataloader,
        conditions=("clean",),
        device=device,
        output_dir=None,
    )
    return float(metrics["conditions"][0]["weighted_f1"])


def train_model(
    model: torch.nn.Module,
    dataloaders: dict[str, torch.utils.data.DataLoader],
    config: ExperimentConfig,
    device: torch.device,
    seed: int,
) -> tuple[torch.nn.Module, list[dict[str, Any]]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    history: list[dict[str, Any]] = []
    best_metric = float("-inf")
    best_state = copy.deepcopy(model.state_dict())
    bad_epochs = 0
    rng = random.Random(seed)

    for epoch in range(1, config.training.max_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            modality_dropout_p=config.model.modality_dropout_p,
            jitter_prob=config.model.jitter_prob,
            gradient_clip=config.training.gradient_clip,
            rng=rng,
            realistic_corruption_p=config.model.realistic_corruption_p,
            audio_dropout_p=config.model.audio_dropout_p,
            vision_dropout_p=config.model.vision_dropout_p,
        )
        valid_f1 = _validation_score(model, dataloaders["valid"], device=device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_weighted_f1": valid_f1,
        }
        history.append(row)
        if valid_f1 > best_metric:
            best_metric = valid_f1
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= config.training.patience:
                break

    model.load_state_dict(best_state)
    return model, history


def train_model_eidmsa(
    model: EIDMSA,
    dataloaders: dict[str, torch.utils.data.DataLoader],
    config: ExperimentConfig,
    device: torch.device,
    seed: int,
) -> tuple[EIDMSA, list[dict[str, Any]]]:
    """Training loop specifically for EIDMSA models with composite loss."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    history: list[dict[str, Any]] = []
    best_metric = float("-inf")
    best_state = copy.deepcopy(model.state_dict())
    bad_epochs = 0
    rng = random.Random(seed)

    for epoch in range(1, config.training.max_epochs + 1):
        losses = train_one_epoch_eidmsa(
            model=model,
            dataloader=dataloaders["train"],
            optimizer=optimizer,
            device=device,
            modality_dropout_p=config.model.modality_dropout_p,
            jitter_prob=config.model.jitter_prob,
            gradient_clip=config.training.gradient_clip,
            rng=rng,
            epoch=epoch,
            total_epochs=config.training.max_epochs,
            pid_weight=config.model.pid_weight,
            use_evidential_loss=(
                config.model.use_evidential_loss
                and epoch > config.model.evidential_warmup_epochs
            ),
            realistic_corruption_p=config.model.realistic_corruption_p,
            audio_dropout_p=config.model.audio_dropout_p,
            vision_dropout_p=config.model.vision_dropout_p,
            alignment_weight=config.model.alignment_weight,
        )
        valid_f1 = _validation_score_eidmsa(model, dataloaders["valid"], device=device)
        row = {
            "epoch": epoch,
            "train_loss_total": losses["total"],
            "train_loss_evidential": losses["evidential"],
            "train_loss_ib": losses["ib"],
            "train_loss_pid": losses["pid"],
            "train_loss_alignment": losses["alignment"],
            "valid_weighted_f1": valid_f1,
        }
        history.append(row)
        print(
            f"  Epoch {epoch:3d} | "
            f"loss={losses['total']:.4f} (ev={losses['evidential']:.4f} "
            f"ib={losses['ib']:.4f} pid={losses['pid']:.4f} align={losses['alignment']:.4f}) | "
            f"val_f1={valid_f1:.4f}"
        )
        if valid_f1 > best_metric:
            best_metric = valid_f1
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= config.training.patience:
                print(f"  Early stopping at epoch {epoch} (patience={config.training.patience})")
                break

    model.load_state_dict(best_state)
    return model, history


def run_experiment(
    config: ExperimentConfig,
    seed: int,
    device_name: str = "auto",
) -> dict[str, Any]:
    set_seed(seed)
    device = resolve_device(device_name)
    run_dir = ensure_dir(Path(config.output_dir) / config.experiment_name / f"seed_{seed}")
    save_config(config, run_dir / "config.yaml")

    dataloaders = build_dataloaders(
        data_path=config.data.data_path,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        max_seq_len=config.data.max_seq_len,
        lower_threshold=config.data.lower_threshold,
        upper_threshold=config.data.upper_threshold,
        label_mode=config.data.label_mode,
    )
    stats = describe_dataset(
        data_path=config.data.data_path,
        split="train",
        max_seq_len=config.data.max_seq_len,
        label_mode=config.data.label_mode,
    )
    input_dims = InputDims(text=stats.text_dim, audio=stats.audio_dim, vision=stats.vision_dim)
    model = build_model(config.model, input_dims=input_dims).to(device)

    # Route to the correct training/evaluation pipeline
    is_eidmsa = isinstance(model, EIDMSA)

    if is_eidmsa:
        trained_model, history = train_model_eidmsa(
            model=model,
            dataloaders=dataloaders,
            config=config,
            device=device,
            seed=seed,
        )
    else:
        trained_model, history = train_model(
            model=model,
            dataloaders=dataloaders,
            config=config,
            device=device,
            seed=seed,
        )

    checkpoint_path = run_dir / "best_model.pt"
    torch.save(trained_model.state_dict(), checkpoint_path)
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    if is_eidmsa:
        results = evaluate_eidmsa(
            model=trained_model,
            dataloader=dataloaders["test"],
            conditions=config.data.conditions,
            device=device,
            output_dir=run_dir,
            use_tta=config.model.use_tta,
            tta_lr=config.model.tta_lr,
            tta_steps=config.model.tta_steps,
        )
    else:
        results = evaluate(
            model=trained_model,
            dataloader=dataloaders["test"],
            conditions=config.data.conditions,
            device=device,
            output_dir=run_dir,
            diagnostics_examples=config.diagnostics_examples,
        )

    results["run"] = {
        "experiment": config.experiment_name,
        "seed": seed,
        "device": str(device),
        "num_parameters": count_parameters(trained_model),
        "checkpoint": str(checkpoint_path),
        "notes": config.notes,
    }
    save_json(results, run_dir / "metrics.json")
    return results
