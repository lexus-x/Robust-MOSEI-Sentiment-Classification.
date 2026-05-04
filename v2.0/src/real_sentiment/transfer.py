"""Temporal-split transfer evaluation for realistic benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from .benchmark import (
    ProtocolSpec,
    build_protocol_specs,
    evaluate_model_on_protocol,
    expected_calibration_error,
    selective_risk_at_coverage,
)


def temporal_split_indices(
    num_samples: int,
    train_fraction: float = 0.70,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Split sample indices temporally (first N% train, last (1-N)% test).

    For MOSEI, samples are ordered by video ID which correlates with upload time,
    giving a rough temporal split without needing explicit timestamps.
    """
    indices = np.arange(num_samples)
    cutoff = int(num_samples * train_fraction)
    return {
        "train": indices[:cutoff],
        "test": indices[cutoff:],
    }


def evaluate_temporal_transfer(
    model: torch.nn.Module,
    full_dataloader: torch.utils.data.DataLoader,
    specs: list[ProtocolSpec],
    device: torch.device,
    train_fraction: float = 0.70,
) -> dict[str, Any]:
    """Evaluate model on temporal-split test portion only.

    This verifies the benchmark is not overfit to one random split by using
    a temporal ordering instead of the standard random split.
    """
    # Collect all predictions on the full dataset under clean condition first
    condition_rows, prediction_rows, summary = evaluate_model_on_protocol(
        model=model,
        dataloader=full_dataloader,
        specs=specs,
        device=device,
    )

    # Separate temporal-split test portion from prediction rows
    total_samples = len(set(row["sample_id"] for row in prediction_rows if row["condition_label"] == "clean"))
    cutoff = int(total_samples * train_fraction)

    # Get unique sample IDs in order and determine which are in the test split
    clean_rows = [r for r in prediction_rows if r["condition_label"] == "clean"]
    unique_ids = list(dict.fromkeys(r["sample_id"] for r in clean_rows))
    test_ids = set(unique_ids[cutoff:])

    # Filter condition_rows based on temporal test split by recomputing
    # from prediction rows that fall in the test split
    temporal_condition_rows = _recompute_condition_metrics(
        prediction_rows=[r for r in prediction_rows if r["sample_id"] in test_ids],
        specs=specs,
    )

    return {
        "transfer_type": "temporal_split",
        "train_fraction": train_fraction,
        "total_samples": total_samples,
        "test_samples": total_samples - cutoff,
        "full_evaluation_summary": summary,
        "temporal_split_conditions": temporal_condition_rows,
        "temporal_split_clean_f1": _get_clean_f1(temporal_condition_rows),
        "temporal_split_avg_perturbed_f1": _get_avg_perturbed_f1(temporal_condition_rows),
    }


def _recompute_condition_metrics(
    prediction_rows: list[dict[str, Any]],
    specs: list[ProtocolSpec],
) -> list[dict[str, Any]]:
    """Recompute condition-level metrics from filtered prediction rows."""
    condition_rows: list[dict[str, Any]] = []
    for spec in specs:
        rows = [r for r in prediction_rows if r["condition_label"] == spec.label]
        if not rows:
            continue
        labels = np.array([r["label"] for r in rows])
        preds = np.array([r["prediction"] for r in rows])
        confidences = np.array([r["confidence"] for r in rows])
        trust_scores = np.array([r["trust_score"] for r in rows])

        metrics: dict[str, Any] = {
            "condition_label": spec.label,
            "condition_name": spec.name,
            "family": spec.family,
            "severity": spec.severity,
            "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
            "accuracy": float(accuracy_score(labels, preds)),
            "ece": expected_calibration_error(labels, preds, confidences, n_bins=10),
            "selective_risk_80": selective_risk_at_coverage(labels, preds, trust_scores, coverage=0.80),
            "num_samples": int(labels.size),
        }
        if "uncertainty" in rows[0]:
            uncertainties = np.array([r["uncertainty"] for r in rows])
            metrics["mean_uncertainty"] = float(uncertainties.mean())
            conflicts = np.array([r["conflict"] for r in rows])
            metrics["mean_conflict"] = float(conflicts.mean())
        condition_rows.append(metrics)
    return condition_rows


def _get_clean_f1(condition_rows: list[dict[str, Any]]) -> float:
    for row in condition_rows:
        if row["condition_label"] == "clean":
            return row["weighted_f1"]
    return float("nan")


def _get_avg_perturbed_f1(condition_rows: list[dict[str, Any]]) -> float:
    perturbed = [r["weighted_f1"] for r in condition_rows if r["condition_label"] != "clean"]
    return float(np.mean(perturbed)) if perturbed else float("nan")
