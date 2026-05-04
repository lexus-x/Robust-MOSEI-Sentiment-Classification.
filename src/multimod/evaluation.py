"""Evaluation helpers for robustness experiments."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from .data import apply_condition
from .models.eidmsa import EIDMSA
from .models.tta import TestTimeAdapter
from .utils import move_batch_to_device, save_json


def _condition_seed(condition: str) -> int:
    return sum(ord(char) for char in condition) + 7


def _class_value_support(num_classes: int) -> np.ndarray | None:
    if num_classes == 3:
        return np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    if num_classes == 7:
        return np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    return None


def _score_mode(num_classes: int) -> str | None:
    if num_classes == 3:
        return "class_expectation_3class"
    if num_classes == 7:
        return "class_expectation_7class"
    return None


def _probabilities_to_scores(probabilities: torch.Tensor) -> tuple[torch.Tensor | None, str | None]:
    support = _class_value_support(probabilities.shape[-1])
    if support is None:
        return None, None
    support_tensor = torch.tensor(support, device=probabilities.device, dtype=probabilities.dtype)
    return probabilities @ support_tensor, _score_mode(probabilities.shape[-1])


def _safe_corrcoef(predictions: np.ndarray, labels: np.ndarray) -> float:
    if predictions.size < 2 or labels.size < 2:
        return float("nan")
    if np.isclose(np.std(predictions), 0.0) or np.isclose(np.std(labels), 0.0):
        return float("nan")
    return float(np.corrcoef(predictions, labels)[0, 1])


def compute_standard_mosei_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.float32).reshape(-1)
    keep = np.isfinite(labels) & np.isfinite(predictions)
    labels = labels[keep]
    predictions = predictions[keep]
    if labels.size == 0:
        return {
            "mosei_mae": float("nan"),
            "mosei_corr": float("nan"),
            "mosei_acc_7": float("nan"),
            "mosei_acc_2_nonneg": float("nan"),
            "mosei_f1_nonneg": float("nan"),
            "mosei_acc_2_negpos": float("nan"),
            "mosei_f1_negpos": float("nan"),
        }

    clipped_labels = np.clip(labels, -3.0, 3.0)
    clipped_predictions = np.clip(predictions, -3.0, 3.0)
    rounded_labels = np.rint(clipped_labels).astype(np.int64)
    rounded_predictions = np.rint(clipped_predictions).astype(np.int64)

    nonneg_true = (clipped_labels >= 0.0).astype(np.int64)
    nonneg_pred = (clipped_predictions >= 0.0).astype(np.int64)

    nonzero_mask = ~np.isclose(clipped_labels, 0.0)
    if nonzero_mask.any():
        negpos_true = (clipped_labels[nonzero_mask] > 0.0).astype(np.int64)
        negpos_pred = (clipped_predictions[nonzero_mask] > 0.0).astype(np.int64)
        acc_2_negpos = float(accuracy_score(negpos_true, negpos_pred))
        f1_negpos = float(f1_score(negpos_true, negpos_pred, average="weighted", zero_division=0))
    else:
        acc_2_negpos = float("nan")
        f1_negpos = float("nan")

    return {
        "mosei_mae": float(np.mean(np.abs(clipped_predictions - clipped_labels))),
        "mosei_corr": _safe_corrcoef(clipped_predictions, clipped_labels),
        "mosei_acc_7": float(accuracy_score(rounded_labels, rounded_predictions)),
        "mosei_acc_2_nonneg": float(accuracy_score(nonneg_true, nonneg_pred)),
        "mosei_f1_nonneg": float(f1_score(nonneg_true, nonneg_pred, average="weighted", zero_division=0)),
        "mosei_acc_2_negpos": acc_2_negpos,
        "mosei_f1_negpos": f1_negpos,
    }


def evaluate_condition(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    condition: str,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rng = random.Random(_condition_seed(condition))
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_confidences: list[float] = []
    all_raw_sentiments: list[float] = []
    all_sentiment_scores: list[float] = []
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            conditioned = apply_condition(batch, condition=condition, rng=rng)
            logits, aux = model(
                conditioned["text"],
                conditioned["audio"],
                conditioned["vision"],
                conditioned["mask"],
            )
            probabilities = torch.softmax(logits, dim=-1)
            sentiment_scores, score_mode = _probabilities_to_scores(probabilities)
            preds = logits.argmax(dim=-1)
            confidences = probabilities.max(dim=-1).values
            labels = conditioned["label"]
            raw_sentiment = conditioned.get("raw_sentiment")
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_confidences.extend(confidences.cpu().tolist())
            if raw_sentiment is not None and sentiment_scores is not None:
                all_raw_sentiments.extend(raw_sentiment.cpu().tolist())
                all_sentiment_scores.extend(sentiment_scores.cpu().tolist())

            gates = aux.get("gates")
            for index, sample_id in enumerate(conditioned["sample_id"]):
                row = {
                    "sample_id": sample_id,
                    "condition": condition,
                    "label": int(labels[index].item()),
                    "prediction": int(preds[index].item()),
                }
                if raw_sentiment is not None:
                    row["raw_sentiment"] = float(raw_sentiment[index].item())
                if sentiment_scores is not None:
                    row["sentiment_score"] = float(sentiment_scores[index].item())
                    row["mosei_score_mode"] = score_mode
                if gates is not None:
                    row["gate_fused"] = float(gates[index, 0].item())
                    row["gate_audio"] = float(gates[index, 1].item())
                    row["gate_vision"] = float(gates[index, 2].item())
                rows.append(row)

    metrics = {
        "condition": condition,
        "weighted_f1": float(f1_score(all_labels, all_preds, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "num_samples": len(all_labels),
        "ece": float(
            _expected_calibration_error(
                np.array(all_labels),
                np.array(all_preds),
                np.array(all_confidences),
                n_bins=10,
            )
        ),
    }
    if all_raw_sentiments and all_sentiment_scores:
        metrics.update(compute_standard_mosei_metrics(np.array(all_raw_sentiments), np.array(all_sentiment_scores)))
        metrics["mosei_score_mode"] = score_mode
    return metrics, rows


def evaluate_condition_eidmsa(
    model: EIDMSA,
    dataloader: torch.utils.data.DataLoader,
    condition: str,
    device: torch.device,
    tta_adapter: TestTimeAdapter | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Evaluate EIDMSA model on a single condition with uncertainty metrics."""
    rng = random.Random(_condition_seed(condition))
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_confidences: list[float] = []
    all_uncertainties: list[float] = []
    all_conflicts: list[float] = []
    all_raw_sentiments: list[float] = []
    all_sentiment_scores: list[float] = []
    rows: list[dict[str, Any]] = []

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        conditioned = apply_condition(batch, condition=condition, rng=rng)

        if tta_adapter is not None:
            output = tta_adapter.adapt_and_predict(conditioned)
        else:
            with torch.no_grad():
                output = model(
                    conditioned["text"],
                    conditioned["audio"],
                    conditioned["vision"],
                    conditioned["mask"],
                )

        logits = output["logits"]
        sentiment_scores, score_mode = _probabilities_to_scores(logits)
        preds = logits.argmax(dim=-1)
        confidences = logits.max(dim=-1).values
        labels = conditioned["label"]
        raw_sentiment = conditioned.get("raw_sentiment")
        uncertainty = output["uncertainty"].squeeze(-1)
        conflict = output["conflict"].squeeze(-1)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_confidences.extend(confidences.cpu().tolist())
        if raw_sentiment is not None and sentiment_scores is not None:
            all_raw_sentiments.extend(raw_sentiment.cpu().tolist())
            all_sentiment_scores.extend(sentiment_scores.cpu().tolist())
        all_uncertainties.extend(uncertainty.cpu().tolist())
        all_conflicts.extend(conflict.cpu().tolist())

        # Per-component evidence for diagnostics
        per_comp_ev = output.get("per_component_evidence", {})
        per_comp_unc = output.get("per_component_uncertainty", {})

        for index, sample_id in enumerate(conditioned["sample_id"]):
            row = {
                "sample_id": sample_id,
                "condition": condition,
                "label": int(labels[index].item()),
                "prediction": int(preds[index].item()),
                "uncertainty": float(uncertainty[index].item()),
                "conflict": float(conflict[index].item()),
            }
            if raw_sentiment is not None:
                row["raw_sentiment"] = float(raw_sentiment[index].item())
            if sentiment_scores is not None:
                row["sentiment_score"] = float(sentiment_scores[index].item())
                row["mosei_score_mode"] = score_mode
            # Add per-component evidence
            for comp_name, ev_tensor in per_comp_ev.items():
                row[f"evidence_{comp_name}_total"] = float(ev_tensor[index].sum().item())
            # Add per-component uncertainty
            for comp_name, unc_tensor in per_comp_unc.items():
                row[f"uncertainty_{comp_name}"] = float(unc_tensor[index].item())
            rows.append(row)

    # Compute standard confidence-based calibration metrics.
    ece = _expected_calibration_error(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_confidences),
        n_bins=10,
    )

    metrics = {
        "condition": condition,
        "weighted_f1": float(f1_score(all_labels, all_preds, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "num_samples": len(all_labels),
        "mean_uncertainty": float(np.mean(all_uncertainties)),
        "mean_conflict": float(np.mean(all_conflicts)),
        "ece": float(ece),
    }
    if all_raw_sentiments and all_sentiment_scores:
        metrics.update(compute_standard_mosei_metrics(np.array(all_raw_sentiments), np.array(all_sentiment_scores)))
        metrics["mosei_score_mode"] = score_mode
    return metrics, rows


def _expected_calibration_error(
    labels: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE) — measures uncertainty quality.

    Lower ECE means better-calibrated confidence scores.
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(labels)
    if total == 0:
        return 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        n_in_bin = in_bin.sum()
        if n_in_bin == 0:
            continue
        avg_confidence = confidences[in_bin].mean()
        avg_accuracy = (predictions[in_bin] == labels[in_bin]).mean()
        ece += (n_in_bin / total) * abs(avg_accuracy - avg_confidence)

    return ece


def summarize_condition_metrics(condition_rows: list[dict[str, float]]) -> dict[str, float]:
    by_condition = {row["condition"]: row for row in condition_rows}
    clean_f1 = by_condition["clean"]["weighted_f1"]
    perturbed_conditions = [name for name in by_condition if name != "clean"]
    perturbed_scores = [by_condition[name]["weighted_f1"] for name in perturbed_conditions]
    summary = {
        "clean_weighted_f1": clean_f1,
        "avg_perturbed_weighted_f1": float(sum(perturbed_scores) / max(len(perturbed_scores), 1)),
    }
    for condition in perturbed_conditions:
        summary[f"{condition}_degradation"] = float(clean_f1 - by_condition[condition]["weighted_f1"])

    if "ece" in by_condition["clean"]:
        summary["clean_ece"] = by_condition["clean"]["ece"]
        perturbed_eces = [by_condition[name]["ece"] for name in perturbed_conditions]
        summary["avg_perturbed_ece"] = float(np.mean(perturbed_eces)) if perturbed_eces else float("nan")

    # Add uncertainty metrics if available.
    if "mean_uncertainty" in by_condition["clean"]:
        summary["clean_uncertainty"] = by_condition["clean"]["mean_uncertainty"]
        summary["clean_conflict"] = by_condition["clean"]["mean_conflict"]
        perturbed_uncertainties = [by_condition[name]["mean_uncertainty"] for name in perturbed_conditions]
        summary["avg_perturbed_uncertainty"] = float(np.mean(perturbed_uncertainties)) if perturbed_uncertainties else float("nan")

    if "mosei_mae" in by_condition["clean"]:
        summary["mosei_score_mode"] = by_condition["clean"].get("mosei_score_mode")
        clean_metric_keys = (
            "mosei_mae",
            "mosei_corr",
            "mosei_acc_7",
            "mosei_acc_2_nonneg",
            "mosei_f1_nonneg",
            "mosei_acc_2_negpos",
            "mosei_f1_negpos",
        )
        for key in clean_metric_keys:
            summary[f"clean_{key}"] = by_condition["clean"][key]
            perturbed_values = [by_condition[name][key] for name in perturbed_conditions if key in by_condition[name]]
            summary[f"avg_perturbed_{key}"] = float(np.mean(perturbed_values)) if perturbed_values else float("nan")

    return summary


def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    conditions: tuple[str, ...],
    device: torch.device,
    output_dir: str | Path | None = None,
    diagnostics_examples: int = 2,
) -> dict[str, Any]:
    condition_metrics: list[dict[str, float]] = []
    prediction_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for condition in conditions:
        metrics, rows = evaluate_condition(
            model=model,
            dataloader=dataloader,
            condition=condition,
            device=device,
        )
        condition_metrics.append(metrics)
        prediction_rows.extend(rows)
        if condition != "clean":
            diagnostics.extend(rows[:diagnostics_examples])

    summary = summarize_condition_metrics(condition_metrics)
    payload: dict[str, Any] = {
        "conditions": condition_metrics,
        "summary": summary,
        "diagnostics": diagnostics[:diagnostics_examples],
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(condition_metrics).to_csv(output_dir / "condition_metrics.csv", index=False)
        pd.DataFrame(prediction_rows).to_csv(output_dir / "predictions.csv", index=False)
        save_json(payload, output_dir / "metrics.json")
    return payload


def evaluate_eidmsa(
    model: EIDMSA,
    dataloader: torch.utils.data.DataLoader,
    conditions: tuple[str, ...],
    device: torch.device,
    output_dir: str | Path | None = None,
    use_tta: bool = False,
    tta_lr: float = 1e-4,
    tta_steps: int = 3,
) -> dict[str, Any]:
    """Evaluate EIDMSA model with uncertainty and conflict metrics."""
    condition_metrics: list[dict[str, float]] = []
    prediction_rows: list[dict[str, Any]] = []

    tta_adapter = None
    if use_tta:
        tta_adapter = TestTimeAdapter(
            model=model,
            lr=tta_lr,
            num_steps=tta_steps,
        )

    for condition in conditions:
        metrics, rows = evaluate_condition_eidmsa(
            model=model,
            dataloader=dataloader,
            condition=condition,
            device=device,
            tta_adapter=tta_adapter,
        )
        condition_metrics.append(metrics)
        prediction_rows.extend(rows)

    summary = summarize_condition_metrics(condition_metrics)
    payload: dict[str, Any] = {
        "conditions": condition_metrics,
        "summary": summary,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(condition_metrics).to_csv(output_dir / "condition_metrics.csv", index=False)
        pd.DataFrame(prediction_rows).to_csv(output_dir / "predictions.csv", index=False)
        save_json(payload, output_dir / "metrics.json")

    return payload
