"""Realistic robustness benchmark wired to the existing multimod checkpoints."""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from .claim import build_benchmark_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MULTIMOD_SRC = PROJECT_ROOT / "src"
if str(MULTIMOD_SRC) not in sys.path:
    sys.path.insert(0, str(MULTIMOD_SRC))

from multimod.config import load_experiment_config
from multimod.data import build_dataloaders, describe_dataset
from multimod.evaluation import compute_standard_mosei_metrics
from multimod.models import InputDims, build_model
from multimod.models.eidmsa import EIDMSA
from multimod.utils import count_parameters, move_batch_to_device, resolve_device


@dataclass(frozen=True)
class ProtocolSpec:
    name: str
    family: str
    severity: str
    target_modalities: tuple[str, ...]
    description: str

    @property
    def label(self) -> str:
        if self.severity == "none":
            return self.name
        return f"{self.name}::{self.severity}"


def build_protocol_specs() -> list[ProtocolSpec]:
    manifest = build_benchmark_manifest()
    specs: list[ProtocolSpec] = []
    for condition in manifest["conditions"]:
        for severity in condition["severity_levels"]:
            specs.append(
                ProtocolSpec(
                    name=condition["name"],
                    family=condition["family"],
                    severity=severity,
                    target_modalities=tuple(condition["target_modalities"]),
                    description=condition["description"],
                )
            )
    specs.sort(key=lambda spec: (spec.name != "clean", spec.name, spec.severity))
    return specs


def collect_run_dirs(path: str | Path) -> list[Path]:
    path = Path(path)
    if (path / "config.yaml").exists() and (path / "best_model.pt").exists():
        return [path]
    return sorted(
        run_dir.parent
        for run_dir in path.rglob("config.yaml")
        if (run_dir.parent / "best_model.pt").exists()
    )


def _seed_from_run_dir(run_dir: Path) -> int | None:
    if run_dir.name.startswith("seed_"):
        try:
            return int(run_dir.name.split("_", maxsplit=1)[1])
        except ValueError:
            return None
    return None


def _clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.clone()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def _condition_seed(label: str) -> int:
    return sum(ord(char) for char in label) + 17


def _severity_value(spec: ProtocolSpec) -> float | int:
    if spec.name in {"block_missing_audio", "block_missing_vision"}:
        return {"mild": 0.20, "moderate": 0.40, "severe": 0.60}[spec.severity]
    if spec.name in {"lead_lag_audio", "lead_lag_vision"}:
        return {"2_frames": 2, "4_frames": 4, "8_frames": 8}[spec.severity]
    if spec.name == "drift_audio":
        return {"mild": 2, "moderate": 4}[spec.severity]
    if spec.name == "burst_noise_vision":
        return {"mild": 0.25, "moderate": 0.50}[spec.severity]
    if spec.name == "compound_audio_vision_failure":
        return {"moderate": 0.30, "severe": 0.50}[spec.severity]
    return 0.0


def _valid_indices(mask_row: torch.Tensor) -> torch.Tensor:
    return torch.nonzero(mask_row, as_tuple=False).flatten()


def _apply_contiguous_drop(
    features: torch.Tensor,
    mask: torch.Tensor,
    fraction: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    output = features.clone()
    for batch_index in range(output.shape[0]):
        valid = _valid_indices(mask[batch_index])
        if valid.numel() == 0 or fraction <= 0.0:
            continue
        span = max(1, int(round(valid.numel() * fraction)))
        span = min(span, valid.numel())
        start_offset = int(rng.integers(0, valid.numel() - span + 1))
        chosen = valid[start_offset : start_offset + span]
        output[batch_index, chosen, :] = 0.0
    return output


def _apply_temporal_shift(
    features: torch.Tensor,
    mask: torch.Tensor,
    shift: int,
) -> torch.Tensor:
    output = features.clone()
    for batch_index in range(output.shape[0]):
        valid = _valid_indices(mask[batch_index])
        if valid.numel() == 0 or shift == 0:
            continue
        values = output[batch_index, valid, :].clone()
        shifted = torch.zeros_like(values)
        if abs(shift) >= valid.numel():
            output[batch_index, valid, :] = 0.0
            continue
        if shift > 0:
            shifted[shift:] = values[:-shift]
        else:
            shifted[:shift] = values[-shift:]
        output[batch_index, valid, :] = shifted
    return output


def _apply_progressive_drift(
    features: torch.Tensor,
    mask: torch.Tensor,
    max_shift: int,
) -> torch.Tensor:
    output = features.clone()
    for batch_index in range(output.shape[0]):
        valid = _valid_indices(mask[batch_index])
        if valid.numel() == 0 or max_shift <= 0:
            continue
        values = output[batch_index, valid, :].clone()
        drifted = torch.zeros_like(values)
        for value_index in range(valid.numel()):
            shift = int(round(max_shift * value_index / max(valid.numel() - 1, 1)))
            src_index = max(0, value_index - shift)
            drifted[value_index] = values[src_index]
        output[batch_index, valid, :] = drifted
    return output


def _apply_burst_noise(
    features: torch.Tensor,
    mask: torch.Tensor,
    span_fraction: float,
    scale: float,
    rng: np.random.Generator,
) -> torch.Tensor:
    output = features.clone()
    for batch_index in range(output.shape[0]):
        valid = _valid_indices(mask[batch_index])
        if valid.numel() == 0 or span_fraction <= 0.0 or scale <= 0.0:
            continue
        span = max(1, int(round(valid.numel() * span_fraction)))
        span = min(span, valid.numel())
        start_offset = int(rng.integers(0, valid.numel() - span + 1))
        chosen = valid[start_offset : start_offset + span]
        noise = torch.from_numpy(
            rng.normal(loc=0.0, scale=scale, size=tuple(output[batch_index, chosen, :].shape))
        ).to(device=output.device, dtype=output.dtype)
        output[batch_index, chosen, :] = output[batch_index, chosen, :] + noise
    return output


def apply_protocol_condition(
    batch: dict[str, Any],
    spec: ProtocolSpec,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    conditioned = _clone_batch(batch)
    if spec.name == "clean":
        return conditioned

    rng = rng or np.random.default_rng(0)
    mask = conditioned["mask"]

    if spec.name == "block_missing_audio":
        conditioned["audio"] = _apply_contiguous_drop(
            conditioned["audio"],
            mask,
            fraction=float(_severity_value(spec)),
            rng=rng,
        )
    elif spec.name == "block_missing_vision":
        conditioned["vision"] = _apply_contiguous_drop(
            conditioned["vision"],
            mask,
            fraction=float(_severity_value(spec)),
            rng=rng,
        )
    elif spec.name == "lead_lag_audio":
        conditioned["audio"] = _apply_temporal_shift(
            conditioned["audio"],
            mask,
            shift=int(_severity_value(spec)),
        )
    elif spec.name == "lead_lag_vision":
        conditioned["vision"] = _apply_temporal_shift(
            conditioned["vision"],
            mask,
            shift=int(_severity_value(spec)),
        )
    elif spec.name == "drift_audio":
        conditioned["audio"] = _apply_progressive_drift(
            conditioned["audio"],
            mask,
            max_shift=int(_severity_value(spec)),
        )
    elif spec.name == "burst_noise_vision":
        conditioned["vision"] = _apply_burst_noise(
            conditioned["vision"],
            mask,
            span_fraction=0.25,
            scale=float(_severity_value(spec)),
            rng=rng,
        )
    elif spec.name == "compound_audio_vision_failure":
        if spec.severity == "moderate":
            audio_fraction = 0.30
            vision_shift = 2
        else:
            audio_fraction = 0.50
            vision_shift = 4
        conditioned["audio"] = _apply_contiguous_drop(
            conditioned["audio"],
            mask,
            fraction=audio_fraction,
            rng=rng,
        )
        conditioned["vision"] = _apply_temporal_shift(
            conditioned["vision"],
            mask,
            shift=vision_shift,
        )
    else:
        raise ValueError(f"Unsupported protocol condition: {spec.label}")

    return conditioned


def _class_support(num_classes: int) -> np.ndarray | None:
    if num_classes == 3:
        return np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    if num_classes == 7:
        return np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    return None


def _probabilities_to_scores(probabilities: torch.Tensor) -> tuple[torch.Tensor | None, str | None]:
    support = _class_support(probabilities.shape[-1])
    if support is None:
        return None, None
    support_tensor = torch.tensor(support, device=probabilities.device, dtype=probabilities.dtype)
    score_mode = (
        "class_expectation_3class"
        if probabilities.shape[-1] == 3
        else "class_expectation_7class"
    )
    return probabilities @ support_tensor, score_mode


def expected_calibration_error(
    labels: np.ndarray,
    predictions: np.ndarray,
    confidences: np.ndarray,
    n_bins: int = 10,
) -> float:
    total = len(labels)
    if total == 0:
        return 0.0
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for index in range(n_bins):
        in_bin = (confidences > bin_boundaries[index]) & (confidences <= bin_boundaries[index + 1])
        n_in_bin = int(in_bin.sum())
        if n_in_bin == 0:
            continue
        avg_confidence = float(confidences[in_bin].mean())
        avg_accuracy = float((predictions[in_bin] == labels[in_bin]).mean())
        ece += (n_in_bin / total) * abs(avg_accuracy - avg_confidence)
    return float(ece)


def selective_risk_at_coverage(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    coverage: float = 0.80,
) -> float:
    if labels.size == 0:
        return float("nan")
    keep = max(1, int(np.ceil(labels.size * coverage)))
    order = np.argsort(scores)[::-1]
    chosen = order[:keep]
    return float((predictions[chosen] != labels[chosen]).mean())


def coverage_at_risk(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    risk_threshold: float = 0.20,
) -> float:
    if labels.size == 0:
        return 0.0
    order = np.argsort(scores)[::-1]
    correctness = (predictions[order] == labels[order]).astype(np.float32)
    cumulative_accuracy = np.cumsum(correctness) / np.arange(1, labels.size + 1)
    cumulative_risk = 1.0 - cumulative_accuracy
    valid = np.flatnonzero(cumulative_risk <= risk_threshold)
    if valid.size == 0:
        return 0.0
    return float((valid[-1] + 1) / labels.size)


def summarize_run_metrics(condition_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_label = {row["condition_label"]: row for row in condition_rows}
    clean = by_label["clean"]
    perturbed = [row for row in condition_rows if row["condition_label"] != "clean"]
    summary: dict[str, Any] = {
        "clean_weighted_f1": clean["weighted_f1"],
        "avg_perturbed_weighted_f1": float(np.mean([row["weighted_f1"] for row in perturbed])),
        "clean_accuracy": clean["accuracy"],
        "avg_perturbed_accuracy": float(np.mean([row["accuracy"] for row in perturbed])),
        "clean_ece": clean["ece"],
        "avg_perturbed_ece": float(np.mean([row["ece"] for row in perturbed])),
        "clean_selective_risk_80": clean["selective_risk_80"],
        "avg_perturbed_selective_risk_80": float(np.mean([row["selective_risk_80"] for row in perturbed])),
        "clean_coverage_at_risk_20": clean["coverage_at_risk_20"],
        "avg_perturbed_coverage_at_risk_20": float(np.mean([row["coverage_at_risk_20"] for row in perturbed])),
    }
    worst = min(perturbed, key=lambda row: row["weighted_f1"])
    summary["worst_condition_label"] = worst["condition_label"]
    summary["worst_condition_weighted_f1"] = worst["weighted_f1"]

    if "mosei_mae" in clean:
        keys = (
            "mosei_mae",
            "mosei_corr",
            "mosei_acc_7",
            "mosei_acc_2_nonneg",
            "mosei_f1_nonneg",
            "mosei_acc_2_negpos",
            "mosei_f1_negpos",
        )
        for key in keys:
            summary[f"clean_{key}"] = clean[key]
            summary[f"avg_perturbed_{key}"] = float(np.mean([row[key] for row in perturbed]))

    if "mean_uncertainty" in clean:
        summary["clean_uncertainty"] = clean["mean_uncertainty"]
        summary["avg_perturbed_uncertainty"] = float(np.mean([row["mean_uncertainty"] for row in perturbed]))
        summary["clean_conflict"] = clean["mean_conflict"]
        summary["avg_perturbed_conflict"] = float(np.mean([row["mean_conflict"] for row in perturbed]))

    return summary


def evaluate_model_on_protocol(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    specs: list[ProtocolSpec],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    condition_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    model.eval()
    for spec in specs:
        rng = np.random.default_rng(_condition_seed(spec.label))
        all_labels: list[int] = []
        all_predictions: list[int] = []
        all_confidences: list[float] = []
        all_trust_scores: list[float] = []
        all_raw_sentiments: list[float] = []
        all_scores: list[float] = []
        all_uncertainties: list[float] = []
        all_conflicts: list[float] = []

        with torch.no_grad():
            for batch in dataloader:
                batch = move_batch_to_device(batch, device)
                conditioned = apply_protocol_condition(batch, spec, rng=rng)

                if isinstance(model, EIDMSA):
                    output = model(
                        conditioned["text"],
                        conditioned["audio"],
                        conditioned["vision"],
                        conditioned["mask"],
                    )
                    probabilities = output["logits"]
                    preds = probabilities.argmax(dim=-1)
                    confidences = probabilities.max(dim=-1).values
                    uncertainties = output["uncertainty"].squeeze(-1)
                    conflicts = output["conflict"].squeeze(-1)
                    trust_scores = (confidences * (1.0 - uncertainties) * (1.0 - conflicts)).clamp(0.0, 1.0)
                else:
                    logits, _ = model(
                        conditioned["text"],
                        conditioned["audio"],
                        conditioned["vision"],
                        conditioned["mask"],
                    )
                    probabilities = torch.softmax(logits, dim=-1)
                    preds = logits.argmax(dim=-1)
                    confidences = probabilities.max(dim=-1).values
                    uncertainties = None
                    conflicts = None
                    trust_scores = confidences

                sentiment_scores, score_mode = _probabilities_to_scores(probabilities)
                labels = conditioned["label"]
                raw_sentiment = conditioned.get("raw_sentiment")

                all_labels.extend(labels.cpu().tolist())
                all_predictions.extend(preds.cpu().tolist())
                all_confidences.extend(confidences.cpu().tolist())
                all_trust_scores.extend(trust_scores.cpu().tolist())
                if raw_sentiment is not None and sentiment_scores is not None:
                    all_raw_sentiments.extend(raw_sentiment.cpu().tolist())
                    all_scores.extend(sentiment_scores.cpu().tolist())
                if uncertainties is not None:
                    all_uncertainties.extend(uncertainties.cpu().tolist())
                if conflicts is not None:
                    all_conflicts.extend(conflicts.cpu().tolist())

                for batch_index, sample_id in enumerate(conditioned["sample_id"]):
                    row = {
                        "sample_id": sample_id,
                        "condition_label": spec.label,
                        "condition_name": spec.name,
                        "family": spec.family,
                        "severity": spec.severity,
                        "label": int(labels[batch_index].item()),
                        "prediction": int(preds[batch_index].item()),
                        "confidence": float(confidences[batch_index].item()),
                        "trust_score": float(trust_scores[batch_index].item()),
                    }
                    if raw_sentiment is not None:
                        row["raw_sentiment"] = float(raw_sentiment[batch_index].item())
                    if sentiment_scores is not None:
                        row["sentiment_score"] = float(sentiment_scores[batch_index].item())
                        row["mosei_score_mode"] = score_mode
                    if uncertainties is not None:
                        row["uncertainty"] = float(uncertainties[batch_index].item())
                    if conflicts is not None:
                        row["conflict"] = float(conflicts[batch_index].item())
                    prediction_rows.append(row)

        labels_np = np.asarray(all_labels)
        predictions_np = np.asarray(all_predictions)
        confidences_np = np.asarray(all_confidences)
        trust_scores_np = np.asarray(all_trust_scores)
        metrics: dict[str, Any] = {
            "condition_label": spec.label,
            "condition_name": spec.name,
            "family": spec.family,
            "severity": spec.severity,
            "weighted_f1": float(f1_score(labels_np, predictions_np, average="weighted", zero_division=0)),
            "accuracy": float(accuracy_score(labels_np, predictions_np)),
            "ece": expected_calibration_error(labels_np, predictions_np, confidences_np, n_bins=10),
            "selective_risk_80": selective_risk_at_coverage(
                labels_np,
                predictions_np,
                trust_scores_np,
                coverage=0.80,
            ),
            "coverage_at_risk_20": coverage_at_risk(
                labels_np,
                predictions_np,
                trust_scores_np,
                risk_threshold=0.20,
            ),
            "num_samples": int(labels_np.size),
        }
        if all_raw_sentiments and all_scores:
            metrics.update(compute_standard_mosei_metrics(np.asarray(all_raw_sentiments), np.asarray(all_scores)))
            metrics["mosei_score_mode"] = score_mode
        if all_uncertainties:
            metrics["mean_uncertainty"] = float(np.mean(all_uncertainties))
            metrics["mean_conflict"] = float(np.mean(all_conflicts))
            # Abstention metrics: abstain when uncertainty > threshold
            uncertainties_np = np.asarray(all_uncertainties)
            abstention_threshold = float(np.percentile(uncertainties_np, 80))
            retained_mask = uncertainties_np <= abstention_threshold
            metrics["abstention_rate"] = float(1.0 - retained_mask.mean())
            if retained_mask.sum() > 0:
                metrics["abstention_accuracy"] = float(
                    accuracy_score(labels_np[retained_mask], predictions_np[retained_mask])
                )
                metrics["abstention_weighted_f1"] = float(
                    f1_score(labels_np[retained_mask], predictions_np[retained_mask], average="weighted", zero_division=0)
                )
            else:
                metrics["abstention_accuracy"] = float("nan")
                metrics["abstention_weighted_f1"] = float("nan")
        condition_rows.append(metrics)

    return condition_rows, prediction_rows, summarize_run_metrics(condition_rows)


def _loader_cache_key(config: Any) -> tuple[Any, ...]:
    return (
        config.data.data_path,
        config.data.batch_size,
        config.data.num_workers,
        config.data.max_seq_len,
        config.data.lower_threshold,
        config.data.upper_threshold,
        getattr(config.data, "label_mode", "3class"),
    )


def _load_model_and_test_loader(
    run_dir: Path,
    device: torch.device,
    loader_cache: dict[tuple[Any, ...], dict[str, torch.utils.data.DataLoader]],
    stats_cache: dict[tuple[Any, ...], Any],
) -> tuple[Any, torch.nn.Module, torch.utils.data.DataLoader]:
    config = load_experiment_config(run_dir / "config.yaml")
    cache_key = _loader_cache_key(config)
    if cache_key not in loader_cache:
        loader_cache[cache_key] = build_dataloaders(
            data_path=config.data.data_path,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            max_seq_len=config.data.max_seq_len,
            lower_threshold=config.data.lower_threshold,
            upper_threshold=config.data.upper_threshold,
            label_mode=getattr(config.data, "label_mode", "3class"),
        )
    if cache_key not in stats_cache:
        stats_cache[cache_key] = describe_dataset(
            data_path=config.data.data_path,
            split="train",
            max_seq_len=config.data.max_seq_len,
            label_mode=getattr(config.data, "label_mode", "3class"),
        )
    stats = stats_cache[cache_key]
    model = build_model(
        config.model,
        input_dims=InputDims(text=stats.text_dim, audio=stats.audio_dim, vision=stats.vision_dim),
    ).to(device)
    state_dict = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state_dict)
    return config, model, loader_cache[cache_key]["test"]


def run_protocol_for_root(
    run_root: str | Path,
    role: str,
    specs: list[ProtocolSpec],
    device_name: str = "auto",
    max_seeds: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    device = resolve_device(device_name)
    run_dirs = collect_run_dirs(run_root)
    if max_seeds is not None:
        run_dirs = run_dirs[:max_seeds]
    loader_cache: dict[tuple[Any, ...], dict[str, torch.utils.data.DataLoader]] = {}
    stats_cache: dict[tuple[Any, ...], Any] = {}

    condition_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        config, model, test_loader = _load_model_and_test_loader(
            run_dir=run_dir,
            device=device,
            loader_cache=loader_cache,
            stats_cache=stats_cache,
        )
        condition_metrics, predictions, summary = evaluate_model_on_protocol(
            model=model,
            dataloader=test_loader,
            specs=specs,
            device=device,
        )
        run_meta = {
            "role": role,
            "run_dir": str(run_dir),
            "experiment": config.experiment_name,
            "seed": _seed_from_run_dir(run_dir),
            "device": str(device),
            "num_parameters": count_parameters(model),
            "checkpoint_mb": (run_dir / "best_model.pt").stat().st_size / (1024 * 1024),
        }
        run_rows.append({**run_meta, **summary})
        for row in condition_metrics:
            condition_rows.append({**run_meta, **row})
        for row in predictions:
            prediction_rows.append({**run_meta, **row})

    return (
        pd.DataFrame(condition_rows),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(run_rows),
    )


def compare_roles(
    condition_df: pd.DataFrame,
    run_df: pd.DataFrame,
    compact_role: str = "compact",
    baseline_role: str = "baseline",
    clean_gap_tolerance: float = 0.02,
    mean_gap_tolerance: float = 0.02,
    worst_gap_tolerance: float = 0.03,
) -> dict[str, Any]:
    compact_runs = run_df[run_df["role"] == compact_role]
    baseline_runs = run_df[run_df["role"] == baseline_role]
    matched_seeds = sorted(set(compact_runs["seed"]).intersection(set(baseline_runs["seed"])))
    if not matched_seeds:
        raise ValueError("No matched seeds between compact and baseline runs.")

    compact_conditions = condition_df[
        (condition_df["role"] == compact_role) & (condition_df["seed"].isin(matched_seeds))
    ].copy()
    baseline_conditions = condition_df[
        (condition_df["role"] == baseline_role) & (condition_df["seed"].isin(matched_seeds))
    ].copy()

    merged = compact_conditions.merge(
        baseline_conditions,
        on=["seed", "condition_label", "condition_name", "family", "severity"],
        suffixes=("_compact", "_baseline"),
    )
    merged["weighted_f1_gap_vs_baseline"] = (
        merged["weighted_f1_baseline"] - merged["weighted_f1_compact"]
    )
    merged["accuracy_gap_vs_baseline"] = merged["accuracy_baseline"] - merged["accuracy_compact"]
    merged["ece_delta_vs_baseline"] = merged["ece_compact"] - merged["ece_baseline"]
    merged["selective_risk_delta_vs_baseline"] = (
        merged["selective_risk_80_compact"] - merged["selective_risk_80_baseline"]
    )

    family_summary = (
        merged.groupby("family", as_index=False)
        .agg(
            compact_weighted_f1=("weighted_f1_compact", "mean"),
            baseline_weighted_f1=("weighted_f1_baseline", "mean"),
            weighted_f1_gap_vs_baseline=("weighted_f1_gap_vs_baseline", "mean"),
            compact_ece=("ece_compact", "mean"),
            baseline_ece=("ece_baseline", "mean"),
            ece_delta_vs_baseline=("ece_delta_vs_baseline", "mean"),
            compact_selective_risk_80=("selective_risk_80_compact", "mean"),
            baseline_selective_risk_80=("selective_risk_80_baseline", "mean"),
            selective_risk_delta_vs_baseline=("selective_risk_delta_vs_baseline", "mean"),
        )
        .sort_values("weighted_f1_gap_vs_baseline")
    )

    clean_rows = merged[merged["condition_label"] == "clean"]
    perturbed_rows = merged[merged["condition_label"] != "clean"]
    worst_row = perturbed_rows.sort_values("weighted_f1_gap_vs_baseline", ascending=False).iloc[0]
    best_row = perturbed_rows.sort_values("weighted_f1_gap_vs_baseline", ascending=True).iloc[0]

    relevant_partial_rows = perturbed_rows[
        perturbed_rows["family"].isin(["missingness", "local_corruption", "compound"])
    ]

    # Gate verdicts
    clean_gate_pass = bool(clean_rows["weighted_f1_gap_vs_baseline"].max() <= clean_gap_tolerance)
    mean_robustness_pass = bool(
        perturbed_rows["weighted_f1_gap_vs_baseline"].mean() <= mean_gap_tolerance
    )
    worst_case_robustness_pass = bool(
        perturbed_rows["weighted_f1_gap_vs_baseline"].max() <= worst_gap_tolerance
    )

    # Calibration gate: compact ECE < baseline ECE on average
    calibration_pass = bool(merged["ece_delta_vs_baseline"].mean() < 0.0)

    # Abstention gate: check if abstention_accuracy columns exist
    has_abstention = (
        "abstention_accuracy_compact" in merged.columns
        and "abstention_accuracy_baseline" in merged.columns
    )
    if has_abstention:
        compact_abstention_acc = merged["abstention_accuracy_compact"].dropna().mean()
        baseline_acc = merged["accuracy_compact"].mean()
        abstention_improves = bool(compact_abstention_acc > baseline_acc)
    else:
        abstention_improves = False

    full_claim_supported = bool(
        clean_gate_pass and mean_robustness_pass and worst_case_robustness_pass and calibration_pass
    )
    partial_missingness_claim_supported = bool(
        clean_gate_pass
        and not relevant_partial_rows.empty
        and relevant_partial_rows["weighted_f1_gap_vs_baseline"].max() <= worst_gap_tolerance
    )

    return {
        "matched_seeds": matched_seeds,
        "compact_role": compact_role,
        "baseline_role": baseline_role,
        "clean_gap_tolerance": clean_gap_tolerance,
        "mean_gap_tolerance": mean_gap_tolerance,
        "worst_gap_tolerance": worst_gap_tolerance,
        "full_realistic_claim_supported": full_claim_supported,
        "partial_missingness_claim_supported": partial_missingness_claim_supported,
        # Individual gate verdicts
        "gate_clean": clean_gate_pass,
        "gate_mean_robustness": mean_robustness_pass,
        "gate_worst_case_robustness": worst_case_robustness_pass,
        "gate_calibration": calibration_pass,
        "gate_abstention": abstention_improves,
        # Numeric summaries
        "clean_gap_mean": float(clean_rows["weighted_f1_gap_vs_baseline"].mean()),
        "clean_gap_worst": float(clean_rows["weighted_f1_gap_vs_baseline"].max()),
        "avg_perturbed_gap_mean": float(perturbed_rows["weighted_f1_gap_vs_baseline"].mean()),
        "avg_perturbed_gap_worst": float(perturbed_rows["weighted_f1_gap_vs_baseline"].max()),
        "partial_claim_gap_worst": float(relevant_partial_rows["weighted_f1_gap_vs_baseline"].max()),
        "mean_ece_delta": float(merged["ece_delta_vs_baseline"].mean()),
        "best_condition_for_compact": {
            "condition_label": str(best_row["condition_label"]),
            "gap_vs_baseline": float(best_row["weighted_f1_gap_vs_baseline"]),
        },
        "worst_condition_for_compact": {
            "condition_label": str(worst_row["condition_label"]),
            "gap_vs_baseline": float(worst_row["weighted_f1_gap_vs_baseline"]),
        },
        "family_summary": family_summary.to_dict(orient="records"),
        "condition_comparison": merged.sort_values(
            ["seed", "condition_name", "severity"]
        ).to_dict(orient="records"),
    }
