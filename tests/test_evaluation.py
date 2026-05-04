"""Tests for evaluation module metrics."""
import math
import numpy as np

from multimod.evaluation import (
    _expected_calibration_error,
    compute_standard_mosei_metrics,
    summarize_condition_metrics,
)


def test_compute_standard_mosei_metrics_perfect_predictions():
    labels = np.array([-2.0, 0.0, 1.0, 3.0], dtype=np.float32)
    predictions = labels.copy()

    metrics = compute_standard_mosei_metrics(labels, predictions)

    assert metrics["mosei_mae"] == 0.0
    assert metrics["mosei_corr"] == 1.0
    assert metrics["mosei_acc_7"] == 1.0
    assert metrics["mosei_acc_2_nonneg"] == 1.0
    assert metrics["mosei_f1_nonneg"] == 1.0
    assert metrics["mosei_acc_2_negpos"] == 1.0
    assert metrics["mosei_f1_negpos"] == 1.0


def test_compute_standard_mosei_metrics_all_zero_labels_returns_nan_for_unstable_cases():
    labels = np.zeros(3, dtype=np.float32)
    predictions = np.zeros(3, dtype=np.float32)

    metrics = compute_standard_mosei_metrics(labels, predictions)

    assert metrics["mosei_mae"] == 0.0
    assert math.isnan(metrics["mosei_corr"])
    assert math.isnan(metrics["mosei_acc_2_negpos"])
    assert math.isnan(metrics["mosei_f1_negpos"])


def test_expected_calibration_error_uses_predicted_confidence():
    labels = np.array([0, 1, 0, 1])
    predictions = np.array([0, 1, 0, 1])
    confidences = np.array([1.0, 1.0, 1.0, 1.0])

    assert _expected_calibration_error(labels, predictions, confidences) == 0.0

def test_summarize_condition_metrics_clean_only():
    """Ensure clean-only validation does not raise NumPy empty-slice warnings."""
    condition_rows = [
        {
            "condition": "clean",
            "weighted_f1": 0.65,
            "accuracy": 0.70,
            "num_samples": 100,
            "mean_uncertainty": 0.3,
            "mean_conflict": 0.1,
            "ece": 0.05,
            "mosei_mae": 0.2,
            "mosei_corr": 0.8,
            "mosei_acc_7": 0.7,
            "mosei_acc_2_nonneg": 0.75,
            "mosei_f1_nonneg": 0.74,
            "mosei_acc_2_negpos": 0.77,
            "mosei_f1_negpos": 0.76,
            "mosei_score_mode": "class_expectation_3class",
        }
    ]
    
    summary = summarize_condition_metrics(condition_rows)
    
    # Check that summary handles missing perturbed conditions properly.
    assert summary["clean_weighted_f1"] == 0.65
    assert summary["clean_uncertainty"] == 0.3
    assert summary["clean_ece"] == 0.05
    assert math.isnan(summary["avg_perturbed_uncertainty"])
    assert math.isnan(summary["avg_perturbed_ece"])
    assert summary["clean_mosei_acc_7"] == 0.7
    assert summary["clean_mosei_mae"] == 0.2
    assert summary["mosei_score_mode"] == "class_expectation_3class"
    assert math.isnan(summary["avg_perturbed_mosei_acc_7"])
