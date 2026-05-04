from __future__ import annotations

import pandas as pd

from multimod.reporting import acceptance_summary, render_final_report


def test_acceptance_summary_uses_configurable_thresholds():
    summary_df = pd.DataFrame(
        [
            {
                "experiment": "xmodal_transformer",
                "seed": 13,
                "clean_weighted_f1": 0.70,
                "avg_perturbed_weighted_f1": 0.50,
            },
            {
                "experiment": "xmodal_transformer_robust",
                "seed": 13,
                "clean_weighted_f1": 0.55,
                "avg_perturbed_weighted_f1": 0.56,
            },
            {
                "experiment": "xmodal_transformer",
                "seed": 17,
                "clean_weighted_f1": 0.68,
                "avg_perturbed_weighted_f1": 0.49,
            },
            {
                "experiment": "xmodal_transformer_robust",
                "seed": 17,
                "clean_weighted_f1": 0.63,
                "avg_perturbed_weighted_f1": 0.52,
            },
        ]
    )

    relaxed = acceptance_summary(summary_df, clean_gap_tolerance=0.2, required_positive_seeds=2)
    strict = acceptance_summary(summary_df, clean_gap_tolerance=0.05, required_positive_seeds=2)

    assert relaxed["meets_clean_criterion_for_all_seeds"] is True
    assert strict["meets_clean_criterion_for_all_seeds"] is False
    assert relaxed["meets_perturbed_direction_criterion"] is True


def test_render_final_report_calls_out_partial_support_and_ablation_limits():
    summary_df = pd.DataFrame(
        [
            {
                "experiment": "xmodal_transformer",
                "seed": 13,
                "clean_weighted_f1": 0.70,
                "avg_perturbed_weighted_f1": 0.50,
            },
            {
                "experiment": "xmodal_transformer_robust",
                "seed": 13,
                "clean_weighted_f1": 0.68,
                "avg_perturbed_weighted_f1": 0.56,
            },
            {
                "experiment": "xmodal_transformer",
                "seed": 17,
                "clean_weighted_f1": 0.68,
                "avg_perturbed_weighted_f1": 0.49,
            },
            {
                "experiment": "xmodal_transformer_robust",
                "seed": 17,
                "clean_weighted_f1": 0.67,
                "avg_perturbed_weighted_f1": 0.52,
            },
            {
                "experiment": "text_only",
                "seed": 13,
                "clean_weighted_f1": 0.40,
                "avg_perturbed_weighted_f1": 0.40,
            },
            {
                "experiment": "early_fusion",
                "seed": 13,
                "clean_weighted_f1": 0.42,
                "avg_perturbed_weighted_f1": 0.35,
            },
            {
                "experiment": "minus_gating",
                "seed": 13,
                "clean_weighted_f1": 0.69,
                "avg_perturbed_weighted_f1": 0.57,
            },
            {
                "experiment": "minus_modality_dropout",
                "seed": 13,
                "clean_weighted_f1": 0.685,
                "avg_perturbed_weighted_f1": 0.565,
            },
            {
                "experiment": "minus_jitter_augmentation",
                "seed": 13,
                "clean_weighted_f1": 0.66,
                "avg_perturbed_weighted_f1": 0.54,
            },
        ]
    )
    aggregate_df = pd.DataFrame(
        [
            {"experiment": "xmodal_transformer", "condition": "clean", "weighted_f1": 0.69},
            {"experiment": "xmodal_transformer", "condition": "missing_audio", "weighted_f1": 0.50},
            {"experiment": "xmodal_transformer", "condition": "missing_vision", "weighted_f1": 0.51},
            {"experiment": "xmodal_transformer", "condition": "missing_audio_vision", "weighted_f1": 0.45},
            {"experiment": "xmodal_transformer", "condition": "mild_jitter", "weighted_f1": 0.50},
            {"experiment": "xmodal_transformer_robust", "condition": "clean", "weighted_f1": 0.675},
            {"experiment": "xmodal_transformer_robust", "condition": "missing_audio", "weighted_f1": 0.56},
            {"experiment": "xmodal_transformer_robust", "condition": "missing_vision", "weighted_f1": 0.55},
            {"experiment": "xmodal_transformer_robust", "condition": "missing_audio_vision", "weighted_f1": 0.53},
            {"experiment": "xmodal_transformer_robust", "condition": "mild_jitter", "weighted_f1": 0.54},
        ]
    )
    acceptance = acceptance_summary(summary_df, clean_gap_tolerance=0.01, required_positive_seeds=2)

    report = render_final_report(summary_df, aggregate_df, acceptance)

    assert "partially supported" in report
    assert "2/2 matched seeds" in report
    assert "0/2 seeds stayed within the allowed 0.010 weighted-F1 drop" in report
    assert "`minus_gating`" in report
    assert "single-seed only" in report
