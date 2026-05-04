from __future__ import annotations

import pandas as pd
import pytest
import torch

from real_sentiment.benchmark import (
    ProtocolSpec,
    apply_protocol_condition,
    build_protocol_specs,
    compare_roles,
    summarize_run_metrics,
)
from real_sentiment.claim import build_benchmark_manifest, build_default_thesis_claim
from real_sentiment.reporting import render_realistic_benchmark_report, render_thesis_report


def _sample_batch() -> dict[str, object]:
    mask = torch.tensor(
        [
            [False, False, True, True, True],
            [False, True, True, True, True],
        ]
    )
    audio = torch.arange(30, dtype=torch.float32).reshape(2, 5, 3)
    vision = torch.arange(30, 60, dtype=torch.float32).reshape(2, 5, 3)
    return {
        "text": torch.ones(2, 5, 4),
        "audio": audio,
        "vision": vision,
        "mask": mask,
        "label": torch.tensor([0, 2]),
        "raw_sentiment": torch.tensor([-1.0, 1.0]),
        "sample_id": ["a", "b"],
    }


def test_build_protocol_specs_expands_manifest():
    specs = build_protocol_specs()

    assert specs[0].label == "clean"
    assert len(specs) == 19
    assert any(spec.label == "lead_lag_audio::8_frames" for spec in specs)


def test_apply_protocol_condition_targets_only_requested_modality():
    batch = _sample_batch()
    spec = ProtocolSpec(
        name="block_missing_audio",
        family="missingness",
        severity="moderate",
        target_modalities=("audio",),
        description="drop audio",
    )

    conditioned = apply_protocol_condition(batch, spec)

    assert torch.equal(conditioned["vision"], batch["vision"])
    assert torch.equal(conditioned["mask"], batch["mask"])
    assert not torch.equal(conditioned["audio"], batch["audio"])


def test_apply_protocol_condition_compound_hits_both_modalities():
    batch = _sample_batch()
    spec = ProtocolSpec(
        name="compound_audio_vision_failure",
        family="compound",
        severity="severe",
        target_modalities=("audio", "vision"),
        description="compound",
    )

    conditioned = apply_protocol_condition(batch, spec)

    assert not torch.equal(conditioned["audio"], batch["audio"])
    assert not torch.equal(conditioned["vision"], batch["vision"])


def test_summarize_run_metrics_tracks_worst_condition():
    rows = [
        {
            "condition_label": "clean",
            "weighted_f1": 0.80,
            "accuracy": 0.81,
            "ece": 0.10,
            "selective_risk_80": 0.12,
            "coverage_at_risk_20": 0.75,
        },
        {
            "condition_label": "cond_a",
            "weighted_f1": 0.70,
            "accuracy": 0.71,
            "ece": 0.15,
            "selective_risk_80": 0.18,
            "coverage_at_risk_20": 0.65,
        },
        {
            "condition_label": "cond_b",
            "weighted_f1": 0.60,
            "accuracy": 0.63,
            "ece": 0.20,
            "selective_risk_80": 0.25,
            "coverage_at_risk_20": 0.55,
        },
    ]

    summary = summarize_run_metrics(rows)

    assert summary["clean_weighted_f1"] == 0.80
    assert summary["worst_condition_label"] == "cond_b"
    assert summary["avg_perturbed_weighted_f1"] == pytest.approx(0.65)


def test_compare_roles_refined_gates():
    """Test the refined gate logic with mean/worst-case split."""
    condition_df = pd.DataFrame(
        [
            {
                "role": "compact",
                "seed": 13,
                "condition_label": "clean",
                "condition_name": "clean",
                "family": "reference",
                "severity": "none",
                "weighted_f1": 0.80,
                "accuracy": 0.81,
                "ece": 0.05,
                "selective_risk_80": 0.12,
            },
            {
                "role": "baseline",
                "seed": 13,
                "condition_label": "clean",
                "condition_name": "clean",
                "family": "reference",
                "severity": "none",
                "weighted_f1": 0.81,
                "accuracy": 0.82,
                "ece": 0.11,
                "selective_risk_80": 0.13,
            },
            {
                "role": "compact",
                "seed": 13,
                "condition_label": "burst_noise_vision::mild",
                "condition_name": "burst_noise_vision",
                "family": "local_corruption",
                "severity": "mild",
                "weighted_f1": 0.78,
                "accuracy": 0.79,
                "ece": 0.06,
                "selective_risk_80": 0.14,
            },
            {
                "role": "baseline",
                "seed": 13,
                "condition_label": "burst_noise_vision::mild",
                "condition_name": "burst_noise_vision",
                "family": "local_corruption",
                "severity": "mild",
                "weighted_f1": 0.79,
                "accuracy": 0.80,
                "ece": 0.15,
                "selective_risk_80": 0.16,
            },
            {
                "role": "compact",
                "seed": 13,
                "condition_label": "lead_lag_audio::8_frames",
                "condition_name": "lead_lag_audio",
                "family": "temporal_shift",
                "severity": "8_frames",
                "weighted_f1": 0.76,
                "accuracy": 0.77,
                "ece": 0.07,
                "selective_risk_80": 0.20,
            },
            {
                "role": "baseline",
                "seed": 13,
                "condition_label": "lead_lag_audio::8_frames",
                "condition_name": "lead_lag_audio",
                "family": "temporal_shift",
                "severity": "8_frames",
                "weighted_f1": 0.78,
                "accuracy": 0.79,
                "ece": 0.18,
                "selective_risk_80": 0.22,
            },
        ]
    )
    run_df = pd.DataFrame(
        [
            {"role": "compact", "seed": 13},
            {"role": "baseline", "seed": 13},
        ]
    )

    comparison = compare_roles(condition_df=condition_df, run_df=run_df)

    # Mean gap = (0.01 + 0.02) / 2 = 0.015, passes 0.02 tolerance
    assert comparison["gate_mean_robustness"] is True
    # Worst gap = 0.02, passes 0.03 tolerance
    assert comparison["gate_worst_case_robustness"] is True
    # Compact ECE < baseline ECE
    assert comparison["gate_calibration"] is True
    # Clean gap = 0.01, passes 0.02 tolerance
    assert comparison["gate_clean"] is True
    # Full claim supported
    assert comparison["full_realistic_claim_supported"] is True


def test_compare_roles_fails_worst_case_over_003():
    """Test that worst-case gate fails when gap > 0.03."""
    condition_df = pd.DataFrame(
        [
            {
                "role": "compact", "seed": 13, "condition_label": "clean",
                "condition_name": "clean", "family": "reference", "severity": "none",
                "weighted_f1": 0.80, "accuracy": 0.81, "ece": 0.05, "selective_risk_80": 0.12,
            },
            {
                "role": "baseline", "seed": 13, "condition_label": "clean",
                "condition_name": "clean", "family": "reference", "severity": "none",
                "weighted_f1": 0.81, "accuracy": 0.82, "ece": 0.11, "selective_risk_80": 0.13,
            },
            {
                "role": "compact", "seed": 13, "condition_label": "block_missing_vision::severe",
                "condition_name": "block_missing_vision", "family": "missingness", "severity": "severe",
                "weighted_f1": 0.70, "accuracy": 0.71, "ece": 0.06, "selective_risk_80": 0.25,
            },
            {
                "role": "baseline", "seed": 13, "condition_label": "block_missing_vision::severe",
                "condition_name": "block_missing_vision", "family": "missingness", "severity": "severe",
                "weighted_f1": 0.74, "accuracy": 0.75, "ece": 0.15, "selective_risk_80": 0.20,
            },
        ]
    )
    run_df = pd.DataFrame([{"role": "compact", "seed": 13}, {"role": "baseline", "seed": 13}])

    comparison = compare_roles(condition_df=condition_df, run_df=run_df)

    # Worst gap = 0.04, exceeds 0.03
    assert comparison["gate_worst_case_robustness"] is False
    assert comparison["full_realistic_claim_supported"] is False


def test_render_thesis_report():
    """Test render_thesis_report produces expected sections."""
    claim = build_default_thesis_claim().to_dict()
    manifest = build_benchmark_manifest()
    bootstrap = {"bootstrap_claim_support": {"parameter_reduction": 0.67, "checkpoint_reduction": 0.66}}
    comparison = {
        "full_realistic_claim_supported": True,
        "partial_missingness_claim_supported": True,
        "gate_clean": True,
        "gate_mean_robustness": True,
        "gate_worst_case_robustness": True,
        "gate_calibration": True,
        "gate_abstention": False,
        "matched_seeds": [13, 17],
        "clean_gap_mean": 0.008,
        "avg_perturbed_gap_mean": 0.009,
        "avg_perturbed_gap_worst": 0.025,
        "mean_gap_tolerance": 0.02,
        "worst_gap_tolerance": 0.03,
        "mean_ece_delta": -0.03,
        "family_summary": [
            {"family": "reference", "compact_weighted_f1": 0.62, "baseline_weighted_f1": 0.63, "weighted_f1_gap_vs_baseline": 0.01},
        ],
    }

    report = render_thesis_report(
        claim=claim,
        benchmark_manifest=manifest,
        bootstrap_evidence=bootstrap,
        comparison_summary=comparison,
    )

    assert "SUPPORTED" in report
    assert "✅ PASS" in report
    assert "Acceptance Gate Results" in report


def test_compare_roles_and_report_render():
    """Legacy test updated for new gate structure."""
    condition_df = pd.DataFrame(
        [
            {
                "role": "compact", "seed": 13, "condition_label": "clean",
                "condition_name": "clean", "family": "reference", "severity": "none",
                "weighted_f1": 0.80, "accuracy": 0.81, "ece": 0.10, "selective_risk_80": 0.12,
            },
            {
                "role": "baseline", "seed": 13, "condition_label": "clean",
                "condition_name": "clean", "family": "reference", "severity": "none",
                "weighted_f1": 0.81, "accuracy": 0.82, "ece": 0.11, "selective_risk_80": 0.13,
            },
            {
                "role": "compact", "seed": 13, "condition_label": "burst_noise_vision::mild",
                "condition_name": "burst_noise_vision", "family": "local_corruption", "severity": "mild",
                "weighted_f1": 0.78, "accuracy": 0.79, "ece": 0.12, "selective_risk_80": 0.14,
            },
            {
                "role": "baseline", "seed": 13, "condition_label": "burst_noise_vision::mild",
                "condition_name": "burst_noise_vision", "family": "local_corruption", "severity": "mild",
                "weighted_f1": 0.79, "accuracy": 0.80, "ece": 0.15, "selective_risk_80": 0.16,
            },
            {
                "role": "compact", "seed": 13, "condition_label": "lead_lag_audio::8_frames",
                "condition_name": "lead_lag_audio", "family": "temporal_shift", "severity": "8_frames",
                "weighted_f1": 0.65, "accuracy": 0.66, "ece": 0.20, "selective_risk_80": 0.30,
            },
            {
                "role": "baseline", "seed": 13, "condition_label": "lead_lag_audio::8_frames",
                "condition_name": "lead_lag_audio", "family": "temporal_shift", "severity": "8_frames",
                "weighted_f1": 0.74, "accuracy": 0.75, "ece": 0.18, "selective_risk_80": 0.22,
            },
        ]
    )
    run_df = pd.DataFrame([{"role": "compact", "seed": 13}, {"role": "baseline", "seed": 13}])

    comparison = compare_roles(condition_df=condition_df, run_df=run_df)
    report = render_realistic_benchmark_report(
        claim=build_default_thesis_claim().to_dict(),
        benchmark_manifest=build_benchmark_manifest(),
        bootstrap_evidence={
            "bootstrap_claim_support": {"parameter_reduction": 0.67, "checkpoint_reduction": 0.66}
        },
        comparison_summary=comparison,
    )

    # Worst gap = 0.09 > 0.03, so full claim should fail
    assert comparison["full_realistic_claim_supported"] is False
    assert comparison["partial_missingness_claim_supported"] is True
    assert "PARTIALLY SUPPORTED" in report
    assert "temporal_shift" in report
