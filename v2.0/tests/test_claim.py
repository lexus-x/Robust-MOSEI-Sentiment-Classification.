from __future__ import annotations

from real_sentiment.claim import build_benchmark_manifest, build_default_thesis_claim


def test_default_claim_has_forbidden_claims_and_gates():
    claim = build_default_thesis_claim()

    assert "current MOSEI SOTA" in claim.cannot_say_now
    assert len(claim.acceptance_gates) >= 7


def test_benchmark_manifest_contains_realistic_families():
    manifest = build_benchmark_manifest()
    names = {condition["name"] for condition in manifest["conditions"]}
    families = {condition["family"] for condition in manifest["conditions"]}

    assert "lead_lag_audio" in names
    assert "compound_audio_vision_failure" in names
    assert {"missingness", "temporal_shift", "temporal_drift", "compound"} <= families


def test_gates_include_refined_structure():
    claim = build_default_thesis_claim()
    gate_names = {gate.name for gate in claim.acceptance_gates}

    assert "mean_robustness_parity" in gate_names
    assert "worst_case_robustness" in gate_names
    assert "calibration_advantage" in gate_names
    assert "uncertainty_aware_abstention" in gate_names
    assert "efficiency" in gate_names
    assert "transfer" in gate_names
