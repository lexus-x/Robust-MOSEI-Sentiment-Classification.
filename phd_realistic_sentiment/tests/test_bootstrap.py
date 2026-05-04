from __future__ import annotations

import json
from pathlib import Path

from real_sentiment.bootstrap import build_bootstrap_evidence


def _write_run(root: Path, seed: int, clean_f1: float, perturbed_f1: float, accuracy: float, params: int, ckpt_bytes: int):
    run_dir = root / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "best_model.pt").write_bytes(b"0" * ckpt_bytes)
    payload = {
        "summary": {
            "clean_weighted_f1": clean_f1,
            "avg_perturbed_weighted_f1": perturbed_f1,
        },
        "conditions": [
            {"condition": "clean", "accuracy": accuracy},
        ],
        "run": {
            "num_parameters": params,
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def test_bootstrap_evidence_builds_from_temp_repo(tmp_path):
    compact_root = tmp_path / "outputs" / "eidmsa_gpu_final" / "eidmsa"
    baseline_root = tmp_path / "outputs" / "main_run" / "xmodal_transformer_robust"
    _write_run(compact_root, 13, 0.61, 0.60, 0.61, 500_000, 2_000_000)
    _write_run(compact_root, 17, 0.62, 0.61, 0.62, 500_000, 2_000_000)
    _write_run(baseline_root, 13, 0.63, 0.62, 0.63, 1_500_000, 6_000_000)
    _write_run(baseline_root, 17, 0.64, 0.63, 0.64, 1_500_000, 6_000_000)

    evidence = build_bootstrap_evidence(tmp_path)

    assert evidence["matched_seeds"] == [13, 17]
    assert evidence["bootstrap_claim_support"]["parameter_reduction"] > 0.6
    assert len(evidence["matched_seed_rows"]) == 2
