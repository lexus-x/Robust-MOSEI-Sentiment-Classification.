from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_claim_report.py"
    spec = importlib.util.spec_from_file_location("build_claim_report", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_run(root: Path, seed: int, clean_f1: float, pert_f1: float, accuracy: float, params: int, ckpt_mb: float):
    run_dir = root / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best_model.pt"
    ckpt_path.write_bytes(b"0" * int(ckpt_mb * 1024 * 1024))
    payload = {
        "summary": {
            "clean_weighted_f1": clean_f1,
            "avg_perturbed_weighted_f1": pert_f1,
        },
        "conditions": [
            {"condition": "clean", "accuracy": accuracy},
        ],
        "run": {
            "num_parameters": params,
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def test_claim_payload_supported_for_compact_competitive_model(tmp_path):
    module = _load_module()
    compact_root = tmp_path / "compact"
    baseline_root = tmp_path / "baseline"

    for seed, compact_clean, compact_pert, base_clean, base_pert in [
        (13, 0.61, 0.60, 0.62, 0.615),
        (17, 0.625, 0.620, 0.629, 0.625),
        (23, 0.630, 0.624, 0.635, 0.632),
    ]:
        _write_run(compact_root, seed, compact_clean, compact_pert, compact_clean, 500_000, 2.1)
        _write_run(baseline_root, seed, base_clean, base_pert, base_clean, 1_700_000, 6.5)

    payload = module.build_claim_payload(
        compact_rows=module.load_run_records(compact_root),
        baseline_rows=module.load_run_records(baseline_root),
        compact_label="Compact",
        baseline_label="Baseline",
        thresholds=module.ClaimThresholds(),
    )

    assert payload["supported"] is True
    assert "Compact stays within 0.02 weighted-F1" in payload["claim"]
    assert payload["summary"]["parameter_reduction"] > 0.6
    assert payload["summary"]["worst_clean_gap"] < 0.02
    assert payload["summary"]["worst_perturbed_gap"] < 0.02
