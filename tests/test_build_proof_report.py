from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def _load_build_proof_report_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_proof_report.py"
    spec = importlib.util.spec_from_file_location("build_proof_report", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_metrics(run_dir: Path, clean_f1: float, perturbed_f1: float, accuracy: float) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "best_model.pt").write_bytes(b"checkpoint")
    payload = {
        "summary": {
            "clean_weighted_f1": clean_f1,
            "avg_perturbed_weighted_f1": perturbed_f1,
        },
        "conditions": [
            {
                "condition": "clean",
                "accuracy": accuracy,
            }
        ],
        "run": {
            "num_parameters": 123,
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def test_load_local_rows_prefers_final_runs_over_fast_runs(tmp_path):
    module = _load_build_proof_report_module()
    module.PROJECT_ROOT = tmp_path

    _write_metrics(
        tmp_path / "outputs" / "eidmsa_gpu_fast" / "eidmsa_kan" / "seed_13",
        clean_f1=0.60,
        perturbed_f1=0.59,
        accuracy=0.61,
    )
    _write_metrics(
        tmp_path / "outputs" / "eidmsa_gpu_final" / "eidmsa_kan" / "seed_13",
        clean_f1=0.62,
        perturbed_f1=0.61,
        accuracy=0.63,
    )
    _write_metrics(
        tmp_path / "outputs" / "eidmsa_gpu_final" / "eidmsa_kan" / "seed_17",
        clean_f1=0.63,
        perturbed_f1=0.62,
        accuracy=0.64,
    )

    source = next(item for item in module.LOCAL_SOURCES if item.experiment == "eidmsa_kan")
    rows = module.load_local_rows(source)

    assert len(rows) == 2
    assert {row["seed"] for row in rows} == {"seed_13", "seed_17"}
    assert {row["source_root"] for row in rows} == {"outputs/eidmsa_gpu_final"}
    assert {row["evidence_level"] for row in rows} == {"completed_2_seed"}
