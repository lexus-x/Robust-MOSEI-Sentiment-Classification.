"""Bootstrap the PhD-track project from current repo evidence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunRecord:
    seed: int
    clean_weighted_f1: float
    avg_perturbed_weighted_f1: float
    clean_accuracy: float
    num_parameters: int
    checkpoint_mb: float
    checkpoint_path: str


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_run_records(run_root: str | Path) -> list[RunRecord]:
    run_root = Path(run_root)
    records: list[RunRecord] = []
    if not run_root.exists():
        return records
    for seed_dir in sorted(path for path in run_root.iterdir() if path.is_dir() and path.name.startswith("seed_")):
        metrics_path = seed_dir / "metrics.json"
        checkpoint_path = seed_dir / "best_model.pt"
        if not metrics_path.exists() or not checkpoint_path.exists():
            continue
        payload = _read_json(metrics_path)
        clean_accuracy = None
        for row in payload.get("conditions", []):
            if row.get("condition") == "clean":
                clean_accuracy = float(row["accuracy"])
                break
        if clean_accuracy is None:
            continue
        summary = payload["summary"]
        run = payload["run"]
        records.append(
            RunRecord(
                seed=int(seed_dir.name.split("_", maxsplit=1)[1]),
                clean_weighted_f1=float(summary["clean_weighted_f1"]),
                avg_perturbed_weighted_f1=float(summary["avg_perturbed_weighted_f1"]),
                clean_accuracy=clean_accuracy,
                num_parameters=int(run["num_parameters"]),
                checkpoint_mb=checkpoint_path.stat().st_size / (1024 * 1024),
                checkpoint_path=str(checkpoint_path),
            )
        )
    return records


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def build_bootstrap_evidence(
    repo_root: str | Path,
    compact_root: str = "outputs/eidmsa_gpu_final/eidmsa",
    baseline_root: str = "outputs/main_run/xmodal_transformer_robust",
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    compact_rows = load_run_records(repo_root / compact_root)
    baseline_rows = load_run_records(repo_root / baseline_root)
    compact_by_seed = {row.seed: row for row in compact_rows}
    baseline_by_seed = {row.seed: row for row in baseline_rows}
    matched_seeds = sorted(set(compact_by_seed) & set(baseline_by_seed))
    if not matched_seeds:
        raise ValueError("No matched seeds between compact and baseline runs.")

    matched_rows: list[dict[str, Any]] = []
    for seed in matched_seeds:
        compact = compact_by_seed[seed]
        baseline = baseline_by_seed[seed]
        matched_rows.append(
            {
                "seed": seed,
                "compact_clean_weighted_f1": compact.clean_weighted_f1,
                "baseline_clean_weighted_f1": baseline.clean_weighted_f1,
                "clean_gap": compact.clean_weighted_f1 - baseline.clean_weighted_f1,
                "compact_avg_perturbed_weighted_f1": compact.avg_perturbed_weighted_f1,
                "baseline_avg_perturbed_weighted_f1": baseline.avg_perturbed_weighted_f1,
                "perturbed_gap": compact.avg_perturbed_weighted_f1 - baseline.avg_perturbed_weighted_f1,
                "compact_clean_accuracy": compact.clean_accuracy,
                "baseline_clean_accuracy": baseline.clean_accuracy,
                "accuracy_gap": compact.clean_accuracy - baseline.clean_accuracy,
            }
        )

    compact_clean = _mean([compact_by_seed[seed].clean_weighted_f1 for seed in matched_seeds])
    baseline_clean = _mean([baseline_by_seed[seed].clean_weighted_f1 for seed in matched_seeds])
    compact_perturbed = _mean([compact_by_seed[seed].avg_perturbed_weighted_f1 for seed in matched_seeds])
    baseline_perturbed = _mean([baseline_by_seed[seed].avg_perturbed_weighted_f1 for seed in matched_seeds])
    compact_params = _mean([compact_by_seed[seed].num_parameters for seed in matched_seeds])
    baseline_params = _mean([baseline_by_seed[seed].num_parameters for seed in matched_seeds])
    compact_ckpt = _mean([compact_by_seed[seed].checkpoint_mb for seed in matched_seeds])
    baseline_ckpt = _mean([baseline_by_seed[seed].checkpoint_mb for seed in matched_seeds])

    return {
        "source": "existing_multimod_repo_outputs",
        "matched_seeds": matched_seeds,
        "compact_model": {
            "name": "EIDMSA",
            "run_root": str(repo_root / compact_root),
            "mean_clean_weighted_f1": compact_clean,
            "mean_avg_perturbed_weighted_f1": compact_perturbed,
            "mean_num_parameters": compact_params,
            "mean_checkpoint_mb": compact_ckpt,
            "runs": [asdict(row) for row in compact_rows],
        },
        "baseline_model": {
            "name": "Robust Transformer",
            "run_root": str(repo_root / baseline_root),
            "mean_clean_weighted_f1": baseline_clean,
            "mean_avg_perturbed_weighted_f1": baseline_perturbed,
            "mean_num_parameters": baseline_params,
            "mean_checkpoint_mb": baseline_ckpt,
            "runs": [asdict(row) for row in baseline_rows],
        },
        "bootstrap_claim_support": {
            "clean_retention_ratio": compact_clean / baseline_clean,
            "perturbed_retention_ratio": compact_perturbed / baseline_perturbed,
            "parameter_reduction": 1.0 - (compact_params / baseline_params),
            "checkpoint_reduction": 1.0 - (compact_ckpt / baseline_ckpt),
            "worst_clean_gap": max(abs(row["clean_gap"]) for row in matched_rows),
            "worst_perturbed_gap": max(abs(row["perturbed_gap"]) for row in matched_rows),
        },
        "matched_seed_rows": matched_rows,
    }
