#!/usr/bin/env python
"""Build a single defensible project claim with proof tables and limits."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ClaimThresholds:
    min_matched_seeds: int = 3
    max_clean_gap: float = 0.02
    max_perturbed_gap: float = 0.02
    min_parameter_reduction: float = 0.60
    min_checkpoint_reduction: float = 0.60


@dataclass
class RunRecord:
    seed: int
    clean_weighted_f1: float
    avg_perturbed_weighted_f1: float
    clean_accuracy: float
    num_parameters: int
    checkpoint_mb: float
    checkpoint_path: str
    metrics_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compact-root",
        default="outputs/eidmsa_gpu_final/eidmsa",
        help="Run directory for the compact model claim target.",
    )
    parser.add_argument(
        "--compact-label",
        default="EIDMSA",
        help="Display label for the compact model.",
    )
    parser.add_argument(
        "--baseline-root",
        default="outputs/main_run/xmodal_transformer_robust",
        help="Run directory for the comparison baseline.",
    )
    parser.add_argument(
        "--baseline-label",
        default="Robust Transformer",
        help="Display label for the baseline.",
    )
    parser.add_argument(
        "--output",
        default="outputs/claim_pack",
        help="Directory for claim artifacts.",
    )
    parser.add_argument(
        "--min-matched-seeds",
        type=int,
        default=3,
        help="Minimum number of matched seeds required for the claim.",
    )
    parser.add_argument(
        "--max-clean-gap",
        type=float,
        default=0.02,
        help="Maximum absolute clean weighted-F1 gap allowed per matched seed.",
    )
    parser.add_argument(
        "--max-perturbed-gap",
        type=float,
        default=0.02,
        help="Maximum absolute perturbed weighted-F1 gap allowed per matched seed.",
    )
    parser.add_argument(
        "--min-parameter-reduction",
        type=float,
        default=0.60,
        help="Minimum relative parameter reduction required for the claim.",
    )
    parser.add_argument(
        "--min-checkpoint-reduction",
        type=float,
        default=0.60,
        help="Minimum relative checkpoint-size reduction required for the claim.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _clean_accuracy(payload: dict[str, Any]) -> float:
    for row in payload.get("conditions", []):
        if row.get("condition") == "clean":
            return float(row["accuracy"])
    raise KeyError("Clean condition accuracy is missing.")


def load_run_records(run_root: str | Path) -> list[RunRecord]:
    run_root = Path(run_root)
    records: list[RunRecord] = []
    if not run_root.exists():
        return records

    for seed_dir in sorted(path for path in run_root.iterdir() if path.is_dir() and path.name.startswith("seed_")):
        try:
            seed = int(seed_dir.name.split("_", maxsplit=1)[1])
        except ValueError:
            continue
        metrics_path = seed_dir / "metrics.json"
        checkpoint_path = seed_dir / "best_model.pt"
        if not metrics_path.exists() or not checkpoint_path.exists():
            continue
        payload = _read_json(metrics_path)
        summary = payload["summary"]
        run = payload["run"]
        records.append(
            RunRecord(
                seed=seed,
                clean_weighted_f1=float(summary["clean_weighted_f1"]),
                avg_perturbed_weighted_f1=float(summary["avg_perturbed_weighted_f1"]),
                clean_accuracy=_clean_accuracy(payload),
                num_parameters=int(run["num_parameters"]),
                checkpoint_mb=checkpoint_path.stat().st_size / (1024 * 1024),
                checkpoint_path=str(checkpoint_path),
                metrics_path=str(metrics_path),
            )
        )
    return records


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def build_claim_payload(
    compact_rows: list[RunRecord],
    baseline_rows: list[RunRecord],
    compact_label: str,
    baseline_label: str,
    thresholds: ClaimThresholds,
) -> dict[str, Any]:
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
                f"clean_weighted_f1_{compact_label}": compact.clean_weighted_f1,
                f"clean_weighted_f1_{baseline_label}": baseline.clean_weighted_f1,
                "clean_gap": compact.clean_weighted_f1 - baseline.clean_weighted_f1,
                f"avg_perturbed_weighted_f1_{compact_label}": compact.avg_perturbed_weighted_f1,
                f"avg_perturbed_weighted_f1_{baseline_label}": baseline.avg_perturbed_weighted_f1,
                "perturbed_gap": compact.avg_perturbed_weighted_f1 - baseline.avg_perturbed_weighted_f1,
                f"clean_accuracy_{compact_label}": compact.clean_accuracy,
                f"clean_accuracy_{baseline_label}": baseline.clean_accuracy,
                "clean_accuracy_gap": compact.clean_accuracy - baseline.clean_accuracy,
            }
        )

    mean_compact_clean = _mean([compact_by_seed[seed].clean_weighted_f1 for seed in matched_seeds])
    mean_baseline_clean = _mean([baseline_by_seed[seed].clean_weighted_f1 for seed in matched_seeds])
    mean_compact_perturbed = _mean([compact_by_seed[seed].avg_perturbed_weighted_f1 for seed in matched_seeds])
    mean_baseline_perturbed = _mean([baseline_by_seed[seed].avg_perturbed_weighted_f1 for seed in matched_seeds])
    mean_compact_accuracy = _mean([compact_by_seed[seed].clean_accuracy for seed in matched_seeds])
    mean_baseline_accuracy = _mean([baseline_by_seed[seed].clean_accuracy for seed in matched_seeds])
    mean_compact_params = _mean([compact_by_seed[seed].num_parameters for seed in matched_seeds])
    mean_baseline_params = _mean([baseline_by_seed[seed].num_parameters for seed in matched_seeds])
    mean_compact_ckpt = _mean([compact_by_seed[seed].checkpoint_mb for seed in matched_seeds])
    mean_baseline_ckpt = _mean([baseline_by_seed[seed].checkpoint_mb for seed in matched_seeds])

    clean_retention = mean_compact_clean / mean_baseline_clean
    perturbed_retention = mean_compact_perturbed / mean_baseline_perturbed
    parameter_reduction = 1.0 - (mean_compact_params / mean_baseline_params)
    checkpoint_reduction = 1.0 - (mean_compact_ckpt / mean_baseline_ckpt)
    worst_clean_gap = max(abs(row["clean_gap"]) for row in matched_rows)
    worst_perturbed_gap = max(abs(row["perturbed_gap"]) for row in matched_rows)

    checks = [
        {
            "name": "matched_seed_count",
            "value": len(matched_seeds),
            "threshold": thresholds.min_matched_seeds,
            "comparison": ">=",
            "passed": len(matched_seeds) >= thresholds.min_matched_seeds,
        },
        {
            "name": "worst_clean_gap",
            "value": worst_clean_gap,
            "threshold": thresholds.max_clean_gap,
            "comparison": "<=",
            "passed": worst_clean_gap <= thresholds.max_clean_gap,
        },
        {
            "name": "worst_perturbed_gap",
            "value": worst_perturbed_gap,
            "threshold": thresholds.max_perturbed_gap,
            "comparison": "<=",
            "passed": worst_perturbed_gap <= thresholds.max_perturbed_gap,
        },
        {
            "name": "parameter_reduction",
            "value": parameter_reduction,
            "threshold": thresholds.min_parameter_reduction,
            "comparison": ">=",
            "passed": parameter_reduction >= thresholds.min_parameter_reduction,
        },
        {
            "name": "checkpoint_reduction",
            "value": checkpoint_reduction,
            "threshold": thresholds.min_checkpoint_reduction,
            "comparison": ">=",
            "passed": checkpoint_reduction >= thresholds.min_checkpoint_reduction,
        },
    ]
    supported = all(check["passed"] for check in checks)

    claim_text = (
        f"Across {len(matched_seeds)} matched seeds on this repo's 3-class CMU-MOSEI robustness protocol, "
        f"{compact_label} stays within {thresholds.max_clean_gap:.2f} weighted-F1 of the {baseline_label} on clean evaluation "
        f"and within {thresholds.max_perturbed_gap:.2f} weighted-F1 on average perturbed evaluation, "
        f"while using {parameter_reduction * 100:.1f}% fewer parameters and {checkpoint_reduction * 100:.1f}% smaller checkpoints."
    )

    return {
        "claim": claim_text,
        "supported": supported,
        "compact_label": compact_label,
        "baseline_label": baseline_label,
        "thresholds": asdict(thresholds),
        "checks": checks,
        "summary": {
            "matched_seeds": matched_seeds,
            "mean_clean_weighted_f1_compact": mean_compact_clean,
            "mean_clean_weighted_f1_baseline": mean_baseline_clean,
            "mean_avg_perturbed_weighted_f1_compact": mean_compact_perturbed,
            "mean_avg_perturbed_weighted_f1_baseline": mean_baseline_perturbed,
            "mean_clean_accuracy_compact": mean_compact_accuracy,
            "mean_clean_accuracy_baseline": mean_baseline_accuracy,
            "clean_retention_ratio": clean_retention,
            "perturbed_retention_ratio": perturbed_retention,
            "parameter_reduction": parameter_reduction,
            "checkpoint_reduction": checkpoint_reduction,
            "worst_clean_gap": worst_clean_gap,
            "worst_perturbed_gap": worst_perturbed_gap,
            "mean_num_parameters_compact": mean_compact_params,
            "mean_num_parameters_baseline": mean_baseline_params,
            "mean_checkpoint_mb_compact": mean_compact_ckpt,
            "mean_checkpoint_mb_baseline": mean_baseline_ckpt,
        },
        "matched_seed_rows": matched_rows,
        "progress": [
            "Claim locked to efficiency-under-robustness, not SOTA.",
            "Matched-seed evidence assembled from saved runs.",
            "Pass/fail gates evaluated automatically.",
            "Limitations written explicitly so the claim stays defensible.",
        ],
        "limits": [
            "This is a custom 3-class CMU-MOSEI protocol, not a standard MOSEI SOTA claim.",
            "The perturbations are controlled synthetic stress tests, not real deployment failures.",
            "The evidence uses three matched seeds, which is useful but still small-sample.",
            "The claim is about efficiency with competitive robustness, not superior accuracy.",
        ],
        "can_say_now": [
            f"{compact_label} is substantially smaller than the {baseline_label}.",
            f"{compact_label} preserves most of the {baseline_label}'s clean and perturbed weighted-F1 on this protocol.",
            f"{compact_label} is a defensible compact alternative when parameter budget matters.",
        ],
        "cannot_say_now": [
            "current CMU-MOSEI SOTA",
            "better than the robust transformer on accuracy or robustness",
            "real-world robustness beyond the synthetic stress tests",
            "strong published-standard-metric superiority",
        ],
    }


def _format_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def render_claim_report(payload: dict[str, Any]) -> str:
    compact_label = payload["compact_label"]
    baseline_label = payload["baseline_label"]
    summary = payload["summary"]
    status = "SUPPORTED" if payload["supported"] else "NOT SUPPORTED"

    lines = [
        "# Defensible Claim Report",
        "",
        "## Claim",
        "",
        payload["claim"],
        "",
        "## Decision",
        "",
        f"Status: **{status}**",
        "",
        "## Evidence Gates",
        "",
        "| Check | Value | Threshold | Result |",
        "| --- | ---: | ---: | --- |",
    ]
    for check in payload["checks"]:
        lines.append(
            f"| {check['name']} | {_format_float(check['value']) if isinstance(check['value'], float) else check['value']} "
            f"| {check['comparison']} {_format_float(check['threshold']) if isinstance(check['threshold'], float) else check['threshold']} "
            f"| {'PASS' if check['passed'] else 'FAIL'} |"
        )

    lines.extend(
        [
            "",
            "## Mean Summary",
            "",
            f"- `{compact_label}` clean weighted F1: {_format_float(summary['mean_clean_weighted_f1_compact'])}",
            f"- `{baseline_label}` clean weighted F1: {_format_float(summary['mean_clean_weighted_f1_baseline'])}",
            f"- `{compact_label}` avg perturbed weighted F1: {_format_float(summary['mean_avg_perturbed_weighted_f1_compact'])}",
            f"- `{baseline_label}` avg perturbed weighted F1: {_format_float(summary['mean_avg_perturbed_weighted_f1_baseline'])}",
            f"- Clean retention ratio: {_format_float(summary['clean_retention_ratio'] * 100, 2)}%",
            f"- Perturbed retention ratio: {_format_float(summary['perturbed_retention_ratio'] * 100, 2)}%",
            f"- Parameter reduction: {_format_float(summary['parameter_reduction'] * 100, 1)}%",
            f"- Checkpoint reduction: {_format_float(summary['checkpoint_reduction'] * 100, 1)}%",
            "",
            "## Matched Seed Proof",
            "",
            "| seed | compact_clean_f1 | baseline_clean_f1 | clean_gap | compact_perturbed_f1 | baseline_perturbed_f1 | perturbed_gap |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["matched_seed_rows"]:
        lines.append(
            f"| {row['seed']} | "
            f"{_format_float(row[f'clean_weighted_f1_{compact_label}'])} | "
            f"{_format_float(row[f'clean_weighted_f1_{baseline_label}'])} | "
            f"{_format_float(row['clean_gap'])} | "
            f"{_format_float(row[f'avg_perturbed_weighted_f1_{compact_label}'])} | "
            f"{_format_float(row[f'avg_perturbed_weighted_f1_{baseline_label}'])} | "
            f"{_format_float(row['perturbed_gap'])} |"
        )

    lines.extend(
        [
            "",
            "## Proof Statement",
            "",
            (
                f"The strongest clean claim is efficiency: `{compact_label}` keeps "
                f"{_format_float(summary['clean_retention_ratio'] * 100, 2)}% of the `{baseline_label}` clean weighted F1 "
                f"and {_format_float(summary['perturbed_retention_ratio'] * 100, 2)}% of its perturbed weighted F1, "
                f"while cutting parameter count by {_format_float(summary['parameter_reduction'] * 100, 1)}%."
            ),
            "",
            "## Progress",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in payload["progress"])
    lines.extend(["", "## Limits", ""])
    lines.extend(f"- {item}" for item in payload["limits"])
    lines.extend(["", "## Can Say Now", ""])
    lines.extend(f"- {item}" for item in payload["can_say_now"])
    lines.extend(["", "## Cannot Say Now", ""])
    lines.extend(f"- {item}" for item in payload["cannot_say_now"])
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    thresholds = ClaimThresholds(
        min_matched_seeds=args.min_matched_seeds,
        max_clean_gap=args.max_clean_gap,
        max_perturbed_gap=args.max_perturbed_gap,
        min_parameter_reduction=args.min_parameter_reduction,
        min_checkpoint_reduction=args.min_checkpoint_reduction,
    )
    compact_rows = load_run_records(PROJECT_ROOT / args.compact_root)
    baseline_rows = load_run_records(PROJECT_ROOT / args.baseline_root)
    if not compact_rows:
        raise SystemExit(f"No compact runs found under {args.compact_root}")
    if not baseline_rows:
        raise SystemExit(f"No baseline runs found under {args.baseline_root}")

    payload = build_claim_payload(
        compact_rows=compact_rows,
        baseline_rows=baseline_rows,
        compact_label=args.compact_label,
        baseline_label=args.baseline_label,
        thresholds=thresholds,
    )

    output_dir = PROJECT_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "claim_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "claim_report.md").write_text(render_claim_report(payload), encoding="utf-8")
    write_csv(output_dir / "matched_seed_comparison.csv", payload["matched_seed_rows"])

    print(output_dir / "claim_summary.json")
    print(output_dir / "claim_report.md")
    print(output_dir / "matched_seed_comparison.csv")


if __name__ == "__main__":
    main()
