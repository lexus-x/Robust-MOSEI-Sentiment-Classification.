#!/usr/bin/env python
"""Run the realistic robustness protocol on saved multimod checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "v2.0" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from real_sentiment import (
    build_benchmark_manifest,
    build_bootstrap_evidence,
    build_default_thesis_claim,
    build_protocol_specs,
    compare_roles,
    render_realistic_benchmark_report,
    run_protocol_for_root,
)
from real_sentiment.reporting import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compact-root",
        default="outputs/eidmsa_gpu_final/eidmsa",
        help="Root directory for compact model runs.",
    )
    parser.add_argument(
        "--baseline-root",
        default="outputs/main_run/xmodal_transformer_robust",
        help="Root directory for baseline model runs.",
    )
    parser.add_argument(
        "--output-dir",
        default="v2.0/outputs/realistic_benchmark_pack",
        help="Directory to write the benchmark artifacts into.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device name, for example auto/cpu/cuda.",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=None,
        help="Optional limit for quick smoke runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    claim = build_default_thesis_claim().to_dict()
    benchmark_manifest = build_benchmark_manifest()
    protocol_specs = build_protocol_specs()

    compact_condition_df, compact_prediction_df, compact_run_df = run_protocol_for_root(
        run_root=args.compact_root,
        role="compact",
        specs=protocol_specs,
        device_name=args.device,
        max_seeds=args.max_seeds,
    )
    baseline_condition_df, baseline_prediction_df, baseline_run_df = run_protocol_for_root(
        run_root=args.baseline_root,
        role="baseline",
        specs=protocol_specs,
        device_name=args.device,
        max_seeds=args.max_seeds,
    )

    condition_df = pd.concat([compact_condition_df, baseline_condition_df], ignore_index=True)
    prediction_df = pd.concat([compact_prediction_df, baseline_prediction_df], ignore_index=True)
    run_df = pd.concat([compact_run_df, baseline_run_df], ignore_index=True)

    comparison_summary = compare_roles(condition_df=condition_df, run_df=run_df)
    bootstrap_evidence = build_bootstrap_evidence(
        repo_root=PROJECT_ROOT,
        compact_root=args.compact_root,
        baseline_root=args.baseline_root,
    )

    model_condition_summary = (
        condition_df.groupby(["role", "condition_label", "condition_name", "family", "severity"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["role", "condition_name", "severity"])
    )
    family_summary = (
        model_condition_summary.groupby(["role", "family"], as_index=False)
        .agg(
            weighted_f1=("weighted_f1", "mean"),
            ece=("ece", "mean"),
            selective_risk_80=("selective_risk_80", "mean"),
            coverage_at_risk_20=("coverage_at_risk_20", "mean"),
        )
        .sort_values(["role", "family"])
    )
    report = render_realistic_benchmark_report(
        claim=claim,
        benchmark_manifest=benchmark_manifest,
        bootstrap_evidence=bootstrap_evidence,
        comparison_summary=comparison_summary,
    )

    condition_df.to_csv(output_dir / "condition_metrics.csv", index=False)
    prediction_df.to_csv(output_dir / "predictions.csv", index=False)
    run_df.to_csv(output_dir / "run_summary.csv", index=False)
    model_condition_summary.to_csv(output_dir / "model_condition_summary.csv", index=False)
    family_summary.to_csv(output_dir / "family_summary.csv", index=False)
    write_json(output_dir / "benchmark_manifest.json", benchmark_manifest)
    write_json(output_dir / "thesis_claim.json", claim)
    write_json(output_dir / "bootstrap_evidence.json", bootstrap_evidence)
    write_json(output_dir / "comparison_summary.json", comparison_summary)
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    print(output_dir / "report.md")


if __name__ == "__main__":
    main()
