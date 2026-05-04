#!/usr/bin/env python
"""Run the EIDMSA study: full model plus ablations and comparisons."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multimod.config import make_experiment_config
from multimod.training import run_experiment
from multimod.utils import ensure_dir, save_json


# Core EIDMSA experiments
EIDMSA_EXPERIMENTS = (
    "eidmsa",
)

# EIDMSA ablations (single seed each)
EIDMSA_ABLATIONS = (
    "eidmsa_no_ib",
    "eidmsa_no_pid",
    "eidmsa_no_evidential",
)

# Comparison baselines (already run in main_run, but included for convenience)
COMPARISON_BASELINES = (
    "xmodal_transformer",
    "xmodal_transformer_robust",
)

# Novel paper integrations
NOVEL_EXPERIMENTS = (
    "eidmsa_kan",
    "eidmsa_mamba",
    "eidmsa_kan_mamba",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to packed mosei_raw.pkl file.")
    parser.add_argument("--output", default="outputs/eidmsa_run", help="Output directory root.")
    parser.add_argument("--device", default="auto", help="Device name, for example auto/cpu/cuda.")
    parser.add_argument("--run-ablations", action="store_true", help="Run the EIDMSA ablations.")
    parser.add_argument("--run-baselines", action="store_true", help="Re-run transformer baselines for comparison.")
    parser.add_argument("--run-7class", action="store_true", help="Run EIDMSA on 7-class sentiment.")
    parser.add_argument("--run-tta", action="store_true", help="Run EIDMSA with test-time adaptation.")
    parser.add_argument("--run-kan", action="store_true", help="Run EIDMSA with KAN projection heads.")
    parser.add_argument("--run-mamba", action="store_true", help="Run EIDMSA with Mamba SSM encoder.")
    parser.add_argument("--run-novel", action="store_true", help="Run all novel paper integrations (KAN + Mamba + combined).")
    return parser.parse_args()


def _result_to_rows(result: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run = result["run"]
    summary = result["summary"]
    condition_rows = []
    for row in result["conditions"]:
        condition_rows.append({**run, **summary, **row})
    summary_row = {**run, **summary}
    return condition_rows, summary_row


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output)
    all_condition_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    experiments_to_run: list[str] = list(EIDMSA_EXPERIMENTS)

    if args.run_ablations:
        experiments_to_run.extend(EIDMSA_ABLATIONS)

    if args.run_baselines:
        experiments_to_run.extend(COMPARISON_BASELINES)

    if args.run_7class:
        experiments_to_run.append("eidmsa_7class")

    if args.run_tta:
        experiments_to_run.append("eidmsa_tta")

    if args.run_kan:
        experiments_to_run.append("eidmsa_kan")

    if args.run_mamba:
        experiments_to_run.append("eidmsa_mamba")

    if args.run_novel:
        experiments_to_run.extend(NOVEL_EXPERIMENTS)

    # Deduplicate while preserving first-seen order
    seen: set[str] = set()
    experiments_to_run = [
        e for e in experiments_to_run if not (e in seen or seen.add(e))
    ]

    for experiment_name in experiments_to_run:
        print(f"\n{'='*60}")
        print(f"Running: {experiment_name}")
        print(f"{'='*60}")

        config = make_experiment_config(
            experiment_name,
            data_path=args.data,
            output_dir=str(output_dir),
        )

        for seed in config.training.seeds:
            print(f"\n  Seed: {seed}")
            result = run_experiment(config=config, seed=seed, device_name=args.device)
            condition_rows, summary_row = _result_to_rows(result)
            all_condition_rows.extend(condition_rows)
            summary_rows.append(summary_row)

            # Print summary
            summary = result["summary"]
            print(f"  Clean F1: {summary['clean_weighted_f1']:.4f}")
            print(f"  Avg Perturbed F1: {summary['avg_perturbed_weighted_f1']:.4f}")
            if "clean_uncertainty" in summary:
                print(f"  Clean Uncertainty: {summary['clean_uncertainty']:.4f}")
                print(f"  Clean ECE: {summary['clean_ece']:.4f}")
                print(f"  Avg Perturbed Uncertainty: {summary['avg_perturbed_uncertainty']:.4f}")

    # Save aggregated results
    condition_df = pd.DataFrame(all_condition_rows)
    summary_df = pd.DataFrame(summary_rows)
    condition_df.to_csv(output_dir / "eidmsa_aggregate_results.csv", index=False)
    summary_df.to_csv(output_dir / "eidmsa_run_summary.csv", index=False)

    print(f"\n{'='*60}")
    print(f"Results saved to {output_dir}")
    print(f"  Aggregate: {output_dir / 'eidmsa_aggregate_results.csv'}")
    print(f"  Summary:   {output_dir / 'eidmsa_run_summary.csv'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
