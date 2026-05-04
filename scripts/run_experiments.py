#!/usr/bin/env python
"""Run the planned MOSEI study and aggregate the outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multimod.config import make_experiment_config
from multimod.reporting import acceptance_summary, write_final_report
from multimod.training import run_experiment
from multimod.utils import ensure_dir, save_json


MAIN_EXPERIMENTS = (
    "text_only",
    "early_fusion",
    "xmodal_transformer",
    "xmodal_transformer_robust",
)
ABLATIONS = (
    "minus_gating",
    "minus_modality_dropout",
    "minus_jitter_augmentation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to packed mosei_raw.pkl file.")
    parser.add_argument("--output", default="outputs/main_run", help="Output directory root.")
    parser.add_argument("--device", default="auto", help="Device name, for example auto/cpu/cuda.")
    parser.add_argument("--run-ablations", action="store_true", help="Run the robust-model ablations.")
    parser.add_argument(
        "--clean-gap-tolerance",
        type=float,
        default=0.01,
        help=(
            "Maximum allowed drop in clean weighted F1 for the robust transformer versus vanilla. "
            "Weighted F1 is on a 0.0-1.0 scale, so 1 F1 point = 0.01."
        ),
    )
    parser.add_argument(
        "--required-positive-seeds",
        type=int,
        default=2,
        help="Minimum number of seeds where robust average perturbed F1 must beat vanilla.",
    )
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

    for experiment_name in MAIN_EXPERIMENTS:
        config = make_experiment_config(experiment_name, data_path=args.data, output_dir=str(output_dir))
        for seed in config.training.seeds:
            result = run_experiment(config=config, seed=seed, device_name=args.device)
            condition_rows, summary_row = _result_to_rows(result)
            all_condition_rows.extend(condition_rows)
            summary_rows.append(summary_row)

    if args.run_ablations:
        for experiment_name in ABLATIONS:
            config = make_experiment_config(experiment_name, data_path=args.data, output_dir=str(output_dir))
            seed = config.training.seeds[0]
            result = run_experiment(config=config, seed=seed, device_name=args.device)
            condition_rows, summary_row = _result_to_rows(result)
            all_condition_rows.extend(condition_rows)
            summary_rows.append(summary_row)

    condition_df = pd.DataFrame(all_condition_rows)
    summary_df = pd.DataFrame(summary_rows)
    condition_df.to_csv(output_dir / "aggregate_results.csv", index=False)
    summary_df.to_csv(output_dir / "run_summary.csv", index=False)
    acceptance = acceptance_summary(
        summary_df,
        clean_gap_tolerance=args.clean_gap_tolerance,
        required_positive_seeds=args.required_positive_seeds,
    )
    save_json(acceptance, output_dir / "acceptance_summary.json")
    report_path = write_final_report(
        summary_df=summary_df,
        aggregate_df=condition_df,
        acceptance=acceptance,
        output_path=output_dir / "final_report.md",
    )

    print(f"Wrote aggregate condition metrics to {output_dir / 'aggregate_results.csv'}")
    print(f"Wrote run-level summaries to {output_dir / 'run_summary.csv'}")
    print(f"Wrote acceptance summary to {output_dir / 'acceptance_summary.json'}")
    print(f"Wrote final markdown report to {report_path}")


if __name__ == "__main__":
    main()
