#!/usr/bin/env python
"""Run one MOSEI experiment preset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multimod.config import available_experiments, make_experiment_config
from multimod.training import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to packed mosei_raw.pkl file.")
    parser.add_argument(
        "--experiment",
        required=True,
        choices=available_experiments(),
        help="Experiment preset to run.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed override for this run.")
    parser.add_argument("--output", default="outputs", help="Output directory root.")
    parser.add_argument("--device", default="auto", help="Device name, for example auto/cpu/cuda.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = make_experiment_config(
        experiment_name=args.experiment,
        data_path=args.data,
        output_dir=args.output,
    )
    seed = args.seed if args.seed is not None else config.training.seeds[0]
    result = run_experiment(config=config, seed=seed, device_name=args.device)
    summary = result["summary"]
    print(f"Experiment: {args.experiment}")
    print(f"Seed: {seed}")
    print(f"Checkpoint: {Path(result['run']['checkpoint'])}")
    print(f"Clean weighted F1: {summary['clean_weighted_f1']:.4f}")
    print(f"Average perturbed weighted F1: {summary['avg_perturbed_weighted_f1']:.4f}")


if __name__ == "__main__":
    main()
