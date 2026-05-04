#!/usr/bin/env python
"""Train EIDMSA robust v2 and run the realistic benchmark protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
SUBPROJECT_SRC = PROJECT_ROOT / "v2.0" / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SUBPROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(SUBPROJECT_SRC))

from multimod.config import make_experiment_config
from multimod.training import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        default="data/mosei_raw.pkl",
        help="Path to MOSEI data pickle.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/eidmsa_robust_v2",
        help="Output directory for checkpoints.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device name (auto/cpu/cuda).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[13, 17, 23],
        help="Seeds to train.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = make_experiment_config(
        experiment_name="eidmsa_robust_v2",
        data_path=args.data_path,
        output_dir=args.output_dir,
    )

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Training eidmsa_robust_v2 | seed={seed}")
        print(f"{'='*60}\n")
        results = run_experiment(config, seed=seed, device_name=args.device)
        summary = results["summary"]
        print(f"\n  Clean weighted-F1: {summary['clean_weighted_f1']:.4f}")
        print(f"  Avg perturbed weighted-F1: {summary['avg_perturbed_weighted_f1']:.4f}")
        if "clean_ece" in summary:
            print(f"  Clean ECE: {summary['clean_ece']:.4f}")
        print(f"  Parameters: {results['run']['num_parameters']}")

    print(f"\n{'='*60}")
    print("All seeds complete. Checkpoints saved to:", args.output_dir)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
