#!/usr/bin/env python
"""Create slide-ready plots from the aggregated experiment CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multimod.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="Path to aggregate_results.csv")
    parser.add_argument("--output", required=True, help="Directory for generated plots")
    return parser.parse_args()


def plot_condition_bars(results: pd.DataFrame, output_dir: Path) -> None:
    grouped = (
        results.groupby(["experiment", "condition"], as_index=False)["weighted_f1"]
        .mean()
        .pivot(index="condition", columns="experiment", values="weighted_f1")
        .sort_index()
    )
    ax = grouped.plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel("Weighted F1")
    ax.set_title("Mean weighted F1 by condition")
    ax.set_ylim(0.0, min(1.0, grouped.max().max() + 0.1))
    plt.tight_layout()
    plt.savefig(output_dir / "condition_weighted_f1.png", dpi=200)
    plt.close()


def plot_degradation(summary: pd.DataFrame, output_dir: Path) -> None:
    target = summary[summary["experiment"].isin(["xmodal_transformer", "xmodal_transformer_robust"])].copy()
    degradation_columns = [
        "missing_audio_degradation",
        "missing_vision_degradation",
        "missing_audio_vision_degradation",
        "mild_jitter_degradation",
    ]
    mean_degradation = target.groupby("experiment")[degradation_columns].mean().T
    ax = mean_degradation.plot(kind="bar", figsize=(10, 5))
    ax.set_ylabel("Weighted F1 degradation from clean")
    ax.set_title("Robustness degradation of the transformer pair")
    plt.tight_layout()
    plt.savefig(output_dir / "transformer_degradation.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output)
    condition_df = pd.read_csv(args.results)
    summary_path = Path(args.results).with_name("run_summary.csv")
    summary_df = pd.read_csv(summary_path)
    plot_condition_bars(condition_df, output_dir)
    plot_degradation(summary_df, output_dir)
    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
