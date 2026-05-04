#!/usr/bin/env python
"""Build a final markdown report from completed experiment outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multimod.reporting import load_acceptance_summary, write_final_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="Path to run_summary.csv")
    parser.add_argument("--aggregate", required=True, help="Path to aggregate_results.csv")
    parser.add_argument("--acceptance", required=True, help="Path to acceptance_summary.json")
    parser.add_argument("--output", required=True, help="Path to final markdown report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_df = pd.read_csv(args.summary)
    aggregate_df = pd.read_csv(args.aggregate)
    acceptance = load_acceptance_summary(args.acceptance)
    report_path = write_final_report(
        summary_df=summary_df,
        aggregate_df=aggregate_df,
        acceptance=acceptance,
        output_path=args.output,
    )
    print(f"Wrote final markdown report to {report_path}")


if __name__ == "__main__":
    main()
