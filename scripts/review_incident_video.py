#!/usr/bin/env python
"""Generate a structured incident/safety report from one uploaded video."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multimod.video_review import review_video_with_gemini


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Path to the video file to review.")
    parser.add_argument(
        "--focus",
        default="safety",
        choices=("safety", "incident", "compliance", "operations"),
        help="Primary review lens for the report.",
    )
    parser.add_argument(
        "--checklist",
        action="append",
        default=[],
        help="Optional checklist item. Repeat the flag to add multiple items.",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model name to use for video review.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to <video_stem>.incident_report.json next to the video.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    output_path = (
        Path(args.output)
        if args.output is not None
        else video_path.with_name(f"{video_path.stem}.incident_report.json")
    )

    report = review_video_with_gemini(
        video_path=video_path,
        focus=args.focus,
        checklist=args.checklist,
        model=args.model,
    )
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
