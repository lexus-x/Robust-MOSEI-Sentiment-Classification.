#!/usr/bin/env python
"""Generate the PhD-track proposal pack."""

from __future__ import annotations

import sys
from pathlib import Path


SUBPROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SUBPROJECT_ROOT.parents[0]
SRC_ROOT = SUBPROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from real_sentiment.bootstrap import build_bootstrap_evidence
from real_sentiment.claim import build_benchmark_manifest, build_default_thesis_claim
from real_sentiment.reporting import render_proposal_report, write_json


def main() -> None:
    output_dir = SUBPROJECT_ROOT / "outputs" / "proposal_pack"
    output_dir.mkdir(parents=True, exist_ok=True)

    claim = build_default_thesis_claim().to_dict()
    benchmark_manifest = build_benchmark_manifest()
    bootstrap_evidence = build_bootstrap_evidence(REPO_ROOT)

    write_json(output_dir / "thesis_claim.json", claim)
    write_json(output_dir / "benchmark_manifest.json", benchmark_manifest)
    write_json(output_dir / "bootstrap_evidence.json", bootstrap_evidence)
    (output_dir / "report.md").write_text(
        render_proposal_report(
            claim=claim,
            benchmark_manifest=benchmark_manifest,
            bootstrap_evidence=bootstrap_evidence,
        ),
        encoding="utf-8",
    )

    print(output_dir / "thesis_claim.json")
    print(output_dir / "benchmark_manifest.json")
    print(output_dir / "bootstrap_evidence.json")
    print(output_dir / "report.md")


if __name__ == "__main__":
    main()
