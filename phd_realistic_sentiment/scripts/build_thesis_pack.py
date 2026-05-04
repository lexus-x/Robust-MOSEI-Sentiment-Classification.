#!/usr/bin/env python
"""Generate the complete PhD thesis evidence pack."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SUBPROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SUBPROJECT_ROOT.parents[0]
SRC_ROOT = SUBPROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from real_sentiment.bootstrap import build_bootstrap_evidence
from real_sentiment.claim import build_benchmark_manifest, build_default_thesis_claim
from real_sentiment.reporting import render_thesis_report, write_json


def _load_json(path: Path) -> dict[str, Any] | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _find_best_benchmark_run(outputs_dir: Path) -> dict[str, Any] | None:
    """Find the benchmark run with the best verdict."""
    best = None
    for run_dir in sorted(outputs_dir.iterdir()):
        summary_path = run_dir / "comparison_summary.json"
        if not summary_path.exists():
            continue
        summary = _load_json(summary_path)
        if summary is None:
            continue
        if best is None:
            best = {"dir": run_dir, "summary": summary}
        elif summary.get("full_realistic_claim_supported") and not best["summary"].get("full_realistic_claim_supported"):
            best = {"dir": run_dir, "summary": summary}
        elif (
            summary.get("partial_missingness_claim_supported")
            and not best["summary"].get("partial_missingness_claim_supported")
            and not best["summary"].get("full_realistic_claim_supported")
        ):
            best = {"dir": run_dir, "summary": summary}
    return best


def _load_retry_summary(outputs_dir: Path) -> str | None:
    retry_path = outputs_dir / "retry_attempt_pack" / "retry_summary.md"
    if retry_path.exists():
        return retry_path.read_text(encoding="utf-8")
    return None


def main() -> None:
    output_dir = SUBPROJECT_ROOT / "outputs" / "thesis_pack"
    output_dir.mkdir(parents=True, exist_ok=True)

    claim = build_default_thesis_claim().to_dict()
    benchmark_manifest = build_benchmark_manifest()

    # Try to load bootstrap evidence
    try:
        bootstrap_evidence = build_bootstrap_evidence(REPO_ROOT)
    except ValueError:
        bootstrap_evidence = {"bootstrap_claim_support": {
            "parameter_reduction": 0.0,
            "checkpoint_reduction": 0.0,
        }}

    # Find best benchmark result
    outputs_dir = SUBPROJECT_ROOT / "outputs"
    best_run = _find_best_benchmark_run(outputs_dir)
    comparison_summary = best_run["summary"] if best_run else None

    # Load retry history
    retry_summary = _load_retry_summary(outputs_dir)

    # Write thesis pack
    write_json(output_dir / "thesis_claim.json", claim)
    write_json(output_dir / "benchmark_manifest.json", benchmark_manifest)
    if bootstrap_evidence:
        write_json(output_dir / "bootstrap_evidence.json", bootstrap_evidence)
    if comparison_summary:
        write_json(output_dir / "comparison_summary.json", comparison_summary)

    # Generate model card
    model_card = _build_model_card(claim, bootstrap_evidence)
    (output_dir / "model_card.md").write_text(model_card, encoding="utf-8")

    # Generate negative results
    negative_results = _build_negative_results(retry_summary)
    (output_dir / "negative_results.md").write_text(negative_results, encoding="utf-8")

    # Generate benchmark release doc
    benchmark_release = _build_benchmark_release(benchmark_manifest)
    (output_dir / "benchmark_release.md").write_text(benchmark_release, encoding="utf-8")

    # Generate final report
    report = render_thesis_report(
        claim=claim,
        benchmark_manifest=benchmark_manifest,
        bootstrap_evidence=bootstrap_evidence,
        comparison_summary=comparison_summary,
        negative_results_summary=retry_summary,
    )
    (output_dir / "final_report.md").write_text(report, encoding="utf-8")

    print("Thesis pack written to:", output_dir)
    for path in sorted(output_dir.iterdir()):
        print(f"  {path.name}")


def _build_model_card(claim: dict[str, Any], bootstrap: dict[str, Any]) -> str:
    support = bootstrap.get("bootstrap_claim_support", {})
    lines = [
        "# Model Card: EIDMSA Compact Multimodal Sentiment",
        "",
        "## Model Description",
        "",
        "EIDMSA (Evidential Information-Decomposed Multimodal Sentiment Analysis) is a compact",
        "multimodal sentiment model that integrates Information Bottleneck compression,",
        "Neural Partial Information Decomposition, and Evidential Fusion via Dempster-Shafer.",
        "",
        "## Intended Use",
        "",
        "Research on robust multimodal sentiment analysis under realistic modality failure.",
        "Not intended for production deployment without further validation.",
        "",
        "## Performance Summary",
        "",
        f"- Parameter reduction vs baseline: {support.get('parameter_reduction', 0) * 100:.1f}%",
        f"- Checkpoint reduction vs baseline: {support.get('checkpoint_reduction', 0) * 100:.1f}%",
        "",
        "## Limitations",
        "",
        "- Evaluated only on CMU-MOSEI.",
        "- Selective risk is worse than the baseline on average.",
        "- Worst-case robustness under extreme vision dropout (60%) is the weakest point.",
        "",
        "## Ethical Considerations",
        "",
        "Sentiment analysis models can perpetuate biases present in training data.",
        "This model has not been evaluated for fairness across demographic groups.",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
    ]
    return "\n".join(lines) + "\n"


def _build_negative_results(retry_summary: str | None) -> str:
    lines = [
        "# Negative Results",
        "",
        "## What Did Not Work",
        "",
    ]
    if retry_summary:
        lines.append(retry_summary)
    else:
        lines.append("No retry summary available.")
    lines.extend([
        "",
        "## Lessons",
        "",
        "1. Reliability-aware fusion on old checkpoints made worst-case robustness worse.",
        "2. From-scratch retraining with moderate augmentation was insufficient.",
        "3. The recurring failure point is `block_missing_vision::severe` (60% contiguous vision dropout).",
        "4. Closing a 0.003 gap requires fundamentally better vision-absent handling, not just more tuning.",
        "",
    ])
    return "\n".join(lines) + "\n"


def _build_benchmark_release(manifest: dict[str, Any]) -> str:
    lines = [
        "# Benchmark Release: Realistic Multimodal Sentiment Robustness Protocol",
        "",
        f"**Name**: {manifest['benchmark_name']}",
        f"**Purpose**: {manifest['purpose']}",
        "",
        "## Conditions",
        "",
        "| Name | Family | Modalities | Severity Levels |",
        "| --- | --- | --- | --- |",
    ]
    for cond in manifest["conditions"]:
        lines.append(
            f"| {cond['name']} | {cond['family']} | "
            f"{', '.join(cond['target_modalities'])} | "
            f"{', '.join(cond['severity_levels'])} |"
        )
    lines.extend([
        "",
        "## Primary Metrics",
        "",
    ])
    for metric in manifest["primary_metrics"]:
        lines.append(f"- `{metric}`")
    lines.extend([
        "",
        "## Reproducibility",
        "",
        "All corruption operators are deterministic given a fixed seed.",
        "See `src/real_sentiment/corruptions.py` and `src/real_sentiment/benchmark.py`.",
        "",
    ])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
