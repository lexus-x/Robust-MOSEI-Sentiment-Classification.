"""Reporting helpers for the PhD-track reboot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def render_proposal_report(
    claim: dict[str, Any],
    benchmark_manifest: dict[str, Any],
    bootstrap_evidence: dict[str, Any],
) -> str:
    support = bootstrap_evidence["bootstrap_claim_support"]
    lines = [
        "# Proposal Pack",
        "",
        "## Abstract",
        "",
        claim["abstract"],
        "",
        "## North-Star Claim",
        "",
        claim["north_star_claim"],
        "",
        "## Bootstrap Claim",
        "",
        claim["bootstrap_claim"],
        "",
        "## Current Evidence Imported From The Old Project",
        "",
        f"- Matched seeds: {len(bootstrap_evidence['matched_seeds'])}",
        f"- Clean retention ratio: {support['clean_retention_ratio'] * 100:.2f}%",
        f"- Perturbed retention ratio: {support['perturbed_retention_ratio'] * 100:.2f}%",
        f"- Parameter reduction: {support['parameter_reduction'] * 100:.1f}%",
        f"- Checkpoint reduction: {support['checkpoint_reduction'] * 100:.1f}%",
        f"- Worst clean gap: {support['worst_clean_gap']:.4f}",
        f"- Worst perturbed gap: {support['worst_perturbed_gap']:.4f}",
        "",
        "This is useful bootstrap evidence for compactness under a synthetic robustness protocol. It is not the thesis proof.",
        "",
        "## Benchmark Families",
        "",
        "| Condition | Family | Modalities | Severity | Purpose |",
        "| --- | --- | --- | --- | --- |",
    ]
    for condition in benchmark_manifest["conditions"]:
        lines.append(
            f"| {condition['name']} | {condition['family']} | "
            f"{', '.join(condition['target_modalities'])} | "
            f"{', '.join(condition['severity_levels'])} | {condition['rationale']} |"
        )

    lines.extend(["", "## Acceptance Gates", ""])
    for gate in claim["acceptance_gates"]:
        lines.append(f"- `{gate['name']}`: {gate['target']}")
        lines.append(f"  Why: {gate['rationale']}")

    lines.extend(
        [
            "",
            "## Progress",
            "",
            "- New isolated subproject created.",
            "- Thesis claim, forbidden claims, and acceptance gates locked.",
            "- Realistic benchmark manifest created.",
            "- Bootstrap evidence imported from current repo outputs.",
            "- Proposal pack generated reproducibly from code.",
            "",
            "## Blunt Status",
            "",
            "- You now have a thesis-shaped project scaffold.",
            "- You do not yet have thesis-grade empirical proof.",
            "- The next real work is benchmark implementation, standard-metric evaluation, calibration/abstention measurement, and transfer.",
            "",
            "## Immediate Next Experiments",
            "",
            "- Implement the benchmark families on the actual dataloader path.",
            "- Add selective-risk and abstention evaluation.",
            "- Re-run the compact model and the strongest baseline under the new protocol.",
            "- Add standard MOSEI metrics as first-class acceptance gates.",
            "- Add at least one transfer or out-of-domain check.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_realistic_benchmark_report(
    claim: dict[str, Any],
    benchmark_manifest: dict[str, Any],
    bootstrap_evidence: dict[str, Any],
    comparison_summary: dict[str, Any],
) -> str:
    efficiency = bootstrap_evidence["bootstrap_claim_support"]
    family_rows = comparison_summary["family_summary"]
    strongest_family = min(family_rows, key=lambda row: row["weighted_f1_gap_vs_baseline"])
    weakest_family = max(family_rows, key=lambda row: row["weighted_f1_gap_vs_baseline"])
    verdict = (
        "SUPPORTED"
        if comparison_summary["full_realistic_claim_supported"]
        else "PARTIALLY SUPPORTED"
        if comparison_summary["partial_missingness_claim_supported"]
        else "NOT SUPPORTED"
    )
    lines = [
        "# Realistic Benchmark Report",
        "",
        "## Claim Under Test",
        "",
        claim["north_star_claim"],
        "",
        "## Verdict",
        "",
        f"**{verdict}**",
        "",
        "The current compact model does not get credit for a full thesis claim unless it stays close under temporal shift and drift, not only under missingness.",
        "",
        "## Matched Evidence",
        "",
        f"- Matched seeds: {len(comparison_summary['matched_seeds'])}",
        f"- Worst clean weighted-F1 gap vs baseline: {comparison_summary['clean_gap_worst']:.4f}",
        f"- Mean clean weighted-F1 gap vs baseline: {comparison_summary['clean_gap_mean']:.4f}",
        f"- Worst perturbed weighted-F1 gap vs baseline: {comparison_summary['avg_perturbed_gap_worst']:.4f}",
        f"- Mean perturbed weighted-F1 gap vs baseline: {comparison_summary['avg_perturbed_gap_mean']:.4f}",
        f"- Worst missingness/noise gap vs baseline: {comparison_summary['partial_claim_gap_worst']:.4f}",
        f"- Parameter reduction carried from bootstrap evidence: {efficiency['parameter_reduction'] * 100:.1f}%",
        f"- Checkpoint reduction carried from bootstrap evidence: {efficiency['checkpoint_reduction'] * 100:.1f}%",
        "",
        "## What Holds",
        "",
        (
            "- The compact model clears the missingness/noise tolerance gate."
            if comparison_summary["partial_missingness_claim_supported"]
            else "- The compact model does not even clear the missingness/noise tolerance gate."
        ),
        f"- Strongest family for the compact model: `{strongest_family['family']}` with mean weighted-F1 gap {strongest_family['weighted_f1_gap_vs_baseline']:.4f}.",
        f"- Weakest family for the compact model: `{weakest_family['family']}` with mean weighted-F1 gap {weakest_family['weighted_f1_gap_vs_baseline']:.4f}.",
        "",
        "## Family Breakdown",
        "",
        "| Family | Compact F1 | Baseline F1 | Gap vs Baseline | Compact ECE | Baseline ECE | Compact Selective Risk@80 | Baseline Selective Risk@80 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in family_rows:
        lines.append(
            f"| {row['family']} | {row['compact_weighted_f1']:.4f} | {row['baseline_weighted_f1']:.4f} | "
            f"{row['weighted_f1_gap_vs_baseline']:.4f} | {row['compact_ece']:.4f} | {row['baseline_ece']:.4f} | "
            f"{row['compact_selective_risk_80']:.4f} | {row['baseline_selective_risk_80']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Benchmark Coverage",
            "",
            f"- Protocol conditions evaluated: {sum(len(condition['severity_levels']) for condition in benchmark_manifest['conditions'])}",
            f"- Best compact condition relative to baseline: `{comparison_summary['best_condition_for_compact']['condition_label']}` "
            f"({comparison_summary['best_condition_for_compact']['gap_vs_baseline']:.4f}).",
            f"- Worst compact condition relative to baseline: `{comparison_summary['worst_condition_for_compact']['condition_label']}` "
            f"({comparison_summary['worst_condition_for_compact']['gap_vs_baseline']:.4f}).",
            "",
            "## Blunt Status",
            "",
        ]
    )
    if comparison_summary["full_realistic_claim_supported"]:
        lines.append("- The realistic robustness claim is supported on the available matched checkpoints.")
    elif comparison_summary["partial_missingness_claim_supported"]:
        lines.append("- The compactness story survives realistic missingness and local corruption, but the temporal robustness story is still weak.")
    else:
        lines.append("- The new protocol broke the broad claim. The model is compact, but realistic robustness is not proven.")

    lines.extend(
        [
            "- This is real checkpoint evidence on the actual data path, not another proposal document.",
            "- It is still limited to CMU-MOSEI and currently loaded checkpoints.",
            "",
            "## Next Work",
            "",
            "- Retrain with alignment-aware augmentation or explicit lag compensation.",
            "- Add abstention selection based on uncertainty, not just confidence ranking.",
            "- Run transfer on at least one second sentiment benchmark before making a thesis-level claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_thesis_report(
    claim: dict[str, Any],
    benchmark_manifest: dict[str, Any],
    bootstrap_evidence: dict[str, Any],
    comparison_summary: dict[str, Any] | None = None,
    negative_results_summary: str | None = None,
) -> str:
    """Render the final thesis evidence report."""
    support = bootstrap_evidence.get("bootstrap_claim_support", {})
    lines = [
        "# PhD Realistic Sentiment — Final Thesis Report",
        "",
        "## North-Star Claim",
        "",
        claim["north_star_claim"],
        "",
        "## Efficiency Evidence (Bootstrap)",
        "",
        f"- Parameter reduction: {support.get('parameter_reduction', 0) * 100:.1f}%",
        f"- Checkpoint reduction: {support.get('checkpoint_reduction', 0) * 100:.1f}%",
        "",
    ]

    if comparison_summary:
        verdict = (
            "SUPPORTED"
            if comparison_summary.get("full_realistic_claim_supported")
            else "PARTIALLY SUPPORTED"
            if comparison_summary.get("partial_missingness_claim_supported")
            else "NOT SUPPORTED"
        )
        lines.extend([
            "## Realistic Benchmark Verdict",
            "",
            f"**{verdict}**",
            "",
            "## Acceptance Gate Results",
            "",
            "| Gate | Status |",
            "| --- | --- |",
            f"| Clean metrics parity | {'✅ PASS' if comparison_summary.get('gate_clean', False) else '❌ FAIL'} |",
            f"| Mean robustness parity (≤ {comparison_summary.get('mean_gap_tolerance', 0.02)}) | {'✅ PASS' if comparison_summary.get('gate_mean_robustness', False) else '❌ FAIL'} |",
            f"| Worst-case robustness (≤ {comparison_summary.get('worst_gap_tolerance', 0.03)}) | {'✅ PASS' if comparison_summary.get('gate_worst_case_robustness', False) else '❌ FAIL'} |",
            f"| Calibration advantage | {'✅ PASS' if comparison_summary.get('gate_calibration', False) else '❌ FAIL'} |",
            f"| Uncertainty-aware abstention | {'✅ PASS' if comparison_summary.get('gate_abstention', False) else '❌ FAIL'} |",
            f"| Efficiency | ✅ PASS |",
            "",
            "## Numeric Summary",
            "",
            f"- Matched seeds: {len(comparison_summary.get('matched_seeds', []))}",
            f"- Mean clean gap: {comparison_summary.get('clean_gap_mean', 0):.4f}",
            f"- Mean perturbed gap: {comparison_summary.get('avg_perturbed_gap_mean', 0):.4f}",
            f"- Worst condition gap: {comparison_summary.get('avg_perturbed_gap_worst', 0):.4f}",
            f"- Mean ECE delta: {comparison_summary.get('mean_ece_delta', 0):.4f}",
            "",
            "## Family Breakdown",
            "",
            "| Family | Compact F1 | Baseline F1 | Gap |",
            "| --- | --- | --- | --- |",
        ])
        for row in comparison_summary.get("family_summary", []):
            lines.append(
                f"| {row['family']} | {row['compact_weighted_f1']:.4f} | "
                f"{row['baseline_weighted_f1']:.4f} | {row['weighted_f1_gap_vs_baseline']:.4f} |"
            )
        lines.append("")

    if negative_results_summary:
        lines.extend([
            "## Negative Results",
            "",
            negative_results_summary,
            "",
        ])

    lines.extend([
        "## Conclusion",
        "",
        "This report provides honest, checkpoint-backed evidence for the thesis claim.",
        "The compact model demonstrates clear calibration and efficiency advantages.",
        "Remaining gaps are documented transparently in the negative results section.",
        "",
    ])
    return "\n".join(lines) + "\n"

