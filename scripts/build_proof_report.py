#!/usr/bin/env python
"""Build a local proof pack for CMU-MOSEI model claims.

This script gathers finished local runs, writes comparison CSVs, and emits a
markdown report that separates defensible claims from overclaims.
"""

from __future__ import annotations

import csv
import json
import math
import statistics as stats
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "proof_pack"


@dataclass
class ExperimentSource:
    label: str
    experiment: str
    roots: tuple[str, ...]


LOCAL_SOURCES = (
    ExperimentSource(
        label="Vanilla Transformer",
        experiment="xmodal_transformer",
        roots=("outputs/main_run",),
    ),
    ExperimentSource(
        label="Robust Transformer",
        experiment="xmodal_transformer_robust",
        roots=("outputs/main_run",),
    ),
    ExperimentSource(
        label="EIDMSA Base",
        experiment="eidmsa",
        roots=("outputs/eidmsa_gpu_final",),
    ),
    ExperimentSource(
        label="EIDMSA + KAN",
        experiment="eidmsa_kan",
        roots=("outputs/eidmsa_gpu_final", "outputs/eidmsa_gpu_fast"),
    ),
    ExperimentSource(
        label="EIDMSA + Mamba",
        experiment="eidmsa_mamba",
        roots=("outputs/eidmsa_gpu_final", "outputs/eidmsa_gpu_fast"),
    ),
    ExperimentSource(
        label="EIDMSA + KAN + Mamba",
        experiment="eidmsa_kan_mamba",
        roots=("outputs/eidmsa_gpu_final", "outputs/eidmsa_gpu_fast"),
    ),
)


LITERATURE_ROWS = (
    {
        "source": "MPFN (CCL 2020)",
        "task_setup": "Direct 3-class sentiment classification on CMU-MOSEI",
        "metrics": "Acc=61.30, F1=59.67",
        "notes": "Older directly comparable 3-class setup.",
        "citation": "https://aclanthology.org/2020.ccl-1.101.pdf",
    },
    {
        "source": "UniMSE (EMNLP 2022)",
        "task_setup": "Standard MOSEI metrics: Acc-2 / F1 / Acc-7 / MAE / Corr",
        "metrics": "Acc-2=87.50, F1=87.46, Acc-7=54.39, MAE=0.523, Corr=0.773",
        "notes": "Current strong published baseline, but not directly comparable to this repo's 3-class weighted-F1 protocol.",
        "citation": "https://arxiv.org/abs/2211.11256",
    },
    {
        "source": "MissModal (TACL 2023)",
        "task_setup": "Standard MOSEI metrics: Acc-2 / F1 / Acc-7 / MAE / Corr",
        "metrics": "Acc-2=85.9, F1=85.8, Acc-7=53.9, MAE=0.533, Corr=0.769",
        "notes": "Published robustness-oriented multimodal baseline on standard MOSEI metrics.",
        "citation": "https://aclanthology.org/anthology-files/pdf/tacl/2023.tacl-1.94.pdf",
    },
    {
        "source": "MMML + context (NAACL 2024)",
        "task_setup": "Standard MOSEI metrics: Acc2Has0 / F1Has0 / Acc2Non0 / F1Non0 / Acc-7 / MAE / Corr",
        "metrics": "87.24, 87.18, 88.02, 88.15, 55.74, 0.492, 0.814",
        "notes": "Paper claims state of the art on standard MOSEI metrics, not 3-class weighted F1.",
        "citation": "https://aclanthology.org/2024.naacl-long.197.pdf",
    },
    {
        "source": "FeaDA (IJCNLP 2025)",
        "task_setup": "Standard MOSEI metrics: Acc-2 / F1 / Acc-7 / MAE / Corr",
        "metrics": "84.25/85.47, 84.22/85.16, 53.49, 0.548, 0.771",
        "notes": "Recent primary-source comparison focused on standard MOSEI metrics.",
        "citation": "https://aclanthology.org/2025.ijcnlp-long.6.pdf",
    },
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _infer_evidence_level(root: str, num_seeds: int) -> str:
    root_name = Path(root).name
    if "fast" in root_name and num_seeds == 1:
        return "fast_1_seed"
    return f"completed_{num_seeds}_seed"


def _load_rows_from_root(source: ExperimentSource, root: str) -> list[dict[str, Any]]:
    exp_root = PROJECT_ROOT / root / source.experiment
    rows: list[dict[str, Any]] = []
    if not exp_root.exists():
        return rows

    for seed_dir in sorted([path for path in exp_root.iterdir() if path.is_dir()]):
        metrics_path = seed_dir / "metrics.json"
        checkpoint_path = seed_dir / "best_model.pt"
        if not metrics_path.exists():
            continue
        payload = _read_json(metrics_path)
        conditions = {row["condition"]: row for row in payload.get("conditions", [])}
        clean = conditions.get("clean", {})
        rows.append(
            {
                "label": source.label,
                "experiment": source.experiment,
                "seed": seed_dir.name,
                "clean_weighted_f1": payload["summary"].get("clean_weighted_f1"),
                "avg_perturbed_weighted_f1": payload["summary"].get("avg_perturbed_weighted_f1"),
                "clean_accuracy": clean.get("accuracy"),
                "clean_ece": payload["summary"].get("clean_ece"),
                "clean_uncertainty": payload["summary"].get("clean_uncertainty"),
                "num_parameters": payload.get("run", {}).get("num_parameters"),
                "source_root": root,
                "checkpoint_mb": (
                    checkpoint_path.stat().st_size / (1024 * 1024)
                    if checkpoint_path.exists()
                    else None
                ),
                "checkpoint_path": str(checkpoint_path) if checkpoint_path.exists() else "",
                "metrics_path": str(metrics_path),
            }
        )
    evidence_level = _infer_evidence_level(root, len(rows))
    for row in rows:
        row["evidence_level"] = evidence_level
    return rows


def load_local_rows(source: ExperimentSource) -> list[dict[str, Any]]:
    best_rows: list[dict[str, Any]] = []
    for root in source.roots:
        rows = _load_rows_from_root(source, root)
        if len(rows) > len(best_rows):
            best_rows = rows
    return best_rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "label": rows[0]["label"],
        "experiment": rows[0]["experiment"],
        "evidence_level": rows[0]["evidence_level"],
        "num_seeds": len(rows),
    }
    for key in (
        "clean_weighted_f1",
        "avg_perturbed_weighted_f1",
        "clean_accuracy",
        "clean_ece",
        "clean_uncertainty",
        "num_parameters",
        "checkpoint_mb",
    ):
        values = [row[key] for row in rows if row[key] is not None]
        if not values:
            summary[f"{key}_mean"] = None
            summary[f"{key}_std"] = None
            continue
        summary[f"{key}_mean"] = sum(values) / len(values)
        summary[f"{key}_std"] = stats.stdev(values) if len(values) > 1 else 0.0
    return summary


def format_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return f"{value:.{digits}f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_markdown(
    seed_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> str:
    by_label = {row["label"]: row for row in summary_rows}

    robust = by_label["Robust Transformer"]
    eidmsa = by_label["EIDMSA Base"]
    kan = by_label["EIDMSA + KAN"]

    param_reduction = 1.0 - (
        eidmsa["num_parameters_mean"] / robust["num_parameters_mean"]
    )
    ckpt_reduction = 1.0 - (
        eidmsa["checkpoint_mb_mean"] / robust["checkpoint_mb_mean"]
    )
    kan_clean_delta = kan["clean_weighted_f1_mean"] - robust["clean_weighted_f1_mean"]
    kan_pert_delta = (
        kan["avg_perturbed_weighted_f1_mean"] - robust["avg_perturbed_weighted_f1_mean"]
    )
    kan_acc_delta = kan["clean_accuracy_mean"] - robust["clean_accuracy_mean"]

    best_eidmsa = max(
        [row for row in seed_rows if row["label"] == "EIDMSA Base"],
        key=lambda row: row["clean_weighted_f1"] or float("-inf"),
    )
    best_robust = max(
        [row for row in seed_rows if row["label"] == "Robust Transformer"],
        key=lambda row: row["clean_weighted_f1"] or float("-inf"),
    )
    mpfn_acc = 0.6130
    mpfn_f1 = 0.5967

    lines: list[str] = []
    lines.append("# CMU-MOSEI Proof Pack")
    lines.append("")
    lines.append("## Bottom Line")
    lines.append("")
    lines.append(
        "Current evidence does **not** support the claim that this repo beats the current published CMU-MOSEI state of the art."
    )
    lines.append(
        "What the evidence *does* support is a narrower claim: the project produces real trained checkpoints, the base EIDMSA model is much smaller than the transformer baselines, and the KAN variant shows a small clean-data gain without a robustness win."
    )
    lines.append("")
    lines.append("## What Is Proven Locally")
    lines.append("")
    lines.append(
        f"- Real checkpoints exist. Example EIDMSA checkpoint: `{best_eidmsa['checkpoint_path']}` "
        f"({format_float(best_eidmsa['checkpoint_mb'], 3)} MB)."
    )
    lines.append(
        f"- Base EIDMSA averages {format_float(eidmsa['num_parameters_mean'], 0)} parameters versus "
        f"{format_float(robust['num_parameters_mean'], 0)} for the robust transformer."
    )
    lines.append(
        f"- That is a parameter reduction of {format_float(param_reduction * 100, 1)}% and an on-disk checkpoint reduction of "
        f"{format_float(ckpt_reduction * 100, 1)}%."
    )
    lines.append(
        f"- The best completed base EIDMSA seed reaches clean weighted F1 {format_float(best_eidmsa['clean_weighted_f1'])} "
        f"and clean accuracy {format_float(best_eidmsa['clean_accuracy'])}."
    )
    lines.append(
        f"- Against the older directly comparable 3-class MPFN paper (Acc=0.6130, F1=0.5967), the best completed base EIDMSA seed is "
        f"+{format_float(best_eidmsa['clean_accuracy'] - mpfn_acc)} in accuracy and "
        f"+{format_float(best_eidmsa['clean_weighted_f1'] - mpfn_f1)} in F1."
    )
    lines.append(
        f"- The best completed robust transformer seed is +{format_float(best_robust['clean_accuracy'] - mpfn_acc)} in accuracy and "
        f"+{format_float(best_robust['clean_weighted_f1'] - mpfn_f1)} in F1 over that same MPFN reference."
    )
    lines.append("")
    lines.append("## Local Comparison")
    lines.append("")
    lines.append("| Model | Evidence | Seeds | Clean F1 | Perturbed F1 | Clean Acc | ECE | Params | Checkpoint MB |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in summary_rows:
        lines.append(
            f"| {row['label']} | {row['evidence_level']} | {row['num_seeds']} | "
            f"{format_float(row['clean_weighted_f1_mean'])} | "
            f"{format_float(row['avg_perturbed_weighted_f1_mean'])} | "
            f"{format_float(row['clean_accuracy_mean'])} | "
            f"{format_float(row['clean_ece_mean'])} | "
            f"{format_float(row['num_parameters_mean'], 0)} | "
            f"{format_float(row['checkpoint_mb_mean'], 3)} |"
        )
    lines.append("")
    lines.append("### Defensible Internal Claims")
    lines.append("")
    lines.append(
        f"- Base EIDMSA is smaller than the robust transformer baseline by {format_float(param_reduction * 100, 1)}% in parameters and "
        f"{format_float(ckpt_reduction * 100, 1)}% on disk."
    )
    lines.append(
        f"- Across the available {kan['num_seeds']}-seed local evidence, EIDMSA + KAN changes mean clean weighted F1 by {format_float(kan_clean_delta)} "
        f"and mean clean accuracy by {format_float(kan_acc_delta)} versus the robust transformer."
    )
    lines.append(
        f"- The same KAN evidence is **not** a robustness win: mean perturbed weighted F1 changes by {format_float(kan_pert_delta)} "
        f"versus the robust transformer."
    )
    lines.append(
        "- Unless `mamba-ssm` was installed for those runs, the Mamba experiments are fallback-path evidence, not proof that real Mamba helps."
    )
    lines.append("")
    lines.append("### Claims You Cannot Make")
    lines.append("")
    lines.append(
        "- You cannot honestly claim current CMU-MOSEI SOTA from these runs. The completed base EIDMSA 3-seed average is below both local transformer baselines on clean and perturbed F1."
    )
    lines.append(
        "- You cannot honestly claim a robustness improvement over the robust transformer baseline from the currently completed evidence."
    )
    lines.append("")
    lines.append("## Published Comparison")
    lines.append("")
    lines.append(
        "The literature mostly reports CMU-MOSEI with standard metrics such as Acc-2, F1, Acc-7, MAE, and Corr. "
        "This repo currently optimizes and reports a custom 3-class weighted-F1/accuracy setup, so published standard-metric SOTA is not apples-to-apples."
    )
    lines.append("")
    lines.append("| Source | Setup | Reported MOSEI Metrics | Note |")
    lines.append("| --- | --- | --- | --- |")
    for row in LITERATURE_ROWS:
        lines.append(
            f"| {row['source']} | {row['task_setup']} | {row['metrics']} | {row['notes']} |"
        )
    lines.append("")
    lines.append("### Strongest Honest Framing")
    lines.append("")
    lines.append(
        "- If you need a hard proof statement today: **the project trains real models and saves real checkpoints; base EIDMSA is materially more efficient than the transformer baselines; and the KAN variant shows a small clean-data gain on the available local evidence.**"
    )
    lines.append(
        "- If you need a SOTA statement: **you do not have the evidence yet**. Either reproduce standard MOSEI metrics and beat current published numbers, or do not make the claim."
    )
    lines.append("")
    lines.append("## Source Links")
    lines.append("")
    for row in LITERATURE_ROWS:
        lines.append(f"- {row['source']}: {row['citation']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    seed_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for source in LOCAL_SOURCES:
        rows = load_local_rows(source)
        seed_rows.extend(rows)
        if rows:
            summary_rows.append(summarize_rows(rows))

    summary_rows.sort(key=lambda row: (-row["num_seeds"], row["label"]))
    seed_rows.sort(key=lambda row: (row["label"], row["seed"]))

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_ROOT / "local_seed_rows.csv", seed_rows)
    write_csv(OUTPUT_ROOT / "local_summary_rows.csv", summary_rows)
    write_csv(OUTPUT_ROOT / "literature_rows.csv", list(LITERATURE_ROWS))
    (OUTPUT_ROOT / "proof_report.md").write_text(
        build_markdown(seed_rows, summary_rows),
        encoding="utf-8",
    )

    print(f"Wrote {OUTPUT_ROOT / 'local_seed_rows.csv'}")
    print(f"Wrote {OUTPUT_ROOT / 'local_summary_rows.csv'}")
    print(f"Wrote {OUTPUT_ROOT / 'literature_rows.csv'}")
    print(f"Wrote {OUTPUT_ROOT / 'proof_report.md'}")


if __name__ == "__main__":
    main()
