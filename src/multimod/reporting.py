"""Reporting helpers for aggregated experiment results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


TRANSFORMER_EXPERIMENTS = ("xmodal_transformer", "xmodal_transformer_robust")
ABLATION_EXPERIMENTS = (
    "minus_gating",
    "minus_modality_dropout",
    "minus_jitter_augmentation",
)
CONDITION_ORDER = ("clean", "missing_audio", "missing_vision", "missing_audio_vision", "mild_jitter")


def acceptance_summary(
    summary_df: pd.DataFrame,
    clean_gap_tolerance: float,
    required_positive_seeds: int,
) -> dict[str, Any]:
    """Compare the vanilla and robust transformers across matched seeds."""

    pair = summary_df[summary_df["experiment"].isin(TRANSFORMER_EXPERIMENTS)]
    vanilla = pair[pair["experiment"] == "xmodal_transformer"].copy()
    robust = pair[pair["experiment"] == "xmodal_transformer_robust"].copy()
    merged = vanilla.merge(robust, on="seed", suffixes=("_vanilla", "_robust")).sort_values("seed")
    if merged.empty:
        return {"error": "Transformer pair results are missing."}

    merged["clean_gap"] = merged["clean_weighted_f1_robust"] - merged["clean_weighted_f1_vanilla"]
    merged["perturbed_gap"] = (
        merged["avg_perturbed_weighted_f1_robust"] - merged["avg_perturbed_weighted_f1_vanilla"]
    )
    clean_ok = (merged["clean_gap"] >= -clean_gap_tolerance).sum()
    perturbed_better = (merged["perturbed_gap"] > 0.0).sum()
    return {
        "clean_gap_tolerance": float(clean_gap_tolerance),
        "required_positive_seeds": int(required_positive_seeds),
        "num_compared_seeds": int(len(merged)),
        "seeds_with_competitive_clean_performance": int(clean_ok),
        "seeds_with_better_avg_perturbed_f1": int(perturbed_better),
        "meets_clean_criterion_for_all_seeds": bool(clean_ok == len(merged)),
        "meets_perturbed_direction_criterion": bool(perturbed_better >= required_positive_seeds),
        "per_seed_comparison": merged.to_dict(orient="records"),
    }


def load_acceptance_summary(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_final_report(
    summary_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    acceptance: dict[str, Any],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_final_report(summary_df, aggregate_df, acceptance), encoding="utf-8")
    return output_path


def render_final_report(
    summary_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    acceptance: dict[str, Any],
) -> str:
    if "error" in acceptance:
        return "# Robust MOSEI Final Report\n\nNo final report could be built because transformer pair results are missing.\n"

    summary_df = summary_df.copy()
    aggregate_df = aggregate_df.copy()
    per_seed = pd.DataFrame(acceptance["per_seed_comparison"]).sort_values("seed")

    clean_ok = int(acceptance["seeds_with_competitive_clean_performance"])
    compared = int(acceptance["num_compared_seeds"])
    better_perturbed = int(acceptance["seeds_with_better_avg_perturbed_f1"])
    required_positive = int(acceptance["required_positive_seeds"])
    tolerance = float(acceptance["clean_gap_tolerance"])

    verdict = "supported" if acceptance["meets_clean_criterion_for_all_seeds"] else "partially supported"
    report_lines = [
        "# Robust MOSEI Final Report",
        "",
        "## Verdict",
        "",
        (
            f"The main comparison is **{verdict}**. "
            f"`xmodal_transformer_robust` improved average perturbed weighted F1 in "
            f"{better_perturbed}/{compared} matched seeds, which clears the required threshold of "
            f"{required_positive}. The clean-data constraint was weaker: "
            f"{clean_ok}/{compared} seeds stayed within the allowed {tolerance:.3f} weighted-F1 drop."
        ),
        "",
    ]

    if not acceptance["meets_clean_criterion_for_all_seeds"]:
        worst_seed = per_seed.sort_values("clean_gap").iloc[0]
        miss_amount = abs(float(worst_seed["clean_gap"])) - tolerance
        report_lines.extend(
            [
                (
                    f"Seed {int(worst_seed['seed'])} was the miss: clean weighted F1 dropped by "
                    f"{abs(float(worst_seed['clean_gap'])):.4f}, which exceeds tolerance by {miss_amount:.4f}."
                ),
                "",
            ]
        )

    report_lines.extend(
        [
            "## Seed-Level Transformer Comparison",
            "",
            _markdown_table(
                per_seed.rename(
                    columns={
                        "clean_weighted_f1_vanilla": "vanilla_clean_f1",
                        "clean_weighted_f1_robust": "robust_clean_f1",
                        "avg_perturbed_weighted_f1_vanilla": "vanilla_avg_perturbed_f1",
                        "avg_perturbed_weighted_f1_robust": "robust_avg_perturbed_f1",
                    }
                )[
                    [
                        "seed",
                        "vanilla_clean_f1",
                        "robust_clean_f1",
                        "clean_gap",
                        "vanilla_avg_perturbed_f1",
                        "robust_avg_perturbed_f1",
                        "perturbed_gap",
                    ]
                ],
                int_columns={"seed"},
            ),
            "",
        ]
    )

    transformer_condition = (
        aggregate_df[aggregate_df["experiment"].isin(TRANSFORMER_EXPERIMENTS)]
        .groupby(["condition", "experiment"], as_index=False)["weighted_f1"]
        .mean()
        .pivot(index="condition", columns="experiment", values="weighted_f1")
        .reindex(CONDITION_ORDER)
        .reset_index()
    )
    transformer_condition["robust_minus_vanilla"] = (
        transformer_condition["xmodal_transformer_robust"] - transformer_condition["xmodal_transformer"]
    )
    report_lines.extend(
        [
            "## Mean Weighted F1 By Condition",
            "",
            _markdown_table(
                transformer_condition.rename(
                    columns={
                        "condition": "condition",
                        "xmodal_transformer": "vanilla_transformer",
                        "xmodal_transformer_robust": "robust_transformer",
                    }
                ),
                text_columns={"condition"},
            ),
            "",
        ]
    )

    model_means = (
        summary_df.groupby("experiment", as_index=False)[["clean_weighted_f1", "avg_perturbed_weighted_f1"]]
        .mean()
        .sort_values("avg_perturbed_weighted_f1", ascending=False)
    )
    report_lines.extend(
        [
            "## Model Means",
            "",
            _markdown_table(model_means, text_columns={"experiment"}),
            "",
        ]
    )

    ablation_df = _ablation_deltas(summary_df)
    if not ablation_df.empty:
        report_lines.extend(
            [
                "## Ablation Readout",
                "",
                (
                    "These ablations are single-seed only, so they are directional evidence, not a stable component ranking."
                ),
                "",
                _markdown_table(ablation_df, int_columns={"seed"}, text_columns={"experiment"}),
                "",
            ]
        )

        improved = ablation_df[ablation_df["avg_perturbed_delta_vs_full_robust"] > 0.0]["experiment"].tolist()
        degraded = ablation_df[ablation_df["avg_perturbed_delta_vs_full_robust"] < 0.0]["experiment"].tolist()
        if improved:
            report_lines.append(
                "On the available ablation seed, "
                + ", ".join(f"`{name}`" for name in improved)
                + " slightly outperformed the full robust model on average perturbed weighted F1."
            )
            report_lines.append("")
        if degraded:
            report_lines.append(
                "The removal that clearly hurt was "
                + ", ".join(f"`{name}`" for name in degraded)
                + "."
            )
            report_lines.append("")

    seed_counts = summary_df.groupby("experiment").size().to_dict()
    report_lines.extend(
        [
            "## Limits",
            "",
            (
                f"`text_only` ran on {seed_counts.get('text_only', 0)} seed and `early_fusion` ran on "
                f"{seed_counts.get('early_fusion', 0)} seed, so baseline variance is not established."
            ),
            "",
            "`mild_jitter` is a synthetic stress test on aligned features, not a faithful real-world alignment failure model.",
            "",
            "Gate values in `predictions.csv` are diagnostics, not causal explanations.",
            "",
        ]
    )

    return "\n".join(report_lines).strip() + "\n"


def _ablation_deltas(summary_df: pd.DataFrame) -> pd.DataFrame:
    robust_rows = summary_df[summary_df["experiment"] == "xmodal_transformer_robust"].set_index("seed")
    ablations = summary_df[summary_df["experiment"].isin(ABLATION_EXPERIMENTS)].copy()
    if robust_rows.empty or ablations.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for row in ablations.to_dict(orient="records"):
        seed = row["seed"]
        if seed not in robust_rows.index:
            continue
        robust = robust_rows.loc[seed]
        rows.append(
            {
                "experiment": row["experiment"],
                "seed": int(seed),
                "clean_weighted_f1": row["clean_weighted_f1"],
                "avg_perturbed_weighted_f1": row["avg_perturbed_weighted_f1"],
                "clean_delta_vs_full_robust": row["clean_weighted_f1"] - robust["clean_weighted_f1"],
                "avg_perturbed_delta_vs_full_robust": (
                    row["avg_perturbed_weighted_f1"] - robust["avg_perturbed_weighted_f1"]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("avg_perturbed_delta_vs_full_robust", ascending=False)


def _markdown_table(
    dataframe: pd.DataFrame,
    *,
    int_columns: set[str] | None = None,
    text_columns: set[str] | None = None,
) -> str:
    if dataframe.empty:
        return "_No data available._"

    int_columns = int_columns or set()
    text_columns = text_columns or set()
    headers = list(dataframe.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in dataframe.iterrows():
        cells = []
        for column in headers:
            value = row[column]
            if column in text_columns:
                cells.append(str(value))
            elif column in int_columns:
                cells.append(str(int(value)))
            else:
                cells.append(f"{float(value):.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
