#!/usr/bin/env python
"""Build slide-ready visuals from completed MOSEI experiment outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multimod.utils import ensure_dir


SLIDE_SIZE = (13.333, 7.5)
CONDITION_ORDER = ("clean", "missing_audio", "missing_vision", "missing_audio_vision", "mild_jitter")
CONDITION_LABELS = {
    "clean": "Clean",
    "missing_audio": "Missing audio",
    "missing_vision": "Missing vision",
    "missing_audio_vision": "Missing audio + vision",
    "mild_jitter": "Mild jitter",
}
EXPERIMENT_LABELS = {
    "text_only": "Text only",
    "early_fusion": "Early fusion",
    "xmodal_transformer": "Vanilla transformer",
    "xmodal_transformer_robust": "Robust transformer",
    "minus_gating": "No gating",
    "minus_modality_dropout": "No modality dropout",
    "minus_jitter_augmentation": "No jitter aug",
}
COLORS = {
    "bg": "#F6F1E8",
    "text": "#1F2937",
    "muted": "#6B7280",
    "grid": "#D6D3D1",
    "positive": "#1D6F42",
    "negative": "#B42318",
    "vanilla": "#5B6574",
    "robust": "#0F766E",
    "audio": "#C2410C",
    "vision": "#2563EB",
    "text_modality": "#374151",
    "missing": "#D1D5DB",
    "jitter": "#E9D5B5",
}
EXPERIMENT_COLORS = {
    "text_only": "#3B82F6",
    "early_fusion": "#F59E0B",
    "xmodal_transformer": COLORS["vanilla"],
    "xmodal_transformer_robust": COLORS["robust"],
    "minus_gating": "#0EA5E9",
    "minus_modality_dropout": "#A16207",
    "minus_jitter_augmentation": "#B42318",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="Path to aggregate_results.csv")
    parser.add_argument("--summary", required=True, help="Path to run_summary.csv")
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to robust-model predictions.csv for gate diagnostics",
    )
    parser.add_argument("--output", required=True, help="Directory for generated visuals")
    return parser.parse_args()


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": COLORS["bg"],
            "axes.facecolor": COLORS["bg"],
            "savefig.facecolor": COLORS["bg"],
            "axes.edgecolor": COLORS["muted"],
            "axes.labelcolor": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "text.color": COLORS["text"],
            "font.size": 12,
            "axes.titlesize": 20,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.7,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _experiment_label(experiment: str) -> str:
    return EXPERIMENT_LABELS.get(experiment, experiment.replace("_", " "))


def _experiment_color(experiment: str) -> str:
    return EXPERIMENT_COLORS.get(experiment, COLORS["text_modality"])


def _tradeoff_map_note(summary_df: pd.DataFrame) -> str:
    seed_counts = summary_df.groupby("experiment").size()
    multi_seed = [f"{_experiment_label(exp)} ({count})" for exp, count in seed_counts.items() if count > 1]
    single_seed = sum(int(count == 1) for count in seed_counts.values)
    parts: list[str] = []
    if multi_seed:
        parts.append("Multi-seed: " + ", ".join(multi_seed) + ".")
    if single_seed:
        parts.append(f"Single-seed: {single_seed} other model(s).")
    return " ".join(parts)


def _seed_tradeoff_note(summary_df: pd.DataFrame) -> str:
    target = summary_df[summary_df["experiment"].isin(["xmodal_transformer", "xmodal_transformer_robust"])]
    if target.empty:
        return "No matched transformer seeds in this summary file."
    pivot = target.pivot(index="seed", columns="experiment", values=["clean_weighted_f1", "avg_perturbed_weighted_f1"])
    perturbed_wins = 0
    clean_losses = 0
    for seed in pivot.index:
        perturbed_wins += int(
            pivot.loc[seed, ("avg_perturbed_weighted_f1", "xmodal_transformer_robust")]
            > pivot.loc[seed, ("avg_perturbed_weighted_f1", "xmodal_transformer")]
        )
        clean_losses += int(
            pivot.loc[seed, ("clean_weighted_f1", "xmodal_transformer_robust")]
            < pivot.loc[seed, ("clean_weighted_f1", "xmodal_transformer")]
        )
    return (
        f"Perturbed wins: {perturbed_wins}/{len(pivot)} matched seed(s). "
        f"Clean F1 is lower on {clean_losses}/{len(pivot)} seed(s)."
    )


def _transformer_condition_note(grouped: pd.DataFrame) -> str:
    deltas = grouped["xmodal_transformer_robust"] - grouped["xmodal_transformer"]
    better = [CONDITION_LABELS[name] for name in grouped.index[deltas > 0.0]]
    worse_or_equal = [CONDITION_LABELS[name] for name in grouped.index[deltas <= 0.0]]
    parts: list[str] = []
    if better:
        parts.append("Robust is better on: " + ", ".join(better) + ".")
    if worse_or_equal:
        parts.append("Robust is not better on: " + ", ".join(worse_or_equal) + ".")
    return " ".join(parts)


def _select_ablation_seed(summary_df: pd.DataFrame) -> int:
    required = {"minus_gating", "minus_modality_dropout", "minus_jitter_augmentation"}
    robust_seeds = set(summary_df[summary_df["experiment"] == "xmodal_transformer_robust"]["seed"].tolist())
    counts = (
        summary_df[summary_df["experiment"].isin(required)]
        .groupby("seed")["experiment"]
        .nunique()
    )
    candidate_seeds = [int(seed) for seed, count in counts.items() if count == len(required) and seed in robust_seeds]
    if not candidate_seeds:
        raise ValueError("No seed contains the full ablation set plus the robust reference run.")
    return min(candidate_seeds)


def _visual_index_notes(summary_df: pd.DataFrame, results_df: pd.DataFrame) -> list[str]:
    notes: list[str] = []

    ablation_seed = _select_ablation_seed(summary_df)
    seed_rows = summary_df[summary_df["seed"] == ablation_seed].set_index("experiment")
    robust = seed_rows.loc["xmodal_transformer_robust"]
    for experiment in ("minus_gating", "minus_modality_dropout", "minus_jitter_augmentation"):
        clean_delta = seed_rows.loc[experiment, "clean_weighted_f1"] - robust["clean_weighted_f1"]
        perturbed_delta = (
            seed_rows.loc[experiment, "avg_perturbed_weighted_f1"] - robust["avg_perturbed_weighted_f1"]
        )
        if clean_delta > 0.0 or perturbed_delta > 0.0:
            notes.append(
                f"On ablation seed {ablation_seed}, `{experiment}` beats the full robust model on "
                f"{'clean' if clean_delta > 0.0 and perturbed_delta <= 0.0 else 'perturbed' if perturbed_delta > 0.0 and clean_delta <= 0.0 else 'both clean and perturbed'} metrics."
            )

    transformer = (
        results_df[results_df["experiment"].isin(["xmodal_transformer", "xmodal_transformer_robust"])]
        .groupby(["condition", "experiment"], as_index=False)["weighted_f1"]
        .mean()
        .pivot(index="condition", columns="experiment", values="weighted_f1")
        .reindex(CONDITION_ORDER)
    )
    if "mild_jitter" in transformer.index:
        jitter_delta = transformer.loc["mild_jitter", "xmodal_transformer_robust"] - transformer.loc["mild_jitter", "xmodal_transformer"]
        if jitter_delta <= 0.0:
            notes.append("Do not claim `mild_jitter` as a win; robust is not better there.")
        else:
            notes.append(f"`mild_jitter` is only a small win ({jitter_delta:+.3f}), not the headline result.")

    notes.append("There is no human-readable utterance text in the saved outputs, so true qualitative example slides are not available from this repo alone.")
    return notes


def plot_model_tradeoff_map(summary_df: pd.DataFrame, output_dir: Path) -> Path:
    means = (
        summary_df.groupby("experiment", as_index=False)[["clean_weighted_f1", "avg_perturbed_weighted_f1"]]
        .mean()
    )
    means["label"] = means["experiment"].map(_experiment_label)
    means["color"] = means["experiment"].map(_experiment_color)

    fig, ax = plt.subplots(figsize=SLIDE_SIZE)
    ax.grid(True, axis="both", linestyle="--", linewidth=0.8)
    ax.set_title("Model Trade-Off Map", pad=18)
    ax.set_xlabel("Clean weighted F1")
    ax.set_ylabel("Average perturbed weighted F1")

    offsets = {
        "text_only": (8, -2),
        "early_fusion": (8, -14),
        "xmodal_transformer": (8, -16),
        "xmodal_transformer_robust": (8, 12),
        "minus_gating": (8, 18),
        "minus_modality_dropout": (8, -8),
        "minus_jitter_augmentation": (8, 2),
    }

    for row in means.itertuples(index=False):
        ax.scatter(
            row.clean_weighted_f1,
            row.avg_perturbed_weighted_f1,
            s=210 if row.experiment.startswith("xmodal_transformer") else 150,
            color=row.color,
            edgecolor=COLORS["bg"],
            linewidth=1.4,
            zorder=3,
        )
        dx, dy = offsets.get(row.experiment, (8, 8))
        ax.annotate(
            row.label,
            (row.clean_weighted_f1, row.avg_perturbed_weighted_f1),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=11,
            weight="bold" if row.experiment.startswith("xmodal_transformer") else None,
        )

    ax.text(
        0.02,
        0.03,
        _tradeoff_map_note(summary_df),
        transform=ax.transAxes,
        fontsize=11,
        color=COLORS["muted"],
    )
    ax.set_xlim(0.39, 0.65)
    ax.set_ylim(0.34, 0.64)
    path = output_dir / "slide_05_model_tradeoff_map.png"
    save_figure(fig, path)
    return path


def plot_conditions_diagram(output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 5, figsize=SLIDE_SIZE)
    fig.suptitle("Controlled Robustness Conditions", fontsize=24, fontweight="bold", y=0.98)
    modality_names = ("Text", "Audio", "Vision")
    modality_colors = (COLORS["text_modality"], COLORS["audio"], COLORS["vision"])

    for ax, condition in zip(axes, CONDITION_ORDER):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(CONDITION_LABELS[condition], fontsize=13, pad=10)
        y_positions = (0.72, 0.50, 0.28)

        for y, name, color in zip(y_positions, modality_names, modality_colors):
            x = 0.12
            fc = color
            label = name
            hatch = None
            alpha = 0.95
            edge = color

            if condition == "missing_audio" and name == "Audio":
                fc = COLORS["missing"]
                edge = COLORS["muted"]
                hatch = "///"
                label = "Audio missing"
            elif condition == "missing_vision" and name == "Vision":
                fc = COLORS["missing"]
                edge = COLORS["muted"]
                hatch = "///"
                label = "Vision missing"
            elif condition == "missing_audio_vision" and name in {"Audio", "Vision"}:
                fc = COLORS["missing"]
                edge = COLORS["muted"]
                hatch = "///"
                label = f"{name} missing"
            elif condition == "mild_jitter" and name in {"Audio", "Vision"}:
                ghost = FancyBboxPatch(
                    (x, y),
                    0.76,
                    0.14,
                    boxstyle="round,pad=0.02,rounding_size=0.03",
                    linewidth=1.2,
                    edgecolor=COLORS["muted"],
                    facecolor="none",
                    linestyle="--",
                )
                ax.add_patch(ghost)
                shift = 0.08 if name == "Audio" else -0.06
                x += shift
                arrow = FancyArrowPatch(
                    (0.50, y + 0.07),
                    (0.50 + shift, y + 0.07),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linewidth=1.2,
                    color=COLORS["muted"],
                )
                ax.add_patch(arrow)
                fc = COLORS["jitter"]
                alpha = 1.0
                label = f"{name} shifted"

            box = FancyBboxPatch(
                (x, y),
                0.76,
                0.14,
                boxstyle="round,pad=0.02,rounding_size=0.03",
                linewidth=1.6,
                edgecolor=edge,
                facecolor=fc,
                hatch=hatch,
                alpha=alpha,
            )
            ax.add_patch(box)
            ax.text(
                x + 0.38,
                y + 0.07,
                label,
                ha="center",
                va="center",
                fontsize=11,
                color="white" if fc not in {COLORS["missing"], COLORS["jitter"]} else COLORS["text"],
                fontweight="bold",
            )

    fig.text(
        0.5,
        0.03,
        "Mild jitter is a synthetic alignment stress test, not a claim about real deployment timing failures.",
        ha="center",
        fontsize=11,
        color=COLORS["muted"],
    )
    path = output_dir / "slide_06_conditions_diagram.png"
    save_figure(fig, path)
    return path


def plot_seed_tradeoff(summary_df: pd.DataFrame, output_dir: Path) -> Path:
    target = summary_df[summary_df["experiment"].isin(["xmodal_transformer", "xmodal_transformer_robust"])].copy()
    pivot = target.pivot(index="seed", columns="experiment", values=["clean_weighted_f1", "avg_perturbed_weighted_f1"])

    fig, ax = plt.subplots(figsize=SLIDE_SIZE)
    ax.grid(True, axis="both", linestyle="--", linewidth=0.8)
    ax.set_title("Seed-Level Trade-Off: Clean vs Perturbed", pad=18)
    ax.set_xlabel("Clean weighted F1")
    ax.set_ylabel("Average perturbed weighted F1")

    for seed in pivot.index:
        van_clean = pivot.loc[seed, ("clean_weighted_f1", "xmodal_transformer")]
        van_pert = pivot.loc[seed, ("avg_perturbed_weighted_f1", "xmodal_transformer")]
        rob_clean = pivot.loc[seed, ("clean_weighted_f1", "xmodal_transformer_robust")]
        rob_pert = pivot.loc[seed, ("avg_perturbed_weighted_f1", "xmodal_transformer_robust")]

        ax.annotate(
            "",
            xy=(rob_clean, rob_pert),
            xytext=(van_clean, van_pert),
            arrowprops=dict(arrowstyle="->", color=COLORS["muted"], lw=1.8),
        )
        ax.scatter(van_clean, van_pert, s=180, color=COLORS["vanilla"], edgecolor=COLORS["bg"], linewidth=1.4, zorder=3)
        ax.scatter(rob_clean, rob_pert, s=200, marker="D", color=COLORS["robust"], edgecolor=COLORS["bg"], linewidth=1.4, zorder=4)
        ax.annotate(f"seed {seed}", (rob_clean, rob_pert), xytext=(8, 8), textcoords="offset points", fontsize=11)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["vanilla"], markeredgecolor=COLORS["bg"], markersize=11, label="Vanilla transformer"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=COLORS["robust"], markeredgecolor=COLORS["bg"], markersize=11, label="Robust transformer"),
    ]
    ax.legend(handles=handles, loc="lower right")
    ax.text(
        0.02,
        0.03,
        _seed_tradeoff_note(summary_df),
        transform=ax.transAxes,
        fontsize=11,
        color=COLORS["muted"],
    )
    ax.set_xlim(0.618, 0.6365)
    ax.set_ylim(0.592, 0.6355)
    path = output_dir / "slide_07_seed_tradeoff.png"
    save_figure(fig, path)
    return path


def plot_transformer_condition_comparison(results_df: pd.DataFrame, output_dir: Path) -> Path:
    grouped = (
        results_df[results_df["experiment"].isin(["xmodal_transformer", "xmodal_transformer_robust"])]
        .groupby(["condition", "experiment"], as_index=False)["weighted_f1"]
        .mean()
        .pivot(index="condition", columns="experiment", values="weighted_f1")
        .reindex(CONDITION_ORDER)
    )

    x = np.arange(len(grouped.index))
    width = 0.34

    fig, ax = plt.subplots(figsize=SLIDE_SIZE)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8)
    vanilla = ax.bar(x - width / 2, grouped["xmodal_transformer"], width, color=COLORS["vanilla"], label="Vanilla transformer")
    robust = ax.bar(x + width / 2, grouped["xmodal_transformer_robust"], width, color=COLORS["robust"], label="Robust transformer")

    ax.set_title("Transformer Pair by Evaluation Condition", pad=18)
    ax.set_ylabel("Mean weighted F1")
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[name].replace(" + ", "\n+\n") for name in grouped.index])
    ax.set_ylim(0.56, 0.645)

    for index, condition in enumerate(grouped.index):
        delta = grouped.loc[condition, "xmodal_transformer_robust"] - grouped.loc[condition, "xmodal_transformer"]
        ax.text(
            x[index],
            max(vanilla[index].get_height(), robust[index].get_height()) + 0.003,
            f"{delta:+.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            color=COLORS["positive"] if delta > 0 else COLORS["negative"],
            fontweight="bold",
        )

    ax.text(
        0.02,
        0.03,
        _transformer_condition_note(grouped),
        transform=ax.transAxes,
        fontsize=11,
        color=COLORS["muted"],
    )
    path = output_dir / "slide_08_transformer_condition_comparison.png"
    save_figure(fig, path)
    return path


def plot_ablation_deltas(summary_df: pd.DataFrame, output_dir: Path) -> Path:
    ablation_seed = _select_ablation_seed(summary_df)
    seed_rows = summary_df[summary_df["seed"] == ablation_seed].set_index("experiment")
    robust = seed_rows.loc["xmodal_transformer_robust"]
    ablations = pd.DataFrame(
        [
            {
                "experiment": "minus_gating",
                "clean_delta": seed_rows.loc["minus_gating", "clean_weighted_f1"] - robust["clean_weighted_f1"],
                "perturbed_delta": seed_rows.loc["minus_gating", "avg_perturbed_weighted_f1"] - robust["avg_perturbed_weighted_f1"],
            },
            {
                "experiment": "minus_modality_dropout",
                "clean_delta": seed_rows.loc["minus_modality_dropout", "clean_weighted_f1"] - robust["clean_weighted_f1"],
                "perturbed_delta": seed_rows.loc["minus_modality_dropout", "avg_perturbed_weighted_f1"] - robust["avg_perturbed_weighted_f1"],
            },
            {
                "experiment": "minus_jitter_augmentation",
                "clean_delta": seed_rows.loc["minus_jitter_augmentation", "clean_weighted_f1"] - robust["clean_weighted_f1"],
                "perturbed_delta": seed_rows.loc["minus_jitter_augmentation", "avg_perturbed_weighted_f1"] - robust["avg_perturbed_weighted_f1"],
            },
        ]
    )
    ablations["label"] = ablations["experiment"].map(EXPERIMENT_LABELS)
    y = np.arange(len(ablations))

    fig, axes = plt.subplots(1, 2, figsize=SLIDE_SIZE, sharey=True)
    fig.suptitle("Ablations vs Full Robust Model", fontsize=24, fontweight="bold", y=0.98)
    fig.subplots_adjust(bottom=0.14, top=0.88, wspace=0.20)

    for ax, column, title in zip(
        axes,
        ["clean_delta", "perturbed_delta"],
        ["Clean weighted F1 delta", "Avg perturbed weighted F1 delta"],
    ):
        values = ablations[column]
        colors = [COLORS["positive"] if value > 0 else COLORS["negative"] for value in values]
        ax.barh(y, values, color=colors, height=0.56)
        ax.axvline(0.0, color=COLORS["muted"], linewidth=1.4)
        ax.grid(True, axis="x", linestyle="--", linewidth=0.8)
        ax.set_title(title, fontsize=16, pad=10)
        ax.set_xlabel("Delta vs full robust")
        ax.set_yticks(y)
        ax.set_yticklabels(ablations["label"])
        ax.set_xlim(-0.0105, 0.0155)
        for ypos, value in zip(y, values):
            align = "left" if value >= 0 else "right"
            offset = 0.0006 if value >= 0 else -0.0006
            ax.text(value + offset, ypos, f"{value:+.3f}", va="center", ha=align, fontweight="bold")

    fig.text(
        0.5,
        0.055,
        f"Single-seed only. Based on seed {ablation_seed}; do not treat this as a stable ranking.",
        ha="center",
        fontsize=11,
        color=COLORS["muted"],
    )
    path = output_dir / "slide_09_ablation_deltas.png"
    save_figure(fig, path)
    return path


def plot_gate_response(predictions_df: pd.DataFrame, output_dir: Path) -> Path:
    means = predictions_df.groupby("condition")[["gate_fused", "gate_audio", "gate_vision"]].mean().reindex(CONDITION_ORDER)
    x = np.arange(len(means.index))

    fig, axes = plt.subplots(1, 3, figsize=SLIDE_SIZE, sharex=True)
    fig.suptitle("Gate Response by Condition", fontsize=24, fontweight="bold", y=0.98)
    fig.subplots_adjust(bottom=0.17, top=0.88, wspace=0.22)

    plot_specs = [
        ("gate_fused", "Fused gate", COLORS["vanilla"], (0.94, 0.965)),
        ("gate_audio", "Audio gate", COLORS["audio"], (0.035, 0.058)),
        ("gate_vision", "Vision gate", COLORS["vision"], (0.034, 0.066)),
    ]

    for ax, (column, title, color, ylim) in zip(axes, plot_specs):
        ax.bar(x, means[column], color=color, width=0.62)
        ax.set_title(title, fontsize=16, pad=10)
        ax.set_ylim(*ylim)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(["Clean", "No\naudio", "No\nvision", "No both", "Mild\njitter"])
        for xpos, value in zip(x, means[column]):
            ax.text(xpos, value + (ylim[1] - ylim[0]) * 0.03, f"{value:.3f}", ha="center", va="bottom", fontsize=10)

    axes[0].set_ylabel("Mean gate value")
    fig.text(
        0.5,
        0.06,
        "Computed from the supplied predictions file. Gates are diagnostics only, not causal explanations.",
        ha="center",
        fontsize=11,
        color=COLORS["muted"],
    )
    path = output_dir / "slide_10_gate_response.png"
    save_figure(fig, path)
    return path


def write_visual_index(paths: list[Path], output_dir: Path, notes: list[str]) -> Path:
    lines = [
        "# PPT Visual Pack",
        "",
        "- Slide 5 or 7: `slide_05_model_tradeoff_map.png` shows where each model lands in clean-vs-perturbed space.",
        "- Slide 6: `slide_06_conditions_diagram.png` explains the perturbation setup.",
        "- Slide 7: `slide_07_seed_tradeoff.png` shows the paired seed-level clean/robustness trade-off.",
        "- Slide 8: `slide_08_transformer_condition_comparison.png` is the main result chart.",
        "- Slide 9: `slide_09_ablation_deltas.png` is honest about the single-seed ablation story.",
        "- Slide 10: `slide_10_gate_response.png` is the best diagnostic visual available from the stored outputs.",
        "",
        "Notes:",
    ]
    lines.extend(f"- {note}" for note in notes)
    lines.extend(["", "Generated files:"])
    lines.extend(f"- `{path.name}`" for path in paths)
    index_path = output_dir / "visual_index.md"
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index_path


def main() -> None:
    args = parse_args()
    apply_style()
    output_dir = ensure_dir(args.output)

    results_df = pd.read_csv(args.results)
    summary_df = pd.read_csv(args.summary)
    predictions_df = pd.read_csv(args.predictions)

    paths = [
        plot_model_tradeoff_map(summary_df, output_dir),
        plot_conditions_diagram(output_dir),
        plot_seed_tradeoff(summary_df, output_dir),
        plot_transformer_condition_comparison(results_df, output_dir),
        plot_ablation_deltas(summary_df, output_dir),
        plot_gate_response(predictions_df, output_dir),
    ]
    index_path = write_visual_index(paths, output_dir, _visual_index_notes(summary_df, results_df))

    print(f"Wrote visual pack to {output_dir}")
    print(f"Wrote index to {index_path}")


if __name__ == "__main__":
    main()
