#!/usr/bin/env python
"""Re-evaluate saved checkpoints without retraining."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multimod.config import load_experiment_config
from multimod.data import build_dataloaders, describe_dataset
from multimod.evaluation import evaluate, evaluate_eidmsa
from multimod.models import InputDims, build_model
from multimod.models.eidmsa import EIDMSA
from multimod.utils import count_parameters, resolve_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        help="Run directory with config.yaml/best_model.pt, or a root directory containing many run dirs.",
    )
    parser.add_argument("--device", default="auto", help="Device name, for example auto/cpu/cuda.")
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Evaluate the clean condition only instead of the full configured condition set.",
    )
    return parser.parse_args()


def _collect_run_dirs(path: Path) -> list[Path]:
    if (path / "config.yaml").exists() and (path / "best_model.pt").exists():
        return [path]
    return sorted(
        run_dir.parent
        for run_dir in path.rglob("config.yaml")
        if (run_dir.parent / "best_model.pt").exists()
    )


def _seed_from_run_dir(run_dir: Path) -> int | None:
    if run_dir.name.startswith("seed_"):
        try:
            return int(run_dir.name.split("_", maxsplit=1)[1])
        except ValueError:
            return None
    return None


def _evaluate_run(run_dir: Path, device_name: str, clean_only: bool) -> dict[str, object]:
    config = load_experiment_config(run_dir / "config.yaml")
    device = resolve_device(device_name)
    dataloaders = build_dataloaders(
        data_path=config.data.data_path,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        max_seq_len=config.data.max_seq_len,
        lower_threshold=config.data.lower_threshold,
        upper_threshold=config.data.upper_threshold,
        label_mode=config.data.label_mode,
    )
    stats = describe_dataset(
        data_path=config.data.data_path,
        split="train",
        max_seq_len=config.data.max_seq_len,
        label_mode=config.data.label_mode,
    )
    model = build_model(
        config.model,
        input_dims=InputDims(text=stats.text_dim, audio=stats.audio_dim, vision=stats.vision_dim),
    ).to(device)
    state_dict = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state_dict)
    conditions = ("clean",) if clean_only else config.data.conditions

    if isinstance(model, EIDMSA):
        results = evaluate_eidmsa(
            model=model,
            dataloader=dataloaders["test"],
            conditions=conditions,
            device=device,
            output_dir=run_dir,
            use_tta=config.model.use_tta,
            tta_lr=config.model.tta_lr,
            tta_steps=config.model.tta_steps,
        )
    else:
        results = evaluate(
            model=model,
            dataloader=dataloaders["test"],
            conditions=conditions,
            device=device,
            output_dir=run_dir,
            diagnostics_examples=config.diagnostics_examples,
        )

    results["run"] = {
        "experiment": config.experiment_name,
        "seed": _seed_from_run_dir(run_dir),
        "device": str(device),
        "num_parameters": count_parameters(model),
        "checkpoint": str(run_dir / "best_model.pt"),
        "notes": config.notes,
    }
    save_json(results, run_dir / "metrics.json")
    return results


def main() -> None:
    args = parse_args()
    target = Path(args.path)
    run_dirs = _collect_run_dirs(target)
    if not run_dirs:
        raise SystemExit(f"No run directories with config.yaml and best_model.pt found under {target}")

    condition_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        results = _evaluate_run(run_dir=run_dir, device_name=args.device, clean_only=args.clean_only)
        run = results["run"]
        summary = results["summary"]
        summary_rows.append({**run, **summary})
        for row in results["conditions"]:
            condition_rows.append({**run, **summary, **row})
        summary = results["summary"]
        print(f"{run_dir}: clean_f1={summary['clean_weighted_f1']:.4f}", end="")
        if "clean_mosei_acc_7" in summary:
            print(
                f" acc7={summary['clean_mosei_acc_7']:.4f}"
                f" acc2_nonneg={summary['clean_mosei_acc_2_nonneg']:.4f}"
                f" mae={summary['clean_mosei_mae']:.4f}",
                end="",
            )
        print()

    if len(run_dirs) > 1 or target != run_dirs[0]:
        output_dir = target if target.is_dir() else target.parent
        pd.DataFrame(condition_rows).to_csv(output_dir / "reevaluated_condition_metrics.csv", index=False)
        pd.DataFrame(summary_rows).to_csv(output_dir / "reevaluated_run_summary.csv", index=False)
        print(output_dir / "reevaluated_run_summary.csv")


if __name__ == "__main__":
    main()
