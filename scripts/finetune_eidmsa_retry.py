#!/usr/bin/env python
"""Fine-tune an existing EIDMSA run with the realistic retry settings."""

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

from multimod.config import load_experiment_config, save_config
from multimod.data import build_dataloaders, describe_dataset
from multimod.evaluation import evaluate_eidmsa
from multimod.models import InputDims, build_model
from multimod.models.eidmsa import EIDMSA
from multimod.training import train_model_eidmsa
from multimod.utils import count_parameters, ensure_dir, resolve_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Existing EIDMSA seed directory with config.yaml and best_model.pt.")
    parser.add_argument("--output", default="outputs/eidmsa_realistic_finetune", help="Output directory root.")
    parser.add_argument("--device", default="auto", help="Device name, for example auto/cpu/cuda.")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Finetuning learning rate.")
    parser.add_argument("--max-epochs", type=int, default=8, help="Maximum finetuning epochs.")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience.")
    parser.add_argument("--audio-dropout", type=float, default=0.10, help="Per-sample audio dropout probability.")
    parser.add_argument("--vision-dropout", type=float, default=0.30, help="Per-sample vision dropout probability.")
    parser.add_argument("--jitter-prob", type=float, default=0.15, help="Local jitter augmentation probability.")
    parser.add_argument(
        "--realistic-corruption-p",
        type=float,
        default=0.50,
        help="Probability of applying one realistic corruption family during training.",
    )
    parser.add_argument(
        "--alignment-weight",
        type=float,
        default=0.10,
        help="Clean-to-corrupted consistency loss weight.",
    )
    return parser.parse_args()


def _seed_from_run_dir(run_dir: Path) -> int:
    if not run_dir.name.startswith("seed_"):
        raise ValueError(f"Could not infer seed from run directory name: {run_dir}")
    return int(run_dir.name.split("_", maxsplit=1)[1])


def main() -> None:
    args = parse_args()
    source_run_dir = Path(args.run_dir)
    if not (source_run_dir / "config.yaml").exists() or not (source_run_dir / "best_model.pt").exists():
        raise SystemExit(f"Run directory must contain config.yaml and best_model.pt: {source_run_dir}")

    config = load_experiment_config(source_run_dir / "config.yaml")
    if config.model.name != "eidmsa":
        raise SystemExit(f"Expected an EIDMSA run, found model={config.model.name!r}")

    seed = _seed_from_run_dir(source_run_dir)
    set_seed(seed)
    device = resolve_device(args.device)

    config.experiment_name = "eidmsa_realistic_finetune"
    config.output_dir = args.output
    config.training.learning_rate = args.learning_rate
    config.training.max_epochs = args.max_epochs
    config.training.patience = args.patience
    config.model.audio_dropout_p = args.audio_dropout
    config.model.vision_dropout_p = args.vision_dropout
    config.model.jitter_prob = args.jitter_prob
    config.model.realistic_corruption_p = args.realistic_corruption_p
    config.model.alignment_weight = args.alignment_weight
    config.model.evidential_warmup_epochs = 0
    config.notes = (
        f"{config.notes} Finetuned from {source_run_dir} with realistic retry settings."
    ).strip()

    run_dir = ensure_dir(Path(config.output_dir) / config.experiment_name / f"seed_{seed}")
    save_config(config, run_dir / "config.yaml")

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
    model.load_state_dict(torch.load(source_run_dir / "best_model.pt", map_location=device))
    if not isinstance(model, EIDMSA):
        raise SystemExit("Loaded model is not EIDMSA after build_model.")

    trained_model, history = train_model_eidmsa(
        model=model,
        dataloaders=dataloaders,
        config=config,
        device=device,
        seed=seed,
    )

    checkpoint_path = run_dir / "best_model.pt"
    torch.save(trained_model.state_dict(), checkpoint_path)
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    results = evaluate_eidmsa(
        model=trained_model,
        dataloader=dataloaders["test"],
        conditions=config.data.conditions,
        device=device,
        output_dir=run_dir,
        use_tta=config.model.use_tta,
        tta_lr=config.model.tta_lr,
        tta_steps=config.model.tta_steps,
    )
    results["run"] = {
        "experiment": config.experiment_name,
        "seed": seed,
        "device": str(device),
        "num_parameters": count_parameters(trained_model),
        "checkpoint": str(checkpoint_path),
        "notes": config.notes,
        "source_checkpoint": str(source_run_dir / "best_model.pt"),
    }
    save_json(results, run_dir / "metrics.json")

    summary = results["summary"]
    print(f"Seed: {seed}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Clean weighted F1: {summary['clean_weighted_f1']:.4f}")
    print(f"Average perturbed weighted F1: {summary['avg_perturbed_weighted_f1']:.4f}")


if __name__ == "__main__":
    main()
