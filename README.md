# Robust MOSEI Sentiment Classification

This repository has two main components:

1. **Baseline robustness study** — class-project experiments on CMU-MOSEI using transformer models with modality gating.
2. **EIDMSA research extension** — PhD-level framework integrating Information Bottleneck, Neural Partial Information Decomposition, Evidential Deep Learning, and Test-Time Adaptation.

## What This Repo Expects

The data loader targets the packed affect format used by MultiBench-style MOSEI preprocessing:

```python
{
  "train": {
    "text": np.ndarray [N, T, D_text],
    "audio": np.ndarray [N, T, D_audio],
    "vision": np.ndarray [N, T, D_vision],
    "labels": np.ndarray [N, 1] or [N, 1, 1],
    "id": list[str],
  },
  "valid": {...},
  "test": {...},
}
```

`text`, `audio`, and `vision` are word-aligned, left-padded sequences. Default label binning:

- `y < -0.5` → `negative`
- `-0.5 <= y <= 0.5` → `neutral`
- `y > 0.5` → `positive`

## Installation

```bash
# Core install
pip install -e .

# Optional: real Mamba SSM support (requires CUDA + compatible PyTorch)
pip install -e '.[mamba]'
```

> **Note on Mamba:** If `mamba-ssm` is not installed, the `eidmsa_mamba` experiment
> uses a conv+GRU fallback encoder, **not real Mamba SSM**. A `RuntimeWarning` is emitted
> at import time when this is the case. Do not report fallback results as Mamba results.

## Quick Start: Baseline Study

Run a single experiment:

```bash
python scripts/train.py \
  --data /path/to/mosei_raw.pkl \
  --experiment xmodal_transformer_robust \
  --seed 13
```

Run the main study plus ablations:

```bash
python scripts/run_experiments.py \
  --data /path/to/mosei_raw.pkl \
  --output outputs/main_run \
  --run-ablations
```

## Quick Start: EIDMSA Extension

Run the full EIDMSA model (3 seeds):

```bash
python scripts/run_eidmsa_experiments.py \
  --data /path/to/mosei_raw.pkl \
  --output outputs/eidmsa_run
```

Run ablations (single seed each):

```bash
python scripts/run_eidmsa_experiments.py \
  --data /path/to/mosei_raw.pkl \
  --output outputs/eidmsa_run \
  --run-ablations
```

Run novel paper integrations:

```bash
python scripts/run_eidmsa_experiments.py \
  --data /path/to/mosei_raw.pkl \
  --output outputs/eidmsa_run \
  --run-novel \
  --run-baselines
```

### Available EIDMSA Experiments

| Flag | Experiments |
|:---|:---|
| *(default)* | `eidmsa` — full model, 3 seeds |
| `--run-ablations` | `eidmsa_no_ib`, `eidmsa_no_pid`, `eidmsa_no_evidential` |
| `--run-7class` | `eidmsa_7class` — 7-class ordinal sentiment |
| `--run-tta` | `eidmsa_tta` — test-time adaptation |
| `--run-kan` | `eidmsa_kan` — KAN projection heads (no extra deps) |
| `--run-mamba` | `eidmsa_mamba` — Mamba/fallback encoder |
| `--run-novel` | All three: `eidmsa_kan`, `eidmsa_mamba`, `eidmsa_kan_mamba` |
| `--run-baselines` | `xmodal_transformer`, `xmodal_transformer_robust` |

Flags can be combined freely. Duplicate experiments are automatically deduplicated.

## Acceptance Summary (Baseline Study)

```bash
python scripts/run_experiments.py \
  --data /path/to/mosei_raw.pkl \
  --output outputs/main_run \
  --run-ablations \
  --clean-gap-tolerance 0.005 \
  --required-positive-seeds 3
```

- clean weighted-F1 drop tolerance: `0.01` (1 F1 point)
- required seeds with better perturbed weighted F1: `2`

## Plots and Reports

```bash
# Generate summary figures
python scripts/plot_results.py \
  --results outputs/main_run/aggregate_results.csv \
  --output outputs/main_run/plots

# Build markdown writeup
python scripts/build_final_report.py \
  --summary outputs/main_run/run_summary.csv \
  --aggregate outputs/main_run/aggregate_results.csv \
  --acceptance outputs/main_run/acceptance_summary.json \
  --output outputs/main_run/final_report.md
```

## Output Structure

- `outputs/.../metrics.json` — per-condition metrics for one run
- `outputs/.../predictions.csv` — predictions, uncertainty, conflict, PID components
- `outputs/.../history.csv` — training loss history
- `outputs/.../aggregate_results.csv` — all runs merged
- `outputs/.../eidmsa_run_summary.csv` — EIDMSA summary with uncertainty/ECE/conflict
- `outputs/.../acceptance_summary.json` — hypothesis check (baseline study only)
- `outputs/.../final_report.md` — human-readable summary

## Standard MOSEI Metrics From Saved Checkpoints

The repo now computes standard clean/perturbed MOSEI-style metrics from score-valued class expectations:

- `clean_mosei_mae`
- `clean_mosei_corr`
- `clean_mosei_acc_7`
- `clean_mosei_acc_2_nonneg`
- `clean_mosei_f1_nonneg`
- `clean_mosei_acc_2_negpos`
- `clean_mosei_f1_negpos`

These are written into `metrics.json` and `condition_metrics.csv`. To backfill them onto existing runs without retraining:

```bash
python scripts/evaluate_checkpoint.py outputs/eidmsa_gpu_final --device cpu
```

`mosei_score_mode` tells you how the score was derived. For the current classifiers this is class-expectation based, not a dedicated regression head, so treat comparisons to regression SOTA honestly.

## Real Video Work

If your actual goal is "upload one video and get useful work back", use the separate video-review path, not the MOSEI sentiment classifier:

- script: `scripts/review_incident_video.py`
- docs: `docs/video_incident_reviewer.md`

## Important Limitations

- `mild_jitter` is a controlled synthetic stress test on word-aligned features. It is intentionally *not* presented as a faithful model of real-world alignment failure.
- `eidmsa_mamba` uses conv+GRU (not Mamba SSM) unless `mamba-ssm` is installed. Install with `pip install -e '.[mamba]'` on a CUDA machine.
- Standard MOSEI metrics are currently derived from class expectations (`mosei_score_mode`) for 3-class / 7-class models. That is useful for comparison, but it is still weaker than a true regression setup.
- KAN layers use B-spline grids initialised on `[-1, 1]`. If your features are very far outside this range, call `layer.update_grid(x)` after a forward pass through representative data.
