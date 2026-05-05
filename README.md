<div align="center">
  <img src="https://raw.githubusercontent.com/lexus-x/Robust-MOSEI-Sentiment-Classification/main/docs/assets/logo.png" alt="MultiMod Logo" width="150" onerror="this.style.display='none'">
  <h1>🌌 Robust MOSEI Sentiment Classification</h1>
  <p><i>A dual-pipeline repository bridging baseline robustness and advanced multimodal thesis research.</i></p>

  <p>
    <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python&logoColor=white" alt="Python"></a>
    <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg?logo=pytorch&logoColor=white" alt="PyTorch"></a>
    <a href="https://github.com/lexus-x/Robust-MOSEI-Sentiment-Classification/actions"><img src="https://img.shields.io/badge/Build-Passing-brightgreen.svg" alt="Build Status"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-purple.svg" alt="License"></a>
  </p>
</div>

---

## 📖 Overview

This repository houses two tightly integrated environments designed for high-end research in **Multimodal Sentiment Analysis**:

1. **`v1.0` Baseline Robustness** 🏗️: Extensive experiments on CMU-MOSEI using transformer architectures with cross-modality gating.
2. **`v2.0` EIDMSA Extension** 🚀: A PhD-level framework integrating Information Bottleneck, Neural Partial Information Decomposition (PID), Evidential Deep Learning, and Test-Time Adaptation (TTA). *(See [`v2.0/README.md`](v2.0/README.md) for details)*.

---

## 🛠️ Installation

```bash
# Core installation
pip install -e .

# Optional: Real Mamba SSM support (requires CUDA + compatible PyTorch)
pip install -e '.[mamba]'
```
> **⚠️ Note on Mamba**: If `mamba-ssm` is not present, experiments will fall back to a Conv+GRU encoder. A `RuntimeWarning` is emitted to prevent accidental misreporting.

---

## 🚀 Quick Start: Baseline Study

**Run a single controlled experiment:**
```bash
python scripts/train.py \
  --data /path/to/mosei_raw.pkl \
  --experiment xmodal_transformer_robust \
  --seed 13
```

**Execute the full study + ablations:**
```bash
python scripts/run_experiments.py \
  --data /path/to/mosei_raw.pkl \
  --output outputs/main_run \
  --run-ablations
```

---

## 🧪 Quick Start: EIDMSA (v2.0)

EIDMSA exposes a highly configurable experiment runner.

**Run the full baseline architecture (3 seeds):**
```bash
python scripts/run_eidmsa_experiments.py \
  --data /path/to/mosei_raw.pkl \
  --output outputs/eidmsa_run
```

### Experiment Matrix

| Flag | Topology & Target |
|:---|:---|
| *(default)* | `eidmsa` — full model, 3 random seeds |
| `--run-ablations` | `eidmsa_no_ib`, `eidmsa_no_pid`, `eidmsa_no_evidential` |
| `--run-7class` | `eidmsa_7class` — 7-class ordinal sentiment classification |
| `--run-tta` | `eidmsa_tta` — test-time adaptation across domains |
| `--run-kan` | `eidmsa_kan` — KAN projection heads (zero-dependency) |
| `--run-mamba` | `eidmsa_mamba` — Mamba sequence modeling |

> Combine flags dynamically (e.g., `--run-novel --run-baselines`). The runner automatically deduplicates identical configurations.

---

## 📈 Evaluation & Artifacts

All scripts automatically aggregate metrics and compile publication-ready markdown reports.

```bash
# 1. Generate visual figures
python scripts/plot_results.py \
  --results outputs/main_run/aggregate_results.csv \
  --output outputs/main_run/plots

# 2. Compile the final markdown report
python scripts/build_final_report.py \
  --summary outputs/main_run/run_summary.csv \
  --aggregate outputs/main_run/aggregate_results.csv \
  --acceptance outputs/main_run/acceptance_summary.json \
  --output outputs/main_run/final_report.md
```

### Metrics Exported
We calculate rigorous CMU-MOSEI metrics based on class-expectations (without dedicated regression heads, ensuring honest baselines):
* `clean_mosei_mae`, `clean_mosei_corr`
* `clean_mosei_acc_7`, `clean_mosei_acc_2_nonneg`, `clean_mosei_acc_2_negpos`
* `clean_mosei_f1_nonneg`, `clean_mosei_f1_negpos`

---

## ⚠️ Known Limitations

- **Synthetic Jitter**: `mild_jitter` is a controlled stress test. It does not perfectly simulate real-world physical desyncs.
- **Mamba Fallbacks**: The framework seamlessly degrades to Conv+GRU on CPU architectures. Always verify the logs when running `--run-mamba` for publication.
- **KAN B-splines**: Initialized on `[-1, 1]`. For unnormalized features, explicitly trigger `layer.update_grid(x)` on representative batches.

---
<div align="center">
  <i>Engineered for robust sentiment analysis and adaptive multimodality.</i>
</div>
## Authors
* **ZHANG MING** - *AI PhD Candidate, Changwon National University*
