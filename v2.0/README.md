<div align="center">
  <h1>🚀 MultiMod v2.0</h1>
  <h3><i>Evidential Information-Decoupled Multimodal Sentiment Analysis</i></h3>
  
  <p>
    <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python&logoColor=white" alt="Python"></a>
    <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg?logo=pytorch&logoColor=white" alt="PyTorch"></a>
    <a href="https://github.com/lexus-x/Robust-MOSEI-Sentiment-Classification/actions"><img src="https://img.shields.io/badge/Build-Passing-brightgreen.svg" alt="Build Status"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-purple.svg" alt="License"></a>
  </p>
</div>

---

## 🧭 North Star Claim

> **"A compact, calibrated multimodal sentiment system can remain competitive under realistic modality failure and temporal shift, while exposing uncertainty and selective abstention."**

This is not another MOSEI leaderboard script pile. This subproject represents the **v2.0 evolution** of the repository, transitioning from a baseline robustness study into a thesis-shaped program backed by rigorous evidence gates.

## 🧠 Architecture Highlights

* **EIDMSA Framework**: A lightweight, parameter-efficient model structure.
* **Information Bottleneck & PID**: Neural Partial Information Decomposition to isolate unique modality contributions.
* **Evidential Deep Learning**: Calibrated uncertainty estimation allowing the model to abstain under heavy corruption.
* **Test-Time Adaptation (TTA)**: Dynamic alignment to distribution shifts.

## 📁 Repository Map

```text
v2.0/
├── docs/               # 📄 Core thesis claims & roadmap
├── src/                # ⚙️ Source code for real sentiment pipelines
│   ├── claim.py        # Thesis gates & benchmark manifest
│   ├── corruptions.py  # Realistic feature-level noise operators
│   └── reporting.py    # Automated Markdown report generation
├── scripts/            # 🛠️ Execution endpoints
└── tests/              # 🧪 Pytest suites
```

## ⚡ Quick Start

**1. Generate the Evidence Pack:**
Compiles the latest bootstrap evidence, benchmark manifests, and renders the proposal report.
```bash
python v2.0/scripts/build_proposal_pack.py
```

**2. Execute the Test Suite:**
```bash
python -m pytest -q v2.0/tests
```

## 📊 Deliverables

The proposal pipeline automatically compiles artifacts into `v2.0/outputs/proposal_pack/`:
* `thesis_claim.json`
* `benchmark_manifest.json`
* `bootstrap_evidence.json`
* `report.md` (Human-readable summary answering the core thesis gates)
