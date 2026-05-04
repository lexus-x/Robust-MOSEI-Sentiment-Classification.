# PhD Realistic Sentiment

This is a separate project inside the repo. It is not another MOSEI leaderboard script pile.

Its purpose is to turn the old project into a thesis-shaped program with a sharper claim:

> A compact, calibrated multimodal sentiment system can remain competitive under realistic modality failure and temporal shift, while exposing uncertainty and selective abstention.

## Why This Exists

The old project could defend an efficiency claim on a custom 3-class CMU-MOSEI protocol, but it could not honestly defend:

- current CMU-MOSEI SOTA
- real-world robustness beyond synthetic stress tests
- strong published-standard-metric superiority

This subproject fixes the framing. The benchmark, evidence gates, and reports are built around a real PhD question instead of an overclaimed model stack.

## What Is In Here

- `docs/THESIS_CLAIM.md`: the north-star claim and what would count as proof
- `docs/ROADMAP.md`: staged plan from bootstrap evidence to thesis-grade evidence
- `src/real_sentiment/claim.py`: thesis claim, benchmark manifest, acceptance gates
- `src/real_sentiment/corruptions.py`: realistic feature-level corruption operators
- `src/real_sentiment/bootstrap.py`: imports current repo results as bootstrap evidence
- `src/real_sentiment/reporting.py`: renders the proposal and progress report
- `scripts/build_proposal_pack.py`: generates the initial evidence/progress/report pack

## Current Status

The current repo only proves a bootstrap result:

- compact `EIDMSA` keeps most of the robust transformer's current synthetic-protocol performance
- it does so with a large parameter and checkpoint reduction

That is useful starting evidence, but it is not the thesis claim.

## Quick Start

Build the proposal pack:

```bash
python phd_realistic_sentiment/scripts/build_proposal_pack.py
```

Run the subproject tests:

```bash
python -m pytest -q phd_realistic_sentiment/tests
```

## Deliverables

The proposal pack writes to `phd_realistic_sentiment/outputs/proposal_pack/`:

- `thesis_claim.json`
- `benchmark_manifest.json`
- `bootstrap_evidence.json`
- `report.md`

The generated report is the blunt answer to three questions:

1. What exactly is the new claim?
2. What evidence exists today?
3. What is still missing before the claim is proven?
