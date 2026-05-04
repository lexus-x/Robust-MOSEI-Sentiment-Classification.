# Roadmap

## Stage 0: Bootstrap

- import current repo evidence
- lock the thesis claim and forbidden claims
- define the benchmark conditions and proof gates

## Stage 1: Realistic Benchmark

- replace single `mild_jitter` story with a protocol family:
  - block missingness
  - temporal lead/lag
  - drift across a clip
  - local burst corruption
  - compound failures across modalities
- publish severity-controlled manifests
- add calibration and abstention metrics

## Stage 2: Model Work

- keep the compact evidential direction only if it helps the claim
- remove dead complexity that does not help robustness, calibration, or efficiency
- compare against at least one strong robust baseline under the new benchmark

## Stage 3: Standard Metrics

- train or evaluate with standard MOSEI metrics as first-class outputs
- stop centering the story on only custom 3-class weighted-F1

## Stage 4: Transfer

- add at least one out-of-domain or cross-dataset transfer check
- prove that the benchmark is not overfit to one preprocessing pipeline

## Stage 5: Thesis Pack

- final claim report
- benchmark release docs
- model card
- reproducibility script
- negative results section
