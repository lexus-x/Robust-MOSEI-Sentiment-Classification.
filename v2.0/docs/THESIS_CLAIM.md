# Thesis Claim

## North-Star Claim

A compact multimodal sentiment model with explicit uncertainty and abstention can remain competitive with larger baselines under realistic modality failure, temporal misalignment, and compound cross-modal corruption.

## What "Competitive" Means

Not vague.

The eventual thesis claim should require all of:

1. Standard MOSEI metrics on clean evaluation are within a narrow tolerance of a strong robust baseline.
2. Robustness remains competitive under a benchmark with realistic missingness, lag, drift, burst corruption, and compound failure.
3. Calibration or selective-risk behavior is better, not just similar.
4. The model remains materially smaller or cheaper than the baseline.
5. The result transfers beyond one benchmark split or one synthetic protocol.

## What Does Not Count

- another custom metric setup with no standard comparison
- synthetic jitter alone
- one-seed ablations
- "close enough" without gates
- claiming robustness with no abstention or calibration story
- claiming impact with no benchmark that others can reuse

## Immediate Defensible Claim

The only claim imported from the old project is a bootstrap one:

> On the repo's current 3-class robustness protocol, `EIDMSA` preserves most of the robust transformer's clean and perturbed weighted-F1 while using far fewer parameters and smaller checkpoints.

That is not the thesis claim. It is the starting point.
