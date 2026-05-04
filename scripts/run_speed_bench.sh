#!/bin/bash
set -e

DATA="data/mosei_raw.pkl"
OUT="outputs/eidmsa_gpu_fast"

for EXP in eidmsa eidmsa_kan eidmsa_mamba eidmsa_kan_mamba xmodal_transformer_robust; do
    echo "=============================="
    echo "Running $EXP"
    echo "=============================="
    python scripts/train.py --experiment "$EXP" --seed 13 --data "$DATA" --output "$OUT" --device cuda
done
