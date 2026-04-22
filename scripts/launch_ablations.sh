#!/bin/bash
# Fires off one Slurm job per shaping variant. Run from the repo root.
set -euo pipefail

VARIANTS=(
    V0_baseline
    V1_support_front
    V2_touch_bonus
    V3_proximity_up
    V4_aggressive_combo
)

for v in "${VARIANTS[@]}"; do
    sbatch \
        --export=ALL,VARIANT="${v}" \
        --job-name="abl_${v}" \
        scripts/soccerstwos_ablation.batch
done
