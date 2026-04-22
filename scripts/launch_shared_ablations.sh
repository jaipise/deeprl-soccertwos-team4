#!/bin/bash
# Fires off one Slurm job per shared-policy shaping variant. Run from repo root.
set -euo pipefail

VARIANTS=(
    V2_touch_bonus
    V4_aggressive_combo
    V5_dynamic_aggressive
)

for v in "${VARIANTS[@]}"; do
    sbatch \
        --export=ALL,VARIANT="${v}" \
        --job-name="shared_${v}" \
        scripts/soccerstwos_shared_ablation.batch
done
