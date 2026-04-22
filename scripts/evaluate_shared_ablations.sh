#!/bin/bash
# Evaluate shared-policy ablation checkpoints against a fixed opponent.
set -euo pipefail

EPISODES="${EPISODES:-20}"
MAX_STEPS="${MAX_STEPS:-1000}"
OPPONENT="${OPPONENT:-/tmp/sct_fresh/ceia_baseline_agent}"
OPPONENT_NAME="${OPPONENT_NAME:-ceia_baseline}"
CHECKPOINT_TEMPLATE="${CHECKPOINT_TEMPLATE:-/tmp/sct_fresh/TEAM4_AGENT_REWARD}"
OUT_DIR="${OUT_DIR:-eval_results}"

VARIANTS=(
    V2_touch_bonus
    V4_aggressive_combo
    V5_dynamic_aggressive
)

mkdir -p "${OUT_DIR}"

for variant in "${VARIANTS[@]}"; do
    run_dir="ray_results/PPO_curriculum_shared_${variant}"
    if [[ ! -d "${run_dir}" ]]; then
        echo "[skip] missing ${run_dir}"
        continue
    fi

    echo "[eval] shared ${variant} vs ${OPPONENT_NAME}"
    python scripts/evaluate_agents.py \
        --agent-a-checkpoint "${run_dir}" \
        --agent-a-name "shared_${variant}" \
        --agent-a-policy-ids default \
        --agent-b "${OPPONENT}" \
        --agent-b-name "${OPPONENT_NAME}" \
        --checkpoint-template "${CHECKPOINT_TEMPLATE}" \
        --episodes "${EPISODES}" \
        --max-steps "${MAX_STEPS}" \
        --swap-sides \
        --csv "${OUT_DIR}/shared_${variant}_vs_${OPPONENT_NAME}.csv"
done
