#!/usr/bin/env bash
 set -euo pipefail

# This script runs multiple hyperparameter combinations for:
#   FiUni/similarity/eval_fisher_uv_similarity.py
#
# Edit the lists below, and all cartesian-product combinations will run sequentially.
# Results for each run go into distinct subfolders under --out_root.

# Only manually enumerate combinations of these hyperparameters:
#   TARGET_MODULES, FEW_SHOT, RANK, LAYER_SELECTION
# with the constraint:
#   BATCH_SIZE == FEW_SHOT
#
# Format per combo string: "TARGET_MODULES|FEW_SHOT|RANK|LAYER_SELECTION"
# Example:
#   "q,k,v,o|32|64|11"decoder.block.11
COMBOS=(
  "q,k,v,o|32|8|all"
  "q,k,v,o|8|8|all"
  "q,k,v,o|2|8|all"
  "q,k,v,o|32|32|all"
  "q,k,v,o|32|16|all"
  "q,k,v,o|32|4|all"
  "q,k,v,o|32|2|all"
  "q,k,v,o|32|8|11"
  "q,k,v,o|32|8|0"
  "q,k,v,o,wi,wo|32|8|all"
  "q,o|32|8|all"
)

# Output + flags.
OUT_ROOT="outputs/similarity"
USE_BFLOAT16="--use_bfloat16"
SHOW_PROGRESS="--show_progress"

echo "[Info] Starting similarity runs from explicit combos..."
echo "[Info] COMBOS count: ${#COMBOS[@]}"
echo

idx=0
total=${#COMBOS[@]}
for combo in "${COMBOS[@]}"; do
  idx=$((idx+1))

  IFS='|' read -r TARGET_MODULES FEW_SHOT RANK LAYER_SELECTION <<< "${combo}"
  BATCH_SIZE="${FEW_SHOT}"

  echo "[$idx/$total] layer_selection=${LAYER_SELECTION}, target_modules=${TARGET_MODULES}, few_shot=${FEW_SHOT}, rank=${RANK}, batch_size=${BATCH_SIZE}"

  python "FiUni/similarity/eval_fisher_uv_similarity.py" \
    --few_shot "${FEW_SHOT}" \
    --rank "${RANK}" \
    --batch_size "${BATCH_SIZE}" \
    --cl_root "./data/CL_clean" \
    --out_root "${OUT_ROOT}" \
    ${USE_BFLOAT16} \
    ${SHOW_PROGRESS} \
    --layer_selection "${LAYER_SELECTION}" \
    --target_modules "${TARGET_MODULES}" \
    --device "cuda:0"

  echo
done

echo "[Done] All similarity runs completed."

