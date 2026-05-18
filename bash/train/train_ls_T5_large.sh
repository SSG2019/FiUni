#!/usr/bin/env bash
# Sweep LS training: orders 4–6 × seeds 42–44. Output under outputs/train/train_ls_T5_large/order{O}/seed{S}/.
set -eu
(set -o pipefail) 2>/dev/null && set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"
CL_ROOT="${ROOT}/data//CL"

for ORDER in 4 5 6; do
  for SEED in 42 43 44; do
    OUT_DIR="${ROOT}/outputs/train/train_ls_T5_large/order${ORDER}/seed${SEED}"
    mkdir -p "${OUT_DIR}"
    echo "[run] order_id=${ORDER} seed=${SEED} out_dir=${OUT_DIR}"

    python "${ROOT}/FiUni/train_ls/train_ls_stream.py" \
      --model_name "t5-large" \
      --order_id "${ORDER}" \
      --seed "${SEED}" \
      --out_dir "${OUT_DIR}" \
      --batch_size 32 \
      --rank 32 \
      --detect_rank 4 \
      --expand_delta_rank 4 \
      --filora_dropout 0.05 \
      --max_source_len 512 \
      --max_target_len 32 \
      --t_low 0.5 \
      --t_high 0.7 \
      --layer_selection all \
      --detect_layer_selection "23" \
      --target_modules "q,v,k,o" \
      --expand_cooldown_steps 200 \
      --new_cooldown_steps 10 \
      --lr 1e-3 \
      --weight_decay 0.0 \
      --gradient_accumulation_steps 1 \
      --cl_root "${CL_ROOT}" \
      --fisher_ag_group_size 6 \
      --use_bfloat16 \
      --device 'cuda:0'
  done
done
