#!/usr/bin/env bash
# Sweep SC training (LLaMA 3.1 8B local): orders 1–3 × seeds 42–44.
set -eu
(set -o pipefail) 2>/dev/null && set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"
CL_ROOT="${ROOT}/data//CL"

MODEL_NAME="${HOME}/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B"

for ORDER in 1 2 3; do
  for SEED in 42 43 44; do
    OUT_DIR="${ROOT}/outputs/train/train_sc_llama/order${ORDER}/seed${SEED}"
    mkdir -p "${OUT_DIR}"
    echo "[run] order_id=${ORDER} seed=${SEED} out_dir=${OUT_DIR}"
    echo "[run] model_name=${MODEL_NAME}"

    python "${ROOT}/FiUni/train_sc/train_sc_stream_qwen.py" \
      --model_name "${MODEL_NAME}" \
      --order_id "${ORDER}" \
      --seed "${SEED}" \
      --out_dir "${OUT_DIR}" \
      --batch_size 4 \
      --rank 128 \
      --detect_rank 8 \
      --expand_delta_rank 8 \
      --filora_dropout 0.1 \
      --max_source_len 512 \
      --max_target_len 32 \
      --t_low 0.6 \
      --t_high 0.7 \
      --layer_selection all \
      --detect_layer_selection '31' \
      --target_modules "q,v" \
      --expand_cooldown_steps 40 \
      --new_cooldown_steps 10 \
      --lr 1e-4 \
      --weight_decay 0.0 \
      --gradient_accumulation_steps 8 \
      --cl_root "${CL_ROOT}" \
      --fisher_ag_group_size 8 \
      --eval_batch_size 4 \
      --use_bfloat16 \
      --device "cuda:0"
  done
done
