#!/usr/bin/env bash
# Sweep TRACE training (LLaMA 3.1 8B local): order 7 × seeds 42–44.
set -eu
(set -o pipefail) 2>/dev/null && set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

TRACE_ROOT="${ROOT}/data/TRACE-benchmark/LLM-CL-Benchmark"
MODEL_NAME="${HOME}/.cache/modelscope/hub/models/LLM-Research/Meta-Llama-3.1-8B"
ORDER=7

for SEED in 42 43 44; do
  OUT_DIR="${ROOT}/outputs/train/train_trace_llama/order${ORDER}/seed${SEED}"
  mkdir -p "${OUT_DIR}"
  echo "[run] order_id=${ORDER} seed=${SEED} out_dir=${OUT_DIR}"
  echo "[run] model_name=${MODEL_NAME}"

  python "${ROOT}/FiUni/train_trace/train_trace_stream_qwen.py" \
    --model_name "${MODEL_NAME}" \
    --order_id "${ORDER}" \
    --seed "${SEED}" \
    --out_dir "${OUT_DIR}" \
    --task_epochs "1,1,1,1,1,1,1,1" \
    --task_batch_sizes "8,8,2,2,2,8,8,2" \
    --task_gradient_accumulation_steps "4,4,16,16,16,4,4,16" \
    --rank 128 \
    --detect_rank 8 \
    --expand_delta_rank 8 \
    --filora_dropout 0 \
    --max_source_len 1024 \
    --max_target_len 500 \
    --t_low 0.6 \
    --t_high 0.7 \
    --layer_selection all \
    --detect_layer_selection '31' \
    --target_modules "q,v" \
    --expand_cooldown_steps 100 \
    --new_cooldown_steps 100 \
    --lr 1e-4 \
    --weight_decay 0.0 \
    --trace_root "${TRACE_ROOT}" \
    --fisher_ag_group_size 8 \
    --eval_do_sample \
    --eval_num_beams 1 \
    --eval_temperature 0.95 \
    --eval_top_p 0.7 \
    --eval_top_k 50 \
    --use_bfloat16 \
    --device "cuda:0"
done
