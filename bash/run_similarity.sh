# Edit hyperparameters directly here.
LAYER_SELECTION="11"
TARGET_MODULES="k,q,v,o"
FEW_SHOT=32
RANK=2
BATCH_SIZE=32

python "FiUni/similarity/eval_fisher_uv_similarity.py" \
  --few_shot "${FEW_SHOT}" \
  --rank "${RANK}" \
  --batch_size "${BATCH_SIZE}" \
  --cl_root "./data/CL_clean" \
  --out_root "outputs/similarity" \
  --use_bfloat16 \
  --show_progress \
  --layer_selection "${LAYER_SELECTION}" \
  --target_modules "${TARGET_MODULES}"
