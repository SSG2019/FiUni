import argparse
import json
import os
import re
import string
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from contextlib import nullcontext
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq

from fiunilib.cl_benchmark_data import prepare_ls_tasks_from_cl
from fiunilib.repro import set_global_seed
from fiunilib.train_strategy import (
    LSTrainOrchestrator,
    build_ls_stream,
    build_seq2seq_dataset,
    loss_fn,
)


def visualize(
    decisions,
    losses,
    boundary_batch_positions,
    task_names: list,
    total_batches: int,
    out_svg: Path,
):
    x = [d["step"] for d in decisions]
    x_new = [d["step"] for d in decisions if d["action"] == "NEW"]
    x_expand = [d["step"] for d in decisions if d["action"] == "EXPAND"]
    x_reuse = [d["step"] for d in decisions if d["action"] == "REUSE"]

    y_loss = [np.nan if v is None else v for v in losses]
    x_plot, y_plot = [], []
    prev_step = None
    for step_i, loss_i in zip(x, y_loss):
        if prev_step is not None and step_i - prev_step > 1:
            # Break line across skipped (non-training) steps.
            x_plot.append(np.nan)
            y_plot.append(np.nan)
        x_plot.append(step_i)
        y_plot.append(loss_i)
        prev_step = step_i

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 9), sharex=True)

    ax1.plot(x_plot, y_plot, linewidth=2.4, color="#528FAD", label="train loss", zorder=2)
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.spines["top"].set_visible(False)

    # Bottom panel: same layout as FiUni/task_boundary_detect/detect_ls_boundaries.py visualize()
    seg_starts = [0] + boundary_batch_positions
    seg_ends = boundary_batch_positions + [total_batches]
    for i, (s, e) in enumerate(zip(seg_starts, seg_ends)):
        mid = 0.5 * (s + e)
        task_label = task_names[i] if i < len(task_names) else f"task_{i}"
        ax2.text(mid, 2.35, task_label, ha="center", va="bottom", fontsize=11, color="#4a5568", rotation=20)

    if x_reuse:
        ax2.scatter(x_reuse, [0.0] * len(x_reuse), marker="o", s=34, color="#2563eb", label="REUSE", zorder=3)
    if x_expand:
        ax2.scatter(x_expand, [1.0] * len(x_expand), marker="^", s=42, color="#ea580c", label="EXPAND", zorder=3)
    if x_new:
        ax2.scatter(x_new, [2.0] * len(x_new), marker="*", s=80, color="#16a34a", label="NEW", zorder=3)

    for xb in boundary_batch_positions:
        ax1.axvline(x=xb, linestyle="--", linewidth=1, alpha=0.55, color="#64748b", zorder=1)
        ax2.axvline(x=xb, linestyle="--", linewidth=1, alpha=0.55, color="#64748b", zorder=1)

    for d in decisions:
        if d["action"] == "EXPAND" and d.get("best_id") is not None:
            ax2.text(d["step"], 1.12, d["best_id"], fontsize=10, ha="center", va="bottom", color="#ea580c", rotation=35)
        if d["action"] == "NEW" and d.get("created_id") is not None:
            ax2.text(d["step"], 2.10, d["created_id"], fontsize=10, ha="center", va="bottom", color="#166534", rotation=35)

    ax2.set_ylim(-0.2, 2.55)
    ax2.set_yticks([0.0, 1.0, 2.0], ["REUSE", "EXPAND", "NEW"])
    ax2.set_xlabel("Window End Batch Index")
    ax2.set_ylabel("Triggered State")
    ax2.grid(axis="y", linestyle=":", alpha=0.35)
    ax2.legend(loc="lower right")
    ax2.spines["top"].set_visible(False)

    plt.tight_layout()
    fig.align_ylabels((ax1, ax2))
    fig.savefig(out_svg, format="svg")
    plt.close(fig)


def _sanitize_model_for_path(model_name: str) -> str:
    return model_name.replace("\\", "_").replace("/", "_").replace(":", "_")


def _default_out_dir(model_name: str, order_id: int) -> Path:
    root = Path("outputs") / "train"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = _sanitize_model_for_path(model_name)
    return root / f"train_ls_{name}_order{order_id}_{stamp}"


def infer_task_subspace_ids(
    decisions: list,
    n_tasks: int,
    boundary_batch_positions: list,
    total_batches: int,
) -> list:
    """One subspace id per stream task, from majority vote of active_sid over that task's batch steps."""
    sids = []
    for k in range(n_tasks):
        start = 0 if k == 0 else boundary_batch_positions[k - 1]
        end = boundary_batch_positions[k] if k < n_tasks - 1 else total_batches
        cands = []
        for d in decisions:
            step = d["step"]
            if start <= step < end:
                sid = d.get("active_sid")
                if sid is not None:
                    cands.append(sid)
        if cands:
            sids.append(Counter(cands).most_common(1)[0][0])
        else:
            sids.append(None)
    return sids


def _decode_label_batch(tokenizer, labels: torch.Tensor) -> list:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    ids = labels.clone()
    ids[ids == -100] = pad_id
    return tokenizer.batch_decode(ids, skip_special_tokens=True)


def _normalize_for_exact_match(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def run_test_eval(
    orchestrator: LSTrainOrchestrator,
    ls_tasks,
    task_names: list,
    task_subspace_ids: list,
    tokenizer,
    model,
    device: torch.device,
    args,
    out_dir: Path,
    log_append,
):
    """Per-task test: loss (teacher-forced) + sequence exact match (greedy decode vs gold).

    Forward uses **all** existing FiLoRA subspaces (same as training); ``task_subspace_ids`` is
    kept only as metadata (majority sid during that task’s train window).
    """
    eval_bs = getattr(args, "eval_batch_size", None) or args.batch_size
    max_gen = getattr(args, "eval_max_new_tokens", None) or args.max_target_len
    results = {
        "per_task": {},
        "macro_avg_sequence_em_acc": None,
        "filora_forward": "sum_all_existing_subspaces",
    }
    per_task_em = []

    model.eval()
    orchestrator._set_forward_subspace("_eval_all_filora")

    for name, sid in zip(task_names, task_subspace_ids):
        td = ls_tasks[name]
        test_ds = build_seq2seq_dataset(td.test, tokenizer, args.max_source_len, args.max_target_len)
        collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors="pt")
        test_loader = DataLoader(test_ds, batch_size=eval_bs, shuffle=False, collate_fn=collator)

        nb_batches = 0
        n_examples = len(test_ds)
        em_correct = 0
        em_total = 0
        cast_ctx = (
            torch.autocast(device_type=device.type, dtype=torch.bfloat16)
            if args.use_bfloat16
            else nullcontext()
        )
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                inputs = {k: v for k, v in batch.items() if torch.is_tensor(v)}
                with cast_ctx:
                    gen_ids = model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_new_tokens=max_gen,
                        do_sample=False,
                        num_beams=1,
                    )
                nb_batches += 1

                refs = _decode_label_batch(tokenizer, batch["labels"])
                preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                for ref, pred in zip(refs, preds):
                    em_total += 1
                    em_correct += int(_normalize_for_exact_match(ref) == _normalize_for_exact_match(pred))

        em_acc = em_correct / em_total if em_total else float("nan")
        per_task_em.append(em_acc)
        results["per_task"][name] = {
            "sequence_exact_match_acc": em_acc,
            "em_correct": em_correct,
            "em_total": em_total,
            "train_window_majority_subspace_id": sid,
            "num_batches": nb_batches,
            "num_examples": n_examples,
        }
        line = (
            f"[Eval] task={name} train_win_sid={sid} seq_EM_acc={em_acc:.4f} "
            f"({em_correct}/{em_total}) batches={nb_batches} (FiLoRA=sum_all)"
        )
        print(line, flush=True)
        log_append(line)

    macro_em = float(np.nanmean(per_task_em)) if per_task_em else float("nan")
    results["macro_avg_sequence_em_acc"] = macro_em
    line = f"[Eval] macro_avg_sequence_em_acc (mean over tasks)={macro_em:.4f}"
    print(line, flush=True)
    log_append(line)

    out_path = out_dir / "ls_test_eval.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    done = f"[Done] Test eval: {out_path}"
    print(done, flush=True)
    log_append(done)


def merge_filora_into_base(orchestrator, log_append):
    """
    Merge all existing FiLoRA banks into each wrapped base linear:
      W <- W + sum_s (U_s @ R_s @ V_s^T)
    Then clear banks so eval forward uses base only.
    """
    merged_layers = 0
    for _, w in orchestrator.wrappers.items():
        if not w.R_bank:
            continue
        delta_w = torch.zeros_like(w.base.weight.data)
        for sid in w.R_bank.keys():
            u = w.U_bank[sid].data
            r = w.R_bank[sid].data
            v = w.V_bank[sid].data
            delta_w.add_(u @ r @ v.t())
        w.base.weight.data.add_(delta_w)
        w.set_active(None)
        w.U_bank = nn.ParameterDict()
        w.V_bank = nn.ParameterDict()
        w.R_bank = nn.ParameterDict()
        w.ranks = {}
        merged_layers += 1

    line = f"[Info] Merged FiLoRA into base weights for {merged_layers} layers."
    print(line, flush=True)
    log_append(line)


def run(args):
    set_global_seed(args.seed)
    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    out_dir_s = (args.out_dir or "").strip()
    if not out_dir_s:
        out_dir = _default_out_dir(args.model_name, args.order_id)
    else:
        out_dir = Path(out_dir_s)
    out_dir.mkdir(parents=True, exist_ok=True)

    args.out_dir = str(out_dir.resolve())

    log_path = out_dir / "ls_run_log.txt"

    def log_append(msg: str):
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    with log_path.open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"[Info] Device: {device}", flush=True)
    print(f"[Info] out_dir: {out_dir}", flush=True)
    log_append(f"[Info] Device: {device}")
    log_append(f"[Info] out_dir: {out_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.offline)
    load_kwargs = {}
    if args.use_bfloat16:
        load_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, local_files_only=args.offline, **load_kwargs)
    model.to(device)

    ls_tasks = prepare_ls_tasks_from_cl(
        cl_root=args.cl_root,
        order_id=args.order_id,
        seed=args.seed,
    )
    stream_ds, task_names, boundary_example_idx = build_ls_stream(ls_tasks)
    print(f"[Info] Stream tasks: {task_names}", flush=True)
    print(f"[Info] Stream size: {len(stream_ds)} examples", flush=True)

    tokenized_stream = build_seq2seq_dataset(stream_ds, tokenizer, args.max_source_len, args.max_target_len)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors="pt")
    loader = DataLoader(tokenized_stream, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    all_batches = list(loader)
    print(f"[Info] Total batches: {len(all_batches)}", flush=True)

    orchestrator = LSTrainOrchestrator(model=model, tokenizer=tokenizer, args=args)
    decisions, losses = orchestrator.run(all_batches)
    orchestrator.save_subspace_meta(out_dir)

    total_r_numel = orchestrator.total_r_parameters()
    r_line = f"[Info] Total FiLoRA R parameters (numel): {total_r_numel:,}"
    print(r_line, flush=True)
    log_append(r_line)

    boundary_batch_positions = [int(np.ceil(x / args.batch_size)) for x in boundary_example_idx[:-1]]
    visualize(
        decisions,
        losses,
        boundary_batch_positions,
        task_names,
        len(all_batches),
        out_dir / "ls_train_timeline.svg",
    )
    detect_decisions = [d for d in decisions if d.get("detect_step", True)]

    summary = {
        "task_names": task_names,
        "actions": {
            "NEW": sum(1 for d in detect_decisions if d["action"] == "NEW"),
            "EXPAND": sum(1 for d in detect_decisions if d["action"] == "EXPAND"),
            "REUSE": sum(1 for d in detect_decisions if d["action"] == "REUSE"),
            "WAIT": sum(1 for d in detect_decisions if d["action"] == "WAIT"),
        },
        "num_subspaces": len(orchestrator.subspaces),
        "boundary_batch_positions": boundary_batch_positions,
        "task_subspace_ids": infer_task_subspace_ids(
            decisions, len(task_names), boundary_batch_positions, len(all_batches)
        ),
        "total_r_parameters": total_r_numel,
        "config": vars(args),
    }

    with (out_dir / "ls_train_decisions.json").open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "decisions": decisions, "losses": losses}, f, ensure_ascii=False, indent=2)

    print("[Done] LS training completed.", flush=True)
    log_append("[Done] LS training completed.")
    print(f"[Done] Timeline: {out_dir / 'ls_train_timeline.svg'}", flush=True)
    print(f"[Done] Decisions: {out_dir / 'ls_train_decisions.json'}", flush=True)
    print(f"[Done] Subspaces: {out_dir / 'subspace_meta.json'}", flush=True)

    if not getattr(args, "skip_test_eval", False):
        merge_filora_into_base(orchestrator, log_append)
        run_test_eval(
            orchestrator,
            ls_tasks,
            task_names,
            summary["task_subspace_ids"],
            tokenizer,
            model,
            device,
            args,
            out_dir,
            log_append,
        )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="t5-base")
    p.add_argument("--order_id", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--detect_rank", type=int, default=4)
    p.add_argument("--expand_delta_rank", type=int, default=2)
    p.add_argument(
        "--expand_direct",
        action="store_true",
        help="On EXPAND: concat the first expand_delta_rank columns of new U,V without orthogonalizing to old U,V.",
    )
    p.add_argument("--max_source_len", type=int, default=256)
    p.add_argument("--max_target_len", type=int, default=32)
    p.add_argument("--t_low", type=float, default=0.40)
    p.add_argument("--t_high", type=float, default=0.65)
    p.add_argument("--layer_selection", type=str, default="all")
    p.add_argument("--detect_layer_selection", type=str, default="all")
    p.add_argument("--target_modules", type=str, default="q,k,v,o,wi,wo")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--filora_dropout", type=float, default=0.0, help="Dropout on FiLoRA delta (train only; off in eval).")
    p.add_argument("--expand_cooldown_steps", type=int, default=0)
    p.add_argument("--new_cooldown_steps", type=int, default=0)
    p.add_argument(
        "--filora_initial_max_optimizer_steps",
        type=int,
        default=None,
        help="If set, each NEW subspace may run at most this many optimizer.step() calls (after grad accum). "
        "Omit for unlimited (legacy). REUSE skips training when the chosen sid has exhausted its budget.",
    )
    p.add_argument(
        "--filora_expand_extra_optimizer_steps",
        type=int,
        default=0,
        help="When --filora_initial_max_optimizer_steps is set, each EXPAND adds this many extra allowed optimizer steps for that sid.",
    )
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--cl_root", type=str, default=None, help="Path to processed CL root (default: <repo>/CL).")
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Empty: outputs/train/train_ls_{model}_order{id}_{timestamp}; else use this path.",
    )
    p.add_argument("--eval_batch_size", type=int, default=None, help="Defaults to --batch_size.")
    p.add_argument(
        "--eval_max_new_tokens",
        type=int,
        default=None,
        help="Greedy decode length cap in test; default = --max_target_len.",
    )
    p.add_argument("--skip_test_eval", action="store_true")
    p.add_argument("--no_progress", action="store_true", help="Disable tqdm bar during orchestrator.run.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument(
        "--fisher_ag_group_size",
        type=int,
        default=1,
        help="During NEW/EXPAND only: split layer_selection into this many contiguous parts "
        "and run one forward+backward per part (>=1). 1 = one pass over all selected layers. "
        "Detection always uses one pass on detect_layer_selection; this flag does not apply there.",
    )
    p.add_argument("--use_bfloat16", action="store_true")
    p.add_argument("--offline", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
