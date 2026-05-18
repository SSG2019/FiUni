import argparse
import json
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq

from fiunilib.cl_benchmark_data import prepare_ls_tasks_from_cl
from fiunilib.fisher import compute_ag_batch_raw, get_UV
from fiunilib.similarity import TaskUV
from fiunilib.train_strategy import build_ls_stream, build_seq2seq_dataset, loss_fn
from fiunilib.task_boundary import (
    DecisionEngineLite,
    parse_layer_selection,
    parse_target_modules,
    select_t5_linear_modules,
)


def _sanitize_for_name(s: str) -> str:
    s = s.strip().lower()
    if not s:
        return "none"
    s = s.replace(" ", "")
    s = s.replace(",", "-")
    s = s.replace("[", "")
    s = s.replace("]", "")
    s = s.replace(".", "_")
    s = s.replace("/", "_")
    return s


def build_output_dir(args) -> Path:
    base = Path("outputs/task_boundary_detect")
    detect_layer_sel = args.detect_layer_selection if args.detect_layer_selection is not None else args.layer_selection
    run_name = (
        f"bs{args.batch_size}"
        f"_r{args.rank}"
        f"_dr{args.detect_rank}"
        f"_low{args.t_low}"
        f"_high{args.t_high}"
        f"_cd{args.expand_cooldown_steps}"
        f"_layer{_sanitize_for_name(args.layer_selection)}"
        f"_dlayer{_sanitize_for_name(detect_layer_sel)}"
        f"_target{_sanitize_for_name(args.target_modules)}"
    )
    return base / run_name


def visualize(decisions, boundary_batch_positions, task_names, total_batches: int, out_svg: Path):
    x_new = [d["step"] for d in decisions if d["action"] == "NEW"]
    x_expand = [d["step"] for d in decisions if d["action"] == "EXPAND"]
    x_reuse = [d["step"] for d in decisions if d["action"] == "REUSE"]

    fig, ax = plt.subplots(figsize=(18, 6))

    # Task labels by segment (without colored segment backgrounds).
    seg_starts = [0] + boundary_batch_positions
    seg_ends = boundary_batch_positions + [total_batches]
    for i, (s, e) in enumerate(zip(seg_starts, seg_ends)):
        mid = 0.5 * (s + e)
        task_label = task_names[i] if i < len(task_names) else f"task_{i}"
        ax.text(mid, 2.35, task_label, ha="center", va="bottom", fontsize=11, color="#4a5568", rotation=20)

    if x_reuse:
        ax.scatter(x_reuse, [0.0] * len(x_reuse), marker="o", s=34, color="#2563eb", label="REUSE", zorder=3)
    if x_expand:
        ax.scatter(x_expand, [1.0] * len(x_expand), marker="^", s=42, color="#ea580c", label="EXPAND", zorder=3)
    if x_new:
        ax.scatter(x_new, [2.0] * len(x_new), marker="*", s=80, color="#16a34a", label="NEW", zorder=3)

    for xb in boundary_batch_positions:
        ax.axvline(x=xb, linestyle="--", linewidth=1, alpha=0.55, color="#64748b", zorder=1)

    # Label EXPAND/NEW only.
    for d in decisions:
        if d["action"] == "EXPAND" and d["best_id"] is not None:
            ax.text(d["step"], 1.12, d["best_id"], fontsize=10, ha="center", va="bottom", color="#ea580c", rotation=35)
        if d["action"] == "NEW" and d["created_id"] is not None:
            ax.text(d["step"], 2.10, d["created_id"], fontsize=10, ha="center", va="bottom", color="#166534", rotation=35)

    ax.set_ylim(-0.2, 2.55)
    ax.set_yticks([0.0, 1.0, 2.0], ["REUSE", "EXPAND", "NEW"])
    ax.set_xlabel("Window End Batch Index")
    ax.set_ylabel("Triggered State")
    ax.set_title("Task Boundary Detection on LS Stream")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_svg, format="svg")
    plt.close(fig)


def run(args):
    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = build_output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Info] Device: {device}", flush=True)
    if args.detect_rank > args.rank:
        raise ValueError(f"detect_rank ({args.detect_rank}) must be <= rank ({args.rank}).")
    print(f"[Info] UV rank: {args.rank}, detect rank: {args.detect_rank}", flush=True)
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
    stream_ds, task_names, boundary_example_idx = build_ls_stream(ls_tasks, samples_per_task=args.samples_per_task)
    print(f"[Info] Stream tasks: {task_names}", flush=True)
    print(f"[Info] Stream size: {len(stream_ds)} examples", flush=True)

    tokenized_stream = build_seq2seq_dataset(stream_ds, tokenizer, args.max_source_len, args.max_target_len)
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors="pt")
    loader = DataLoader(tokenized_stream, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    train_layer_selection = parse_layer_selection(args.layer_selection)
    detect_layer_selection_raw = args.detect_layer_selection if args.detect_layer_selection is not None else args.layer_selection
    detect_layer_selection = parse_layer_selection(detect_layer_selection_raw)
    module_keywords = parse_target_modules(args.target_modules)
    full_layers = select_t5_linear_modules(model, train_layer_selection, module_keywords)
    detect_layers = select_t5_linear_modules(model, detect_layer_selection, module_keywords)
    if not full_layers:
        raise RuntimeError("No layers selected. Check layer_selection and target_modules.")
    if not detect_layers:
        raise RuntimeError("No layers selected. Check detect_layer_selection and target_modules.")
    detect_layer_set = set(detect_layers)
    full_layer_set = set(full_layers)
    same_layer_set = full_layer_set == detect_layer_set
    print(f"[Info] UV(full) linear layers: {len(full_layers)}", flush=True)
    print(f"[Info] UV(detect) linear layers: {len(detect_layers)}", flush=True)

    engine = DecisionEngineLite(
        t_low=args.t_low,
        t_high=args.t_high,
        expand_cooldown_steps=args.expand_cooldown_steps,
        new_cooldown_steps=args.new_cooldown_steps,
    )

    decisions: List[dict] = []
    step = -1

    for step, batch in enumerate(loader):
        stats = compute_ag_batch_raw(
            model=model,
            layer_refs=detect_layers,
            batch=batch,
            loss_fn=loss_fn,
            use_autocast=args.use_bfloat16,
            autocast_dtype=torch.bfloat16 if args.use_bfloat16 else None,
        )

        end_idx = step
        # One detection window equals one dataloader batch.
        # Always compute full UV with rank for downstream FiLoRA subspace construction,
        # then use only the first detect_rank directions for boundary detection.
        uv_detect_full = get_UV(stats, rank=args.rank)
        uv_detect = {
            name: (u[:, : args.detect_rank], v[:, : args.detect_rank])
            for name, (u, v) in uv_detect_full.items()
            if name in detect_layer_set
        }
        window_uv = TaskUV(name=f"window_{end_idx}", uv=uv_detect)
        decision = engine.update(step=end_idx, window_uv=window_uv)

        # On NEW/EXPAND, recompute UV on full layer_selection for this batch
        # and refresh the matched/new subspace representation.
        if decision.action in {"NEW", "EXPAND"}:
            if same_layer_set:
                uv_full = uv_detect_full
            else:
                full_stats = compute_ag_batch_raw(
                    model=model,
                    layer_refs=full_layers,
                    batch=batch,
                    loss_fn=loss_fn,
                    use_autocast=args.use_bfloat16,
                    autocast_dtype=torch.bfloat16 if args.use_bfloat16 else None,
                )
                uv_full = get_UV(full_stats, rank=args.rank)

            if decision.action == "NEW" and decision.created_id is not None:
                engine.subspaces[decision.created_id] = TaskUV(name=decision.created_id, uv=uv_full)
            elif decision.action == "EXPAND" and decision.best_id is not None:
                engine.subspaces[decision.best_id] = TaskUV(name=decision.best_id, uv=uv_full)

        decisions.append(
            {
                "step": decision.step,
                "action": decision.action,
                "best_id": decision.best_id,
                "best_sim": decision.best_sim,
                "created_id": decision.created_id,
            }
        )
        print(
            f"[Step {end_idx}] action={decision.action}, best_id={decision.best_id}, best_sim={decision.best_sim:.4f}",
            flush=True,
        )

    print(f"[Info] Total batches: {step + 1}", flush=True)

    boundary_batch_positions = [int(np.ceil(x / args.batch_size)) for x in boundary_example_idx[:-1]]
    visualize(
        decisions=decisions,
        boundary_batch_positions=boundary_batch_positions,
        task_names=task_names,
        total_batches=step + 1,
        out_svg=out_dir / "ls_task_boundary_triggers.svg",
    )

    summary = {
        "task_names": task_names,
        "decision_count": len(decisions),
        "actions": {
            "NEW": sum(1 for d in decisions if d["action"] == "NEW"),
            "EXPAND": sum(1 for d in decisions if d["action"] == "EXPAND"),
            "REUSE": sum(1 for d in decisions if d["action"] == "REUSE"),
            "WAIT": sum(1 for d in decisions if d["action"] == "WAIT"),
        },
        "boundary_batch_positions": boundary_batch_positions,
        "config": vars(args),
    }
    with (out_dir / "ls_task_boundary_decisions.json").open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "decisions": decisions}, f, ensure_ascii=False, indent=2)

    print("[Done] Boundary detection completed.", flush=True)
    print(f"[Done] Figure: {out_dir / 'ls_task_boundary_triggers.svg'}", flush=True)
    print(f"[Done] Decisions: {out_dir / 'ls_task_boundary_decisions.json'}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="t5-base")
    p.add_argument("--order_id", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samples_per_task", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--detect_rank", type=int, default=4)
    p.add_argument("--max_source_len", type=int, default=256)
    p.add_argument("--max_target_len", type=int, default=32)
    p.add_argument("--t_low", type=float, default=0.40)
    p.add_argument("--t_high", type=float, default=0.65)
    p.add_argument("--expand_cooldown_steps", type=int, default=0)
    p.add_argument("--new_cooldown_steps", type=int, default=0)
    p.add_argument("--layer_selection", type=str, default="all")
    p.add_argument("--detect_layer_selection", type=str, default=None)
    p.add_argument("--target_modules", type=str, default="q,k,v,o,wi,wo")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--cl_root", type=str, default=None, help="Path to processed CL root (default: <repo>/CL).")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="outputs/task_boundary_detect")
    p.add_argument("--use_bfloat16", action="store_true")
    p.add_argument("--offline", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
