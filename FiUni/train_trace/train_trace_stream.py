import argparse
import json
import os
import re
import string
from collections import Counter
from contextlib import nullcontext
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq

try:
    import evaluate
except Exception:  # pragma: no cover
    evaluate = None

try:
    from easse.sari import corpus_sari as easse_corpus_sari
except Exception:  # pragma: no cover
    easse_corpus_sari = None

try:
    from rapidfuzz.distance import Levenshtein as RFLevenshtein
except Exception:  # pragma: no cover
    RFLevenshtein = None

from fiunilib.cl_benchmark_data import prepare_trace_tasks_from_cl
from fiunilib.repro import set_global_seed
from fiunilib.train_strategy import (
    LSTrainOrchestrator,
    build_ls_stream,
    build_seq2seq_dataset,
)


TRACE_TASK_METRICS = {
    "scienceqa": "accuracy",
    "fomc": "accuracy",
    "c-stance": "accuracy",
    "numglue-cm": "accuracy",
    "numglue-ds": "accuracy",
    "meetingbank": "rouge_l",
    "20minuten": "sari",
    "py150": "edit_similarity",
}

_ROUGE_METRIC = None
if evaluate is not None:
    try:
        _ROUGE_METRIC = evaluate.load("rouge")
    except Exception:  # pragma: no cover
        _ROUGE_METRIC = None


def visualize(decisions, losses, boundary_batch_positions, task_names: list, total_batches: int, out_svg: Path):
    x = [d["step"] for d in decisions]
    x_new = [d["step"] for d in decisions if d["action"] == "NEW"]
    x_expand = [d["step"] for d in decisions if d["action"] == "EXPAND"]
    x_reuse = [d["step"] for d in decisions if d["action"] == "REUSE"]
    y_loss = [np.nan if v is None else v for v in losses]

    x_plot, y_plot = [], []
    prev_step = None
    for step_i, loss_i in zip(x, y_loss):
        if prev_step is not None and step_i - prev_step > 1:
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
    return root / f"train_trace_{name}_order{order_id}_{stamp}"


def _parse_int_list(raw: str, expected_len: int, name: str) -> List[int]:
    txt = str(raw).replace("，", ",")
    vals = [v.strip() for v in txt.split(",") if v.strip()]
    if len(vals) != expected_len:
        raise ValueError(f"{name} must provide {expected_len} integers, got {len(vals)}: {raw}")
    out = [int(v) for v in vals]
    if any(v <= 0 for v in out):
        raise ValueError(f"{name} must contain positive integers, got: {out}")
    return out


def infer_task_subspace_ids(decisions: list, n_tasks: int, boundary_batch_positions: list, total_batches: int) -> list:
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
        sids.append(Counter(cands).most_common(1)[0][0] if cands else None)
    return sids


def _decode_label_batch(tokenizer, labels: torch.Tensor) -> list:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    ids = labels.clone()
    ids[ids == -100] = pad_id
    return tokenizer.batch_decode(ids, skip_special_tokens=True)


def _normalize_for_exact_match(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize_for_metric(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", str(text).strip().lower())
    if not text:
        return []
    return text.split(" ")


def _lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            cur = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = cur
    return dp[n]


def _rouge_l_f1(ref: str, pred: str) -> float:
    ref_toks = _tokenize_for_metric(ref)
    pred_toks = _tokenize_for_metric(pred)
    if not ref_toks and not pred_toks:
        return 1.0
    if not ref_toks or not pred_toks:
        return 0.0
    lcs = _lcs_len(ref_toks, pred_toks)
    p = lcs / max(1, len(pred_toks))
    r = lcs / max(1, len(ref_toks))
    return 0.0 if (p + r) == 0 else (2.0 * p * r / (p + r))


def _ngrams(tokens: List[str], n: int) -> set:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _safe_f1(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0 else (2.0 * p * r / (p + r))


def _sari_sentence(source: str, prediction: str, references: List[str]) -> float:
    src_toks = _tokenize_for_metric(source)
    pred_toks = _tokenize_for_metric(prediction)
    ref_toks_list = [_tokenize_for_metric(r) for r in references if str(r).strip()]
    if not ref_toks_list:
        ref_toks_list = [[]]

    scores = []
    for n in range(1, 5):
        src_ngr = _ngrams(src_toks, n)
        pred_ngr = _ngrams(pred_toks, n)
        ref_ngr_union = set()
        for r in ref_toks_list:
            ref_ngr_union |= _ngrams(r, n)

        keep_sys = src_ngr & pred_ngr
        keep_ref = src_ngr & ref_ngr_union
        keep_good = keep_sys & keep_ref
        p_keep = 1.0 if not keep_sys else len(keep_good) / len(keep_sys)
        r_keep = 1.0 if not keep_ref else len(keep_good) / len(keep_ref)
        keep_score = _safe_f1(p_keep, r_keep)

        del_sys = src_ngr - pred_ngr
        del_ref = src_ngr - ref_ngr_union
        del_good = del_sys & del_ref
        p_del = 1.0 if not del_sys else len(del_good) / len(del_sys)
        r_del = 1.0 if not del_ref else len(del_good) / len(del_ref)
        del_score = _safe_f1(p_del, r_del)

        add_sys = pred_ngr - src_ngr
        add_ref = ref_ngr_union - src_ngr
        add_good = add_sys & add_ref
        p_add = 1.0 if not add_sys else len(add_good) / len(add_sys)
        r_add = 1.0 if not add_ref else len(add_good) / len(add_ref)
        add_score = _safe_f1(p_add, r_add)

        scores.append((keep_score + del_score + add_score) / 3.0)

    return float(np.mean(scores))


def _score_trace_example(metric_name: str, source: str, reference: str, prediction: str) -> float:
    if metric_name == "accuracy":
        return float(_normalize_for_exact_match(reference) == _normalize_for_exact_match(prediction))
    if metric_name == "rouge_l":
        if _ROUGE_METRIC is not None:
            try:
                out = _ROUGE_METRIC.compute(predictions=[prediction], references=[reference], use_stemmer=True)
                return float(out.get("rougeL", 0.0))
            except Exception:
                pass
        return _rouge_l_f1(reference, prediction)
    if metric_name == "sari":
        if easse_corpus_sari is not None:
            try:
                # EASSE returns 0-100; convert to 0-1 for consistency with others.
                return float(easse_corpus_sari(orig_sents=[source], sys_sents=[prediction], refs_sents=[[reference]])) / 100.0
            except Exception:
                pass
        return _sari_sentence(source, prediction, [reference])
    if metric_name == "edit_similarity":
        if RFLevenshtein is not None:
            try:
                return float(RFLevenshtein.normalized_similarity(reference, prediction))
            except Exception:
                pass
        return SequenceMatcher(a=reference, b=prediction).ratio()
    raise ValueError(f"Unknown TRACE metric: {metric_name}")


def _extract_choice_letter(text: str) -> str:
    m = re.search(r"\b([A-E])\b", str(text).upper())
    return m.group(1) if m else ""


def _to_percent(score: float) -> float:
    # Unify all metrics to 0-100. Keep already-percent scores unchanged.
    return float(score * 100.0) if score <= 1.0 else float(score)


def run_test_eval(
    orchestrator,
    trace_tasks,
    task_names,
    task_subspace_ids,
    task_eval_batch_sizes,
    tokenizer,
    model,
    device,
    args,
    out_dir: Path,
    log_append,
):
    max_gen = getattr(args, "eval_max_new_tokens", None) or args.max_target_len
    results = {
        "per_task": {},
        "macro_avg_task_metric": None,
        "filora_forward": "sum_all_existing_subspaces",
    }
    per_task_scores = []

    model.eval()
    orchestrator._set_forward_subspace("_eval_all_filora")

    for ti, (name, sid) in enumerate(zip(task_names, task_subspace_ids)):
        metric_name = TRACE_TASK_METRICS.get(name, "accuracy")
        eval_bs = int(task_eval_batch_sizes[ti])
        td = trace_tasks[name]
        test_ds = build_seq2seq_dataset(td.test, tokenizer, args.max_source_len, args.max_target_len)
        collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors="pt")
        test_loader = DataLoader(test_ds, batch_size=eval_bs, shuffle=False, collate_fn=collator)

        nb_batches = 0
        n_examples = len(test_ds)
        total_score = 0.0
        n_scored = 0
        acc_correct = 0
        acc_total = 0
        example_cursor = 0
        cast_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if args.use_bfloat16 else nullcontext()
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                with cast_ctx:
                    gen_ids = model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_new_tokens=max_gen,
                        do_sample=bool(args.eval_do_sample),
                        num_beams=int(args.eval_num_beams),
                        temperature=float(args.eval_temperature),
                        top_p=float(args.eval_top_p),
                        top_k=int(args.eval_top_k),
                    )
                nb_batches += 1

                preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                bs = len(preds)
                raw_examples = [td.test[example_cursor + i] for i in range(bs)]
                example_cursor += bs
                refs = [str(ex.get("answer", "")) for ex in raw_examples]

                for ex, ref, pred in zip(raw_examples, refs, preds):
                    source = str(ex.get("prompt", ""))
                    if name == "scienceqa":
                        score = float(_extract_choice_letter(ref) == _extract_choice_letter(pred))
                    else:
                        score = _score_trace_example(metric_name, source, ref, pred)
                    total_score += score
                    n_scored += 1
                    if metric_name == "accuracy":
                        acc_total += 1
                        acc_correct += int(score >= 0.5)

        metric_score = _to_percent((total_score / n_scored) if n_scored else float("nan"))
        per_task_scores.append(metric_score)
        results["per_task"][name] = {
            "metric": metric_name,
            "metric_score": metric_score,
            "train_window_majority_subspace_id": sid,
            "num_batches": nb_batches,
            "num_examples": n_examples,
        }
        if metric_name == "accuracy":
            results["per_task"][name]["acc_correct"] = acc_correct
            results["per_task"][name]["acc_total"] = acc_total

        line = (
            f"[Eval] task={name} train_win_sid={sid} metric={metric_name} "
            f"score={metric_score:.4f} batches={nb_batches} (FiLoRA=sum_all)"
        )
        print(line, flush=True)
        log_append(line)

    macro_score = float(np.nanmean(per_task_scores)) if per_task_scores else float("nan")
    results["macro_avg_task_metric"] = macro_score
    print(f"[Eval] macro_avg_task_metric={macro_score:.4f}", flush=True)
    log_append(f"[Eval] macro_avg_task_metric={macro_score:.4f}")

    out_path = out_dir / "trace_test_eval.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    done = f"[Done] Test eval: {out_path}"
    print(done, flush=True)
    log_append(done)


def merge_filora_into_base(orchestrator, log_append):
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
    out_dir = Path(args.out_dir.strip()) if (args.out_dir or "").strip() else _default_out_dir(args.model_name, args.order_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir = str(out_dir.resolve())

    log_path = out_dir / "trace_run_log.txt"

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

    trace_tasks = prepare_trace_tasks_from_cl(
        trace_root=args.trace_root,
        order_id=args.order_id,
        seed=args.seed,
    )
    task_names = list(trace_tasks.keys())
    n_tasks = len(task_names)
    task_epochs = _parse_int_list(args.task_epochs, n_tasks, "task_epochs")
    task_batch_sizes = _parse_int_list(args.task_batch_sizes, n_tasks, "task_batch_sizes")
    task_grad_accs = _parse_int_list(args.task_gradient_accumulation_steps, n_tasks, "task_gradient_accumulation_steps")
    if str(getattr(args, "task_eval_batch_sizes", "")).strip():
        task_eval_batch_sizes = _parse_int_list(args.task_eval_batch_sizes, n_tasks, "task_eval_batch_sizes")
    elif getattr(args, "eval_batch_size", None):
        task_eval_batch_sizes = [int(args.eval_batch_size)] * n_tasks
    else:
        task_eval_batch_sizes = list(task_batch_sizes)
    print(f"[Info] TRACE tasks (order{args.order_id}): {task_names}", flush=True)
    print(
        "[Info] Per-task schedule: "
        + ", ".join(
            f"{task_names[i]}(ep={task_epochs[i]},bs={task_batch_sizes[i]},ga={task_grad_accs[i]},ebs={task_eval_batch_sizes[i]})"
            for i in range(n_tasks)
        ),
        flush=True,
    )

    orchestrator = LSTrainOrchestrator(model=model, tokenizer=tokenizer, args=args)
    decisions: List[dict] = []
    losses: List[float] = []
    boundary_batch_positions: List[int] = []
    global_step_offset = 0
    total_examples = 0

    for i, name in enumerate(task_names):
        td = trace_tasks[name]
        ep = task_epochs[i]
        bs = task_batch_sizes[i]
        ga = task_grad_accs[i]
        task_tokenized = build_seq2seq_dataset(td.train, tokenizer, args.max_source_len, args.max_target_len)
        collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors="pt")
        print(f"[Info] Task {i+1}/{n_tasks} {name}: train_examples={len(task_tokenized)} epochs={ep} bs={bs} ga={ga}", flush=True)
        total_examples += len(task_tokenized) * ep
        # Task-boundary reset: NEW cooldown should not leak across different tasks.
        orchestrator.engine.last_new_step = None

        for e in range(ep):
            loader = DataLoader(task_tokenized, batch_size=bs, shuffle=False, collate_fn=collator)
            task_batches = list(loader)
            orchestrator._flush_accumulated_step()
            orchestrator.grad_acc_steps = max(1, int(ga))
            orchestrator.engine.new_cooldown_steps = int(getattr(args, "new_cooldown_steps", 0) or 0) * orchestrator.grad_acc_steps
            ep_decisions, ep_losses = orchestrator.run(task_batches)
            for d in ep_decisions:
                d2 = dict(d)
                d2["step"] = int(d2["step"]) + global_step_offset
                d2["task_name"] = name
                d2["task_epoch"] = e + 1
                decisions.append(d2)
            losses.extend(ep_losses)
            global_step_offset += len(task_batches)
            print(
                f"[Info]   epoch {e+1}/{ep}: batches={len(task_batches)} cumulative_batches={global_step_offset}",
                flush=True,
            )
        boundary_batch_positions.append(global_step_offset)

    orchestrator.save_subspace_meta(out_dir)

    total_r_numel = orchestrator.total_r_parameters()
    r_line = f"[Info] Total FiLoRA R parameters (numel): {total_r_numel:,}"
    print(r_line, flush=True)
    log_append(r_line)

    total_batches = global_step_offset
    draw_boundaries = boundary_batch_positions[:-1]
    visualize(decisions, losses, draw_boundaries, task_names, total_batches, out_dir / "trace_train_timeline.svg")
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
        "boundary_batch_positions": draw_boundaries,
        "task_subspace_ids": infer_task_subspace_ids(decisions, len(task_names), draw_boundaries, total_batches),
        "task_train_epochs": task_epochs,
        "task_train_batch_sizes": task_batch_sizes,
        "task_train_gradient_accumulation_steps": task_grad_accs,
        "task_eval_batch_sizes": task_eval_batch_sizes,
        "total_stream_examples_with_repeats": int(total_examples),
        "total_r_parameters": total_r_numel,
        "config": vars(args),
    }
    with (out_dir / "trace_train_decisions.json").open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "decisions": decisions, "losses": losses}, f, ensure_ascii=False, indent=2)

    print("[Done] TRACE training completed.", flush=True)
    log_append("[Done] TRACE training completed.")
    print(f"[Done] Timeline: {out_dir / 'trace_train_timeline.svg'}", flush=True)
    print(f"[Done] Decisions: {out_dir / 'trace_train_decisions.json'}", flush=True)
    print(f"[Done] Subspaces: {out_dir / 'subspace_meta.json'}", flush=True)

    if not getattr(args, "skip_test_eval", False):
        merge_filora_into_base(orchestrator, log_append)
        run_test_eval(
            orchestrator,
            trace_tasks,
            task_names,
            summary["task_subspace_ids"],
            summary["task_eval_batch_sizes"],
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
    p.add_argument("--order_id", type=int, default=7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument(
        "--task_epochs",
        type=str,
        default="1,1,1,1,1,1,1,1",
        help="Comma-separated epochs per TRACE task (order 7), e.g. '2,2,2,3,4,5,2,3'.",
    )
    p.add_argument(
        "--task_batch_sizes",
        type=str,
        default="4,4,4,4,4,4,4,4",
        help="Comma-separated batch sizes per TRACE task.",
    )
    p.add_argument(
        "--task_gradient_accumulation_steps",
        type=str,
        default="1,1,1,1,1,1,1,1",
        help="Comma-separated gradient_accumulation_steps per TRACE task.",
    )
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--detect_rank", type=int, default=4)
    p.add_argument("--expand_delta_rank", type=int, default=2)
    p.add_argument("--expand_direct", action="store_true")
    p.add_argument("--max_source_len", type=int, default=512)
    p.add_argument("--max_target_len", type=int, default=128)
    p.add_argument("--t_low", type=float, default=0.40)
    p.add_argument("--t_high", type=float, default=0.65)
    p.add_argument("--layer_selection", type=str, default="all")
    p.add_argument("--detect_layer_selection", type=str, default="all")
    p.add_argument("--target_modules", type=str, default="q,k,v,o,wi,wo")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--filora_dropout", type=float, default=0.0)
    p.add_argument("--expand_cooldown_steps", type=int, default=0)
    p.add_argument("--new_cooldown_steps", type=int, default=0)
    p.add_argument("--filora_initial_max_optimizer_steps", type=int, default=None)
    p.add_argument("--filora_expand_extra_optimizer_steps", type=int, default=0)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument(
        "--trace_root",
        type=str,
        default="data/TRACE-benchmark/LLM-CL-Benchmark",
        help="Path to TRACE root. Fallback resolution also checks TRACE-Benchmark variants.",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="")
    p.add_argument("--eval_batch_size", type=int, default=None)
    p.add_argument(
        "--task_eval_batch_sizes",
        type=str,
        default="",
        help="Comma-separated eval batch sizes per TRACE task; empty means use task_batch_sizes.",
    )
    p.add_argument("--eval_max_new_tokens", type=int, default=None)
    p.add_argument("--eval_do_sample", action="store_true", default=True)
    p.add_argument("--no_eval_do_sample", action="store_false", dest="eval_do_sample")
    p.add_argument("--eval_num_beams", type=int, default=1)
    p.add_argument("--eval_temperature", type=float, default=0.95)
    p.add_argument("--eval_top_p", type=float, default=0.7)
    p.add_argument("--eval_top_k", type=int, default=50)
    p.add_argument("--skip_test_eval", action="store_true")
    p.add_argument("--no_progress", action="store_true")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--fisher_ag_group_size", type=int, default=1)
    p.add_argument("--use_bfloat16", action="store_true")
    p.add_argument("--offline", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
