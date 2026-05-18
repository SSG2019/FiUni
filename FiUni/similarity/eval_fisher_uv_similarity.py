import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq

from fiunilib.cl_benchmark_data import prepare_ls_tasks_from_cl
from fiunilib.fisher import compute_ag, get_uv
from fiunilib.similarity import TaskUV, build_matrix, uv_similarity


def _sanitize_modules_for_name(target_modules: str) -> str:
    parts = [p.strip().lower() for p in target_modules.split(",") if p.strip()]
    return "-".join(parts) if parts else "none"


def _sanitize_layer_for_name(layer_selection: str) -> str:
    s = layer_selection.strip().lower()
    if not s:
        return "none"
    s = s.replace(" ", "")
    s = s.replace(",", "-")
    s = s.replace("[", "")
    s = s.replace("]", "")
    s = s.replace(".", "_")
    return s


def build_output_dir(out_root: str, few_shot: int, rank: int, target_modules: str, layer_selection: str) -> Path:
    name = (
        f"fisher_uv_shot{few_shot}_rank{rank}"
        f"_target{_sanitize_modules_for_name(target_modules)}"
        f"_layer{_sanitize_layer_for_name(layer_selection)}"
    )
    return Path(out_root) / name


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    row_sum = mat.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    return mat / row_sum


def save_heatmap(matrix: np.ndarray, task_names: List[str], out_path: Path, title: str):
    plt.figure(figsize=(14, 12))
    plt.imshow(matrix, cmap="viridis", aspect="auto")
    plt.colorbar(label="Similarity")
    # Make task names easier to read (often long strings).
    plt.xticks(
        ticks=np.arange(len(task_names)),
        labels=task_names,
        rotation=45,
        ha="right",
        fontsize=15,
    )
    plt.yticks(
        ticks=np.arange(len(task_names)),
        labels=task_names,
        fontsize=15,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, format="svg")
    plt.close()


def build_seq2seq_dataset(hf_dataset, tokenizer, max_source_len: int, max_target_len: int):
    def _map_fn(ex):
        model_inputs = tokenizer(ex["prompt"], truncation=True, max_length=max_source_len)
        labels = tokenizer(text_target=ex["answer"], truncation=True, max_length=max_target_len)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return hf_dataset.map(_map_fn, remove_columns=hf_dataset.column_names)


def sample_two_fewshot_groups(dataset, few_shot: int, seed: int):
    n = len(dataset)
    k = min(few_shot, n)
    if k == 0:
        return dataset.select([]), dataset.select([])

    all_indices = list(range(n))
    rng_a = random.Random(seed)
    idx_a = sorted(rng_a.sample(all_indices, k))
    remain = sorted(list(set(all_indices) - set(idx_a)))

    if len(remain) >= k:
        rng_b = random.Random(seed + 1)
        idx_b = sorted(rng_b.sample(remain, k))
    else:
        rng_b = random.Random(seed + 1)
        idx_b = sorted(rng_b.sample(all_indices, k))

    return dataset.select(idx_a), dataset.select(idx_b)


def parse_layer_selection(layer_selection: str) -> Optional[Dict[str, Set[int]]]:
    s = layer_selection.strip().lower()
    if s == "all":
        return None
    raw = layer_selection.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    if not raw.strip():
        return set()
    out: Dict[str, Set[int]] = {"*": set(), "encoder": set(), "decoder": set()}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        p = part.lower()
        # Accept plain indices ("11"), plus scoped forms ("decoder.block.11", "encoder.block.3").
        m = re.search(r"(\d+)(?!.*\d)", p)
        if not m:
            raise ValueError(
                f"Invalid layer token '{part}'. Use 'all', integer ids like '11', "
                "lists like '0,1,2', or path-like tokens such as 'decoder.block.11'."
            )
        idx = int(m.group(1))
        if "decoder" in p:
            out["decoder"].add(idx)
        elif "encoder" in p:
            out["encoder"].add(idx)
        else:
            out["*"].add(idx)
    return out


def parse_target_modules(target_modules: str) -> Tuple[str, ...]:
    mapping = {
        "q": ".q",
        "k": ".k",
        "v": ".v",
        "o": ".o",
        "wi": ".wi",
        "wo": ".wo",
    }
    mods = []
    for token in target_modules.split(","):
        key = token.strip().lower()
        if not key:
            continue
        if key not in mapping:
            raise ValueError(f"Unsupported module token '{key}'. Allowed: {list(mapping.keys())}")
        mods.append(mapping[key])
    if not mods:
        raise ValueError("target_modules cannot be empty.")
    return tuple(mods)


def select_t5_linear_modules(model, layer_selection: Optional[Dict[str, Set[int]]], module_keywords: Tuple[str, ...]) -> List[str]:
    selected = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if not any(k in name for k in module_keywords):
            continue

        if layer_selection is not None:
            m = re.search(r"(?:^|\.)(encoder|decoder)\.block\.(\d+)\.", name)
            if not m:
                continue
            scope = m.group(1)
            layer_id = int(m.group(2))
            allow_any = layer_id in layer_selection["*"]
            allow_scoped = layer_id in layer_selection[scope]
            if not (allow_any or allow_scoped):
                continue

        selected.append(name)
    return sorted(selected)


def loss_fn(outputs, batch):
    return outputs.loss


def run(args):
    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = build_output_dir(
        out_root=args.out_root,
        few_shot=args.few_shot,
        rank=args.rank,
        target_modules=args.target_modules,
        layer_selection=args.layer_selection,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Info] Device: {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("t5-base", local_files_only=args.offline)
    load_kwargs = {}
    if args.use_bfloat16:
        load_kwargs["torch_dtype"] = torch.bfloat16
        print("[Info] BF16 model load is enabled.", flush=True)
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-base", local_files_only=args.offline, **load_kwargs)
    model.to(device)

    layer_selection = parse_layer_selection(args.layer_selection)
    module_keywords = parse_target_modules(args.target_modules)
    print(f"[Info] layer_selection={args.layer_selection}, target_modules={args.target_modules}", flush=True)
    print(f"[Info] Output dir: {out_dir}", flush=True)

    ls_tasks = prepare_ls_tasks_from_cl(
        cl_root=args.cl_root,
        order_id=args.order_id,
        seed=args.seed,
    )
    task_names = list(ls_tasks.keys())
    print(f"[Info] Tasks ({len(task_names)}): {task_names}", flush=True)

    selected_layers = select_t5_linear_modules(model, layer_selection, module_keywords)
    if not selected_layers:
        raise RuntimeError("No layers selected. Check layer_selection and target_modules.")
    print(f"[Info] Selected linear layers: {len(selected_layers)}", flush=True)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, return_tensors="pt")
    task_uv_list: List[TaskUV] = []
    task_self_sims: List[float] = []

    for task_idx, (task_name, task_data) in enumerate(ls_tasks.items()):
        print(f"\n[Task] {task_name}", flush=True)
        fewshot_a, fewshot_b = sample_two_fewshot_groups(task_data.train, args.few_shot, args.seed + 100 * task_idx)

        tokenized_a = build_seq2seq_dataset(fewshot_a, tokenizer, args.max_source_len, args.max_target_len)
        tokenized_b = build_seq2seq_dataset(fewshot_b, tokenizer, args.max_source_len, args.max_target_len)

        loader_a = DataLoader(tokenized_a, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
        loader_b = DataLoader(tokenized_b, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

        stats_a = compute_ag(
            model=model,
            layer_refs=selected_layers,
            dataloader=loader_a,
            loss_fn=loss_fn,
            use_autocast=args.use_bfloat16,
            autocast_dtype=torch.bfloat16 if args.use_bfloat16 else None,
            show_progress=args.show_progress,
            progress_desc=f"{task_name} [A]",
        )
        uv_a = get_uv(stats_a, rank=args.rank)
        task_uv_list.append(TaskUV(name=task_name, uv=uv_a))

        stats_b = compute_ag(
            model=model,
            layer_refs=selected_layers,
            dataloader=loader_b,
            loss_fn=loss_fn,
            use_autocast=args.use_bfloat16,
            autocast_dtype=torch.bfloat16 if args.use_bfloat16 else None,
            show_progress=args.show_progress,
            progress_desc=f"{task_name} [B]",
        )
        uv_b = get_uv(stats_b, rank=args.rank)
        self_sim = uv_similarity(TaskUV(name=task_name, uv=uv_a), TaskUV(name=task_name, uv=uv_b))
        task_self_sims.append(self_sim)
        print(f"[Task] {task_name}: self-sim={self_sim:.4f}", flush=True)

    sim = build_matrix(task_uv_list)
    confusion_like = normalize_rows(sim.copy())
    self_sim_diag = np.diag(task_self_sims).astype(np.float64)

    np.save(out_dir / "fisher_uv_similarity_matrix.npy", sim)
    np.save(out_dir / "fisher_uv_confusion_like_matrix.npy", confusion_like)
    np.save(out_dir / "fisher_uv_within_task_self_similarity_diag.npy", self_sim_diag)

    with (out_dir / "fisher_uv_similarity_table.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "task_names": task_names,
                "similarity_matrix": sim.tolist(),
                "confusion_like_matrix": confusion_like.tolist(),
                "within_task_self_similarity": task_self_sims,
                "within_task_self_similarity_diag_matrix": self_sim_diag.tolist(),
                "layer_selection": args.layer_selection,
                "target_modules": args.target_modules,
                "selected_layers_count": len(selected_layers),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    save_heatmap(sim, task_names, out_dir / "fisher_uv_similarity_heatmap.svg", "Fisher UV Similarity Matrix")
    save_heatmap(confusion_like, task_names, out_dir / "fisher_uv_confusion_like_heatmap.svg", "Row-normalized Fisher UV Matrix")
    save_heatmap(
        self_sim_diag,
        task_names,
        out_dir / "fisher_uv_within_task_self_similarity_diag_heatmap.svg",
        "Within-task Self Similarity Diagonal Matrix",
    )
    print("[Done] Similarity evaluation finished.", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--order_id", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--few_shot", type=int, default=32)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_source_len", type=int, default=256)
    p.add_argument("--max_target_len", type=int, default=32)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--cl_root", type=str, default=None, help="Path to processed CL root (default: <repo>/CL).")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out_root", type=str, default="outputs")
    p.add_argument("--use_bfloat16", action="store_true")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--show_progress", action="store_true")
    p.add_argument(
        "--layer_selection",
        type=str,
        default="all",
        help="Layer indices for T5 blocks: 'all' or '[0,1,2]' or '0,1,2'.",
    )
    p.add_argument(
        "--target_modules",
        type=str,
        default="q,k,v,o,wi,wo",
        help="Comma-separated module groups from {q,k,v,o,wi,wo}.",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
