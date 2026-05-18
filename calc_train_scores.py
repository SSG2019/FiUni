import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_RUN_GROUPS = [
    "train_ls_T5_large",
    "train_sc_T5_large",
    "train_ls_llama",
    "train_sc_llama",
    "train_trace_T5_large",
    "train_trace_llama",
]

EVAL_FILENAMES = ("ls_test_eval.json", "sc_test_eval.json", "trace_test_eval.json")
SCORE_KEYS = ("macro_avg_sequence_em_acc", "macro_avg_task_metric", "macro_avg_loss")


def _read_eval_file(seed_dir: Path) -> Optional[Tuple[Path, Dict]]:
    for name in EVAL_FILENAMES:
        p = seed_dir / name
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                return p, json.load(f)
    return None


def _pick_score(payload: Dict) -> Tuple[Optional[str], Optional[float]]:
    for k in SCORE_KEYS:
        v = payload.get(k, None)
        if isinstance(v, (int, float)):
            return k, float(v)
    return None, None


def _extract_task_weighted_em(payload: Dict) -> Optional[float]:
    """
    Return task_weighted EM only (equal-task average / macro over tasks).
    """
    task_weighted: Optional[float] = None

    per_task = payload.get("per_task", None)
    if isinstance(per_task, dict) and per_task:
        task_accs: List[float] = []
        for _, v in per_task.items():
            if not isinstance(v, dict):
                continue
            acc = v.get("sequence_exact_match_acc", None)
            if not isinstance(acc, (int, float)):
                acc = v.get("metric_score", None)
            if isinstance(acc, (int, float)):
                task_accs.append(float(acc))
        if task_accs:
            task_weighted = _safe_mean(task_accs)

    # Prefer explicit macro if available (same intent as task-weighted average).
    macro = payload.get("macro_avg_sequence_em_acc", None)
    if isinstance(macro, (int, float)):
        task_weighted = float(macro)
    else:
        macro_trace = payload.get("macro_avg_task_metric", None)
        if isinstance(macro_trace, (int, float)):
            task_weighted = float(macro_trace)

    return task_weighted


def _safe_mean(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return sum(vals) / len(vals)


def _safe_std(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    m = _safe_mean(vals)
    if m is None:
        return None
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    return var ** 0.5


def summarize_group(group_dir: Path) -> None:
    if not group_dir.exists():
        print(f"[Skip] {group_dir} (not found)")
        return

    banner = "#" * 28
    print(f"\n{banner} {group_dir.name} {banner}")
    order_dirs = sorted([d for d in group_dir.iterdir() if d.is_dir() and d.name.startswith("order")], key=lambda p: p.name)
    if not order_dirs:
        print("No order folders found.")
        return

    all_task_weighted: List[float] = []

    for order_dir in order_dirs:
        seed_dirs = sorted([d for d in order_dir.iterdir() if d.is_dir() and d.name.startswith("seed")], key=lambda p: p.name)
        order_task_weighted: List[float] = []

        for seed_dir in seed_dirs:
            loaded = _read_eval_file(seed_dir)
            if loaded is None:
                continue
            _, payload = loaded
            tw = _extract_task_weighted_em(payload)
            if tw is not None:
                order_task_weighted.append(tw)
                all_task_weighted.append(tw)

        if order_task_weighted:
            tm = _safe_mean(order_task_weighted)
            ts = _safe_std(order_task_weighted)
            print(f"{order_dir.name}: mean={tm:.6f}  std={ts:.6f}")
        else:
            print(f"{order_dir.name}: mean=nan  std=nan")

    if not all_task_weighted:
        print("overall: mean=nan  std=nan")
    else:
        tm = _safe_mean(all_task_weighted)
        ts = _safe_std(all_task_weighted)
        print(f"overall: mean={tm:.6f}  std={ts:.6f}")


def main():
    p = argparse.ArgumentParser(description="Summarize train eval scores by order and overall mean.")
    p.add_argument(
        "--train_root",
        type=str,
        default="outputs/train",
        help="Root folder containing train_* groups (default: outputs/train).",
    )
    p.add_argument(
        "--groups",
        type=str,
        default=",".join(DEFAULT_RUN_GROUPS),
        help="Comma-separated run groups under train_root.",
    )
    args = p.parse_args()

    train_root = Path(args.train_root)
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]

    print(f"Scanning: {train_root.resolve()}")
    for g in groups:
        summarize_group(train_root / g)


if __name__ == "__main__":
    main()
