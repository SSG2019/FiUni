import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt


DEFAULT_JSON = (
    "e:/my_paper/FU_CL/outputs/task_boundary_detect/"
    "bs32_r32_dr4_low0.45_high0.7_cd40_layerall_dlayer11_targetq-k-v-o/"
    "ls_task_boundary_decisions.json"
)
DEFAULT_INPUT_DIR = "e:/my_paper/FU_CL/outputs/task_boundary_detect/"


def visualize(decisions, boundary_batch_positions, task_names, total_batches: int, out_svg: Path):
    y_reuse = 0.2
    y_expand = 0.65
    y_new = 1.1

    x_new = [d["step"] for d in decisions if d["action"] == "NEW"]
    x_expand = [d["step"] for d in decisions if d["action"] == "EXPAND"]
    x_reuse = [d["step"] for d in decisions if d["action"] == "REUSE"]

    # Keep readability, trim whitespace instead of squeezing content.
    fig, ax = plt.subplots(figsize=(18, 5.0))

    seg_starts = [0] + boundary_batch_positions
    seg_ends = boundary_batch_positions + [total_batches]
    for i, (s, e) in enumerate(zip(seg_starts, seg_ends)):
        mid = 0.5 * (s + e)
        task_label = task_names[i] if i < len(task_names) else f"task_{i}"
        ax.text(mid, y_new + 0.20, task_label, ha="center", va="bottom", fontsize=13, color="#4a5568", rotation=20)

    if x_reuse:
        ax.scatter(x_reuse, [y_reuse] * len(x_reuse), marker="o", s=30, color="#2563eb", label="REUSE", zorder=3)
    if x_expand:
        ax.scatter(x_expand, [y_expand] * len(x_expand), marker="^", s=38, color="#ea580c", label="EXPAND", zorder=3)
    if x_new:
        ax.scatter(x_new, [y_new] * len(x_new), marker="*", s=72, color="#16a34a", label="NEW", zorder=3)

    for xb in boundary_batch_positions:
        ax.axvline(x=xb, linestyle="--", linewidth=1, alpha=0.55, color="#64748b", zorder=1)

    for d in decisions:
        if d["action"] == "EXPAND" and d["best_id"] is not None:
            ax.text(d["step"], y_expand + 0.06, d["best_id"], fontsize=12, ha="center", va="bottom", color="#ea580c", rotation=35)
        if d["action"] == "NEW" and d["created_id"] is not None:
            ax.text(d["step"], y_new + 0.06, d["created_id"], fontsize=12, ha="center", va="bottom", color="#166534", rotation=35)

    ax.set_ylim(-0.12, y_new + 0.32)
    ax.margins(x=0.01)
    ax.set_yticks([y_reuse, y_expand, y_new], ["REUSE", "EXPAND", "NEW"])
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlabel("Window End Batch Index", fontsize=13)
    ax.set_ylabel("Triggered State", fontsize=13)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_svg, format="svg", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def _redraw_one(in_json: Path, out_svg_override: Optional[str] = None):
    if not in_json.exists():
        raise FileNotFoundError(f"Input json not found: {in_json}")

    with in_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    summary = payload["summary"]
    decisions = payload["decisions"]
    task_names = summary["task_names"]
    boundary_batch_positions = summary["boundary_batch_positions"]
    total_batches = int(summary["decision_count"])

    out_svg = Path(out_svg_override) if out_svg_override else in_json.parent / "ls_task_boundary_triggers_short.svg"
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    visualize(decisions, boundary_batch_positions, task_names, total_batches, out_svg)
    print(f"[Done] Redrawn figure saved to: {out_svg}", flush=True)


def run(args):
    # Keep compatibility: when input_json is provided, process only that one.
    if args.input_json:
        _redraw_one(Path(args.input_json), args.out_svg)
        return

    root = Path(args.input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")

    json_files = sorted(root.rglob("*task_boundary_decisions.json"))
    if not json_files:
        print(f"[Warn] No decision json found under: {root}", flush=True)
        return

    print(f"[Info] Found {len(json_files)} decision files under: {root}", flush=True)
    ok = 0
    fail = 0
    for jf in json_files:
        try:
            _redraw_one(jf, None)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"[Error] Failed on {jf}: {e}", flush=True)
    print(f"[Done] Batch redraw completed. success={ok}, failed={fail}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_json", type=str, default=None, help="Single json path. If set, only redraw this file.")
    p.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR, help="Root directory for batch redraw.")
    p.add_argument("--out_svg", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())

