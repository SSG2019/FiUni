import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def load_table(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_task_separation(task_names: List[str], sim_matrix: np.ndarray, self_sims: np.ndarray, out_svg: Path, title: str):
    n = len(task_names)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(6, 6))

    rng = np.random.default_rng(42)
    jitter_scale = 0.14

    # Plot "other-task" similarities (blue), one task contributes n-1 points.
    for i in range(n):
        others = [sim_matrix[i, j] for j in range(n) if j != i]
        x_jitter = x[i] + rng.uniform(-jitter_scale, jitter_scale, size=len(others))
        ax.scatter(
            x_jitter,
            others,
            s=20,
            alpha=0.45,
            color="#2563eb",
            edgecolors="none",
            label="task-to-other similarity" if i == 0 else None,
            zorder=2,
        )

    # Plot within-task similarity from different sampling (red), one point per task.
    ax.scatter(
        x,
        self_sims,
        s=58,
        color="#dc2626",
        edgecolors="white",
        linewidths=0.7,
        label="within-task self similarity",
        zorder=3,
    )

    ax.set_xticks(x, task_names, rotation=35, ha="right", fontsize=10)
    ax.set_ylim(0.0, 1.02)
    # ax.set_ylabel("Similarity", fontsize=7)
    # ax.set_xlabel("Task", fontsize=7)
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(loc="lower right", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_svg, format="svg")
    plt.close(fig)


def run(args):
    root = Path(args.results_root)
    if not root.exists():
        raise FileNotFoundError(f"Results root does not exist: {root}")

    json_files = sorted(root.rglob("fisher_uv_similarity_table.json"))
    if not json_files:
        print(f"[Info] No similarity table found under: {root}", flush=True)
        return

    print(f"[Info] Found {len(json_files)} result folders.", flush=True)
    for idx, json_path in enumerate(json_files, start=1):
        payload = load_table(json_path)
        task_names = payload["task_names"]
        sim_matrix = np.asarray(payload["similarity_matrix"], dtype=np.float64)
        self_sims = np.asarray(payload["within_task_self_similarity"], dtype=np.float64)

        out_svg = json_path.parent / "fisher_uv_task_separation_scatter.svg"
        title = f"Task Separation Scatter ({json_path.parent.name})"
        plot_task_separation(task_names, sim_matrix, self_sims, out_svg, title)
        print(f"[{idx}/{len(json_files)}] Saved: {out_svg}", flush=True)

    print("[Done] Task-separation plots generated for all result folders.", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", type=str, default="outputs/similarity")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())

