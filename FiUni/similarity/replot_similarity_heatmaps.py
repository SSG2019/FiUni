import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


MATRIX_FILE = "fisher_uv_similarity_matrix.npy"
CONFUSION_FILE = "fisher_uv_confusion_like_matrix.npy"
SELF_DIAG_FILE = "fisher_uv_within_task_self_similarity_diag.npy"
TABLE_FILE = "fisher_uv_similarity_table.json"

SIM_HEATMAP = "fisher_uv_similarity_heatmap.svg"
CONF_HEATMAP = "fisher_uv_confusion_like_heatmap.svg"
SELF_HEATMAP = "fisher_uv_within_task_self_similarity_diag_heatmap.svg"


def _load_task_names(folder: Path, n: int) -> List[str]:
    table_path = folder / TABLE_FILE
    if table_path.exists():
        try:
            with table_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            names = payload.get("task_names", None)
            if isinstance(names, list) and len(names) == n:
                return [str(x) for x in names]
        except Exception:
            pass
    return [f"task_{i}" for i in range(n)]


def _save_heatmap(
    matrix: np.ndarray,
    task_names: List[str],
    out_path: Path,
    title: str,
    tick_fontsize: int,
    title_fontsize: int,
    cbar_fontsize: int,
):
    plt.figure(figsize=(14, 12))
    plt.imshow(matrix, cmap="viridis", aspect="auto")
    cbar = plt.colorbar(label="Similarity")
    cbar.ax.yaxis.label.set_size(cbar_fontsize)
    cbar.ax.tick_params(labelsize=cbar_fontsize)
    plt.xticks(
        ticks=np.arange(len(task_names)),
        labels=task_names,
        rotation=45,
        ha="right",
        fontsize=tick_fontsize,
    )
    plt.yticks(
        ticks=np.arange(len(task_names)),
        labels=task_names,
        fontsize=tick_fontsize,
    )
    plt.title(title, fontsize=title_fontsize)
    plt.tight_layout()
    plt.savefig(out_path, format="svg")
    plt.close()


def _is_result_dir(folder: Path) -> bool:
    return (
        (folder / MATRIX_FILE).exists()
        and (folder / CONFUSION_FILE).exists()
        and (folder / SELF_DIAG_FILE).exists()
    )


def _collect_result_dirs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    out = []
    for p in sorted(root.rglob("*")):
        if p.is_dir() and _is_result_dir(p):
            out.append(p)
    return out


def _resolve_similarity_root(cli_root: Optional[str]) -> Path:
    if cli_root and cli_root.strip():
        return Path(cli_root).expanduser().resolve()
    # Prefer "output/similarity" from user requirement; fallback to existing "outputs/similarity".
    root_a = Path("output") / "similarity"
    root_b = Path("outputs") / "similarity"
    if root_a.exists():
        return root_a.resolve()
    return root_b.resolve()


def replot_one(folder: Path, tick_fontsize: int, title_fontsize: int, cbar_fontsize: int) -> Tuple[bool, str]:
    try:
        sim = np.load(folder / MATRIX_FILE)
        conf = np.load(folder / CONFUSION_FILE)
        self_diag = np.load(folder / SELF_DIAG_FILE)
    except Exception as e:
        return False, f"load failed: {e}"

    if sim.ndim != 2:
        return False, f"{MATRIX_FILE} is not 2D"
    n = sim.shape[0]
    task_names = _load_task_names(folder, n)

    _save_heatmap(
        sim,
        task_names,
        folder / SIM_HEATMAP,
        "Fisher UV Similarity Matrix",
        tick_fontsize=tick_fontsize,
        title_fontsize=title_fontsize,
        cbar_fontsize=cbar_fontsize,
    )
    _save_heatmap(
        conf,
        task_names,
        folder / CONF_HEATMAP,
        "Row-normalized Fisher UV Matrix",
        tick_fontsize=tick_fontsize,
        title_fontsize=title_fontsize,
        cbar_fontsize=cbar_fontsize,
    )
    _save_heatmap(
        self_diag,
        task_names,
        folder / SELF_HEATMAP,
        "Within-task Self Similarity Diagonal Matrix",
        tick_fontsize=tick_fontsize,
        title_fontsize=title_fontsize,
        cbar_fontsize=cbar_fontsize,
    )
    return True, "ok"


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Replot all Fisher-UV heatmaps under a similarity root with larger fonts. "
            "Overwrites existing SVGs in each result folder."
        )
    )
    p.add_argument(
        "--sim_root",
        type=str,
        default="",
        help="Root folder containing similarity result subfolders (default: output/similarity or outputs/similarity).",
    )
    p.add_argument("--tick_fontsize", type=int, default=27)
    p.add_argument("--title_fontsize", type=int, default=27)
    p.add_argument("--cbar_fontsize", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    sim_root = _resolve_similarity_root(args.sim_root)
    print(f"[Info] similarity root: {sim_root}", flush=True)
    result_dirs = _collect_result_dirs(sim_root)
    if not result_dirs:
        print("[Warn] no result folders found.", flush=True)
        return

    print(f"[Info] found {len(result_dirs)} result folders", flush=True)
    ok_count = 0
    for folder in result_dirs:
        ok, msg = replot_one(
            folder,
            tick_fontsize=args.tick_fontsize,
            title_fontsize=args.title_fontsize,
            cbar_fontsize=args.cbar_fontsize,
        )
        if ok:
            ok_count += 1
            print(f"[Done] {folder}", flush=True)
        else:
            print(f"[Skip] {folder} ({msg})", flush=True)
    print(f"[Summary] success={ok_count}/{len(result_dirs)}", flush=True)


if __name__ == "__main__":
    main()
