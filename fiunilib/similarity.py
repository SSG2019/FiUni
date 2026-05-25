from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch


@dataclass
class TaskUV:
    name: str
    uv: Dict[str, Tuple[torch.Tensor, torch.Tensor]]


def subspace_similarity(x1: torch.Tensor, x2: torch.Tensor) -> float:
    r = min(x1.shape[1], x2.shape[1])
    if r == 0:
        return 0.0
    a = x1[:, :r]
    b = x2[:, :r]
    cross = a.T @ b
    score = (cross.pow(2).sum() / r).item()
    return float(max(0.0, min(1.0, score)))


def uv_similarity(task_a: TaskUV, task_b: TaskUV) -> float:
    common_layers = sorted(set(task_a.uv.keys()) & set(task_b.uv.keys()))
    if not common_layers:
        return 0.0
    sims = []
    for layer in common_layers:
        ua, va = task_a.uv[layer]
        ub, vb = task_b.uv[layer]
        su = subspace_similarity(ua, ub)
        sv = subspace_similarity(va, vb)
        sims.append(0.5 * (su + sv))
    return float(np.mean(sims))


def build_matrix(task_uv_list: List[TaskUV]) -> np.ndarray:
    n = len(task_uv_list)
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i, j] = 1.0
            elif j < i:
                mat[i, j] = mat[j, i]
            else:
                mat[i, j] = uv_similarity(task_uv_list[i], task_uv_list[j])
    return mat
