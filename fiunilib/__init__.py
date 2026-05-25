"""
FiUniLib: Fisher-based utilities for continual learning.
"""

from .fisher import (
    compute_ag,
    compute_ag_batch_raw,
    compute_uv_from_ag,
    get_uv,
    merge_raw_window_to_ag,
)
from .layers import FiLoRALinear, merge_filora_layers, replace_one_with_filora
from .similarity import TaskUV, build_matrix, subspace_similarity, uv_similarity
from .task_boundary import (
    Decision,
    DecisionEngineLite,
    parse_layer_selection,
    parse_target_modules,
    select_t5_linear_modules,
)
from .train_strategy import LSTrainOrchestrator, SubspaceState

__all__ = [
    "FiLoRALinear",
    "TaskUV",
    "build_matrix",
    "compute_ag",
    "compute_ag_batch_raw",
    "compute_uv_from_ag",
    "Decision",
    "DecisionEngineLite",
    "get_uv",
    "merge_raw_window_to_ag",
    "merge_filora_layers",
    "LSTrainOrchestrator",
    "parse_layer_selection",
    "parse_target_modules",
    "replace_one_with_filora",
    "select_t5_linear_modules",
    "SubspaceState",
    "subspace_similarity",
    "uv_similarity",
]
