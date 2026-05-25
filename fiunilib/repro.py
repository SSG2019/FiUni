"""Global RNG and backend settings so training runs follow ``--seed``."""

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Sync Python / NumPy / PyTorch RNGs and prefer deterministic cuDNN where possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Helps select deterministic CUDA matmul algorithms when available (PyTorch 1.8+).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
