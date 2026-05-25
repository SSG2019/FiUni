from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Set, Tuple

import torch

from .similarity import TaskUV, uv_similarity


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


def parse_target_modules_llama(target_modules: str) -> Tuple[str, ...]:
    mapping = {
        "q": ".q_proj",
        "k": ".k_proj",
        "v": ".v_proj",
        "o": ".o_proj",
        "up": ".up_proj",
        "gate": ".gate_proj",
        "down": ".down_proj",
    }
    mods = []
    for token in target_modules.split(","):
        key = token.strip().lower()
        if not key:
            continue
        if key not in mapping:
            raise ValueError(f"Unsupported LLaMA module token '{key}'. Allowed: {list(mapping.keys())}")
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


def select_llama_linear_modules(
    model,
    layer_selection: Optional[Dict[str, Set[int]]],
    module_keywords: Tuple[str, ...],
) -> List[str]:
    selected = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if not any(k in name for k in module_keywords):
            continue

        if layer_selection is not None:
            m = re.search(r"(?:^|\.)(?:model\.)?layers\.(\d+)\.", name)
            if not m:
                continue
            layer_id = int(m.group(1))
            allow_any = layer_id in layer_selection.get("*", set())
            # Be permissive: parse_layer_selection may have scoped entries from old T5-style args.
            allow_scoped = (
                layer_id in layer_selection.get("encoder", set())
                or layer_id in layer_selection.get("decoder", set())
            )
            if not (allow_any or allow_scoped):
                continue
        selected.append(name)
    return sorted(selected)


@dataclass
class Decision:
    step: int
    action: str  # NEW | REUSE | EXPAND | WAIT
    best_id: Optional[str]
    best_sim: float
    created_id: Optional[str] = None


class DecisionEngineLite:
    """
    Two-window decision engine:
      - REUSE if current best similarity > t_high
      - NEW if current < t_low and previous < t_high
      - EXPAND if both current and previous are in [t_low, t_high)
      - otherwise REUSE
    """

    def __init__(
        self,
        t_low: float = 0.40,
        t_high: float = 0.60,
        vote_windows: int = 3,  # kept for backward compatibility
        min_agree: int = 2,  # kept for backward compatibility
        expand_cooldown_steps: int = 0,
        new_cooldown_steps: int = 0,
    ):
        self.t_low = t_low
        self.t_high = t_high
        self.vote_windows = vote_windows
        self.min_agree = min_agree
        self.expand_cooldown_steps = max(0, int(expand_cooldown_steps))
        self.new_cooldown_steps = max(0, int(new_cooldown_steps))

        self.subspaces: Dict[str, TaskUV] = {}
        self.next_id = 0
        self.prev_best_sim: Optional[float] = None
        self.prev_best_id: Optional[str] = None
        self.current_id: Optional[str] = None
        self.last_expand_step: Optional[int] = None
        self.last_new_step: Optional[int] = None

    def _new_subspace_id(self) -> str:
        sid = f"sp_{self.next_id}"
        self.next_id += 1
        return sid

    def _match(self, window_uv: TaskUV) -> Tuple[Optional[str], float]:
        if not self.subspaces:
            return None, 0.0
        best_id = None
        best_sim = -1.0
        for sid, suv in self.subspaces.items():
            sim = uv_similarity(window_uv, suv)
            if sim > best_sim:
                best_sim = sim
                best_id = sid
        return best_id, float(best_sim)

    def update(self, step: int, window_uv: TaskUV) -> Decision:
        best_id, best_sim = self._match(window_uv)

        # Bootstrap: first subspace always created from the first window.
        if not self.subspaces:
            sid = self._new_subspace_id()
            self.subspaces[sid] = window_uv
            self.prev_best_sim = best_sim
            self.prev_best_id = sid
            self.current_id = sid
            self.last_new_step = step
            return Decision(step=step, action="NEW", best_id=None, best_sim=0.0, created_id=sid)

        prev_sim = self.prev_best_sim
        if prev_sim is None:
            prev_sim = best_sim
        prev_id = self.prev_best_id

        if best_sim > self.t_high:
            action = "REUSE"
            created_id = None
        elif best_sim < self.t_low and prev_sim < self.t_low:
            in_new_cooldown = (
                self.last_new_step is not None
                and (step - self.last_new_step) <= self.new_cooldown_steps
            )
            if in_new_cooldown:
                action = "REUSE"
                created_id = None
            else:
                sid = self._new_subspace_id()
                self.subspaces[sid] = window_uv
                action = "NEW"
                created_id = sid
        elif (
            self.t_low <= best_sim < self.t_high
            and self.t_low <= prev_sim < self.t_high
            and best_id is not None
            and best_id == prev_id
        ):
            in_cooldown = (
                self.last_expand_step is not None
                and (step - self.last_expand_step) <= self.expand_cooldown_steps
            )
            if in_cooldown:
                action = "REUSE"
                created_id = None
            else:
                action = "EXPAND"
                created_id = None
        else:
            action = "REUSE"
            created_id = None

        self.prev_best_sim = best_sim
        if created_id is not None:
            # NEW creates a subspace from the current window itself.
            # Keep previous-state references in the same coordinate system
            # for the next step: (new subspace id, self-similarity baseline).
            self.prev_best_sim = 1.0
            self.prev_best_id = created_id
            self.current_id = created_id
        else:
            self.prev_best_id = best_id
            if best_id is not None:
                self.current_id = best_id
        if action == "EXPAND":
            self.last_expand_step = step
        if action == "NEW":
            self.last_new_step = step
        return Decision(step=step, action=action, best_id=best_id, best_sim=best_sim, created_id=created_id)
