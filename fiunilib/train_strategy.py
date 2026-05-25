import json
from dataclasses import dataclass
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from datasets import Dataset
from .fisher import compute_ag_batch_raw, compute_ag_window_grouped, get_UV
from .similarity import TaskUV
from .task_boundary import (
    Decision,
    DecisionEngineLite,
    parse_layer_selection,
    parse_target_modules_llama,
    parse_target_modules,
    select_llama_linear_modules,
    select_t5_linear_modules,
)

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # pragma: no cover
    _tqdm = None


@dataclass
class SubspaceState:
    sid: str
    rank: int
    # Per-subspace FiLoRA training budget in **optimizer.step()** units (after grad accumulation).
    # ``None`` means no cap (legacy / unlimited).
    max_optimizer_steps: Optional[int] = None
    optimizer_steps_done: int = 0


def build_seq2seq_dataset(hf_dataset, tokenizer, max_source_len: int, max_target_len: int):
    def _map_fn(ex):
        model_inputs = tokenizer(ex["prompt"], truncation=True, max_length=max_source_len)
        labels = tokenizer(text_target=ex["answer"], truncation=True, max_length=max_target_len)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return hf_dataset.map(_map_fn, remove_columns=hf_dataset.column_names)


def build_ls_stream(ls_tasks, samples_per_task: Optional[int] = None):
    rows = []
    task_names = list(ls_tasks.keys())
    task_boundary_example_idx = []
    cursor = 0
    for tidx, name in enumerate(task_names):
        train = ls_tasks[name].train
        n = len(train) if samples_per_task is None else min(samples_per_task, len(train))
        sub = train.select(range(n))
        for i in range(n):
            rows.append(
                {
                    "prompt": sub[i]["prompt"],
                    "answer": sub[i]["answer"],
                    "task_name": name,
                    "task_idx": tidx,
                }
            )
        cursor += n
        task_boundary_example_idx.append(cursor)
    return Dataset.from_list(rows), task_names, task_boundary_example_idx


class BankedFiLoRALinear(nn.Module):
    """
    One shared base linear + multiple subspace banks {U,V,R}.

    * ``active_sid is None`` (detection / backbone-only): forward is **base(x)** only.
    * ``active_sid`` is set (train / eval with adapters): forward is
      **base(x) + sum_s delta_s(x)** over **every** existing subspace ``s`` in this layer’s
      bank. All FiLoRA branches participate in the graph; only the subspace selected by
      ``set_trainable_sid`` receives gradients on **R** (others’ R are frozen).
    """

    def __init__(self, linear: nn.Linear, dropout: float = 0.0):
        super().__init__()
        self.base = linear
        self.active_sid: Optional[str] = None
        self.U_bank = nn.ParameterDict()
        self.V_bank = nn.ParameterDict()
        self.R_bank = nn.ParameterDict()
        self.ranks: Dict[str, int] = {}
        self.delta_dropout = nn.Dropout(dropout) if dropout and dropout > 0 else None

    def add_subspace(self, sid: str, U: torch.Tensor, V: torch.Tensor):
        if sid in self.R_bank:
            return
        dev, dt = self.base.weight.device, self.base.weight.dtype
        U = U.to(device=dev, dtype=dt).contiguous()
        V = V.to(device=dev, dtype=dt).contiguous()
        r = U.shape[1]
        sigma = 1e-5
        R = torch.randn(r, r, device=dev, dtype=dt) * sigma

        self.U_bank[sid] = nn.Parameter(U, requires_grad=False)
        self.V_bank[sid] = nn.Parameter(V, requires_grad=False)
        self.R_bank[sid] = nn.Parameter(R, requires_grad=False)
        self.ranks[sid] = int(r)

    def set_active(self, sid: Optional[str]):
        self.active_sid = sid

    def set_trainable_sid(self, sid: Optional[str]):
        for k, p in self.R_bank.items():
            p.requires_grad_(k == sid)

    def _orthogonal_increment(self, old_basis: torch.Tensor, new_basis: torch.Tensor, delta_rank: int) -> torch.Tensor:
        # Remove projection on old basis and keep the most energetic orthogonal directions.
        proj = old_basis @ (old_basis.transpose(0, 1) @ new_basis)
        residual = new_basis - proj
        # torch.linalg.qr is not implemented for BFloat16 on CUDA (geqrf_cuda)
        res_f = residual.float()
        q, _ = torch.linalg.qr(res_f, mode="reduced")
        q = q.to(dtype=residual.dtype)
        return q[:, :delta_rank].contiguous()

    def expand_subspace(
        self,
        sid: str,
        U_new: torch.Tensor,
        V_new: torch.Tensor,
        delta_rank: int,
        orthogonal: bool = True,
    ):
        if delta_rank <= 0 or sid not in self.R_bank:
            return

        dev, dt = self.base.weight.device, self.base.weight.dtype
        old_u = self.U_bank[sid].data
        old_v = self.V_bank[sid].data
        old_r = old_u.shape[1]

        u_new = U_new.to(device=dev, dtype=dt).contiguous()
        v_new = V_new.to(device=dev, dtype=dt).contiguous()
        if orthogonal:
            du = self._orthogonal_increment(old_u, u_new, delta_rank)
            dv = self._orthogonal_increment(old_v, v_new, delta_rank)
        else:
            # Concatenate leading columns of the new Fisher U/V without projecting off old basis.
            du = u_new[:, : min(delta_rank, u_new.shape[1])].contiguous()
            dv = v_new[:, : min(delta_rank, v_new.shape[1])].contiguous()

        # If orthogonal residual rank is insufficient, pad with tiny noise.
        if du.shape[1] < delta_rank:
            pad = delta_rank - du.shape[1]
            add = torch.randn(old_u.shape[0], pad, device=dev, dtype=dt)
            add = add / (add.norm(dim=0, keepdim=True) + 1e-8)
            du = torch.cat([du, add], dim=1)
        if dv.shape[1] < delta_rank:
            pad = delta_rank - dv.shape[1]
            add = torch.randn(old_v.shape[0], pad, device=dev, dtype=dt)
            add = add / (add.norm(dim=0, keepdim=True) + 1e-8)
            dv = torch.cat([dv, add], dim=1)

        new_u = torch.cat([old_u, du[:, :delta_rank]], dim=1).contiguous()
        new_v = torch.cat([old_v, dv[:, :delta_rank]], dim=1).contiguous()
        new_r = old_r + delta_rank

        old_R = self.R_bank[sid].data
        R = torch.zeros(new_r, new_r, device=dev, dtype=dt)
        R[:old_r, :old_r].copy_(old_R)
        sigma = 1e-5
        if delta_rank > 0:
            R[old_r:, old_r:] = torch.randn(delta_rank, delta_rank, device=dev, dtype=dt) * sigma

        self.U_bank[sid] = nn.Parameter(new_u, requires_grad=False)
        self.V_bank[sid] = nn.Parameter(new_v, requires_grad=False)
        self.R_bank[sid] = nn.Parameter(R, requires_grad=False)
        self.ranks[sid] = int(new_r)

    def _delta_for_sid(self, x: torch.Tensor, sid: str) -> torch.Tensor:
        U = self.U_bank[sid]
        V = self.V_bank[sid]
        R = self.R_bank[sid]
        t = torch.nn.functional.linear(x, V.t())
        t = t.matmul(R.t())
        return torch.nn.functional.linear(t, U)

    def forward(self, x: torch.Tensor):
        base_out = self.base(x)
        if self.active_sid is None or not self.R_bank:
            return base_out
        delta_sum = None
        for sid in self.R_bank.keys():
            d = self._delta_for_sid(x, sid)
            delta_sum = d if delta_sum is None else (delta_sum + d)
        if delta_sum is None:
            return base_out
        if self.delta_dropout is not None:
            delta_sum = self.delta_dropout(delta_sum)
        return base_out + delta_sum


def _get_submodule(root: nn.Module, dotted: str):
    obj = root
    for p in dotted.split("."):
        obj = getattr(obj, p)
    return obj


def _replace_submodule(root: nn.Module, dotted: str, new_module: nn.Module):
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def install_banked_filora_wrappers(model: nn.Module, layer_names: List[str], dropout: float = 0.0):
    wrappers: Dict[str, BankedFiLoRALinear] = {}
    for name in layer_names:
        target = _get_submodule(model, name)
        if isinstance(target, BankedFiLoRALinear):
            wrappers[name] = target
            continue
        if not isinstance(target, nn.Linear):
            raise TypeError(f"{name} is not nn.Linear")
        wrapped = BankedFiLoRALinear(target, dropout=dropout)
        _replace_submodule(model, name, wrapped)
        wrappers[name] = wrapped
    return wrappers


def loss_fn(outputs, batch):
    return outputs.loss


class LSTrainOrchestrator:
    """
    Detection-driven online training:
      - Detect boundary states from sliding windows on frozen base model
      - Trigger NEW / REUSE / EXPAND
      - Train active subspace (R matrices only) on each incoming batch

    Forward semantics (important):
      * Boundary / Fisher stats (`_compute_batch_stats_detect`): `active_sid=None` → **base only**
        (all FiLoRA deltas skipped). Hooks only on ``detect_layers`` to save memory.
        On NEW/EXPAND, full ``selected_layers`` AG is recomputed over the same detect window
        using ``fisher_ag_group_size`` chunking.
      * Training / eval with adapters (`_train_active_step`, test): `active_sid` set to any string
        → every wrapped layer runs **base(x) + sum_s delta_s(x)** over **all** existing
        subspaces ``s``. `active_sid` only selects which **R** is trainable via
        ``set_trainable_sid``; the forward still uses every bank’s U,V,R in the sum.
    """

    def __init__(self, model, tokenizer, args):
        self.args = args
        self.device = next(model.parameters()).device
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tokenizer = tokenizer

        train_layer_selection = parse_layer_selection(args.layer_selection)
        detect_layer_selection = parse_layer_selection(args.detect_layer_selection)
        model_family = str(getattr(args, "model_family", "t5")).strip().lower()
        if model_family in {"llama", "causal"}:
            module_keywords = parse_target_modules_llama(args.target_modules)
            selector = select_llama_linear_modules
        else:
            module_keywords = parse_target_modules(args.target_modules)
            selector = select_t5_linear_modules
        self.selected_layers = selector(self.model, train_layer_selection, module_keywords)
        self.detect_layers = selector(self.model, detect_layer_selection, module_keywords)
        if not self.selected_layers:
            raise RuntimeError("No layers selected. Check layer_selection and target_modules.")
        if not self.detect_layers:
            raise RuntimeError("No layers selected. Check detect_layer_selection and target_modules.")
        filora_dropout = float(getattr(args, "filora_dropout", 0.0) or 0.0)
        self.wrappers = install_banked_filora_wrappers(self.model, self.selected_layers, dropout=filora_dropout)
        # Fisher: detect uses one pass (see _compute_batch_stats_detect). NEW/EXPAND full-layer
        # AG splits ``layer_selection`` into ``fisher_ag_group_size`` contiguous chunks (passes).
        self.ag_group_size = max(1, int(getattr(args, "fisher_ag_group_size", 1) or 1))
        grad_accum_steps = max(1, int(getattr(args, "gradient_accumulation_steps", 1)))
        expand_cooldown_steps = int(getattr(args, "expand_cooldown_steps", 0) or 0) * grad_accum_steps
        new_cooldown_steps = int(getattr(args, "new_cooldown_steps", 0) or 0) * grad_accum_steps
        self.engine = DecisionEngineLite(
            t_low=args.t_low,
            t_high=args.t_high,
            expand_cooldown_steps=expand_cooldown_steps,
            new_cooldown_steps=new_cooldown_steps,
        )
        self.subspaces: Dict[str, SubspaceState] = {}
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.active_sid: Optional[str] = None
        self.grad_accum_steps = grad_accum_steps
        self._accum_count = 0
        self._accum_sid: Optional[str] = None
        self.rank_initial = args.rank
        self.detect_rank = args.detect_rank
        self.expand_delta = args.expand_delta_rank
        self.expand_subspace_orthogonal = not bool(getattr(args, "expand_direct", False))
        _init_cap = getattr(args, "filora_initial_max_optimizer_steps", None)
        self._budget_enabled = _init_cap is not None
        self._initial_max_optimizer_steps = int(_init_cap) if self._budget_enabled else None
        self._expand_extra_optimizer_steps = int(getattr(args, "filora_expand_extra_optimizer_steps", 0) or 0)
        if self.detect_rank > self.rank_initial:
            raise ValueError(f"detect_rank ({self.detect_rank}) must be <= rank ({self.rank_initial}).")

    def _sid_may_run_optimizer(self, sid: str) -> bool:
        if not self._budget_enabled:
            return True
        st = self.subspaces.get(sid)
        if st is None or st.max_optimizer_steps is None:
            return True
        return st.optimizer_steps_done < st.max_optimizer_steps

    def _record_optimizer_step(self, sid: Optional[str]):
        if not self._budget_enabled or sid is None:
            return
        st = self.subspaces.get(sid)
        if st is not None:
            st.optimizer_steps_done += 1

    def _create_optimizer_for_sid(self, sid: str):
        for w in self.wrappers.values():
            w.set_trainable_sid(sid)
            w.set_active(sid)
        params = []
        for w in self.wrappers.values():
            if sid in w.R_bank:
                params.append(w.R_bank[sid])
        if not params:
            raise RuntimeError(f"No trainable R parameters for sid={sid}")
        return torch.optim.AdamW(params, lr=self.args.lr, weight_decay=self.args.weight_decay)

    def _set_forward_subspace(self, sid: Optional[str]):
        for w in self.wrappers.values():
            w.set_active(sid)

    def _flush_accumulated_step(self):
        if self.optimizer is None:
            self._accum_count = 0
            self._accum_sid = None
            return
        if self._accum_count > 0:
            stepped_sid = self._accum_sid
            self.optimizer.step()
            self._record_optimizer_step(stepped_sid)
            self.optimizer.zero_grad(set_to_none=True)
        self._accum_count = 0
        self._accum_sid = None

    def _switch_active_sid(self, sid: Optional[str], force_rebind: bool = False):
        if sid == self.active_sid and not force_rebind:
            return
        # Do not mix gradients across different subspaces.
        self._flush_accumulated_step()
        self.active_sid = sid
        if sid is None:
            self.optimizer = None
            return
        self.optimizer = self._create_optimizer_for_sid(sid)
        self.optimizer.zero_grad(set_to_none=True)

    def _compute_batch_stats_detect(self, batch: dict) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        # Detection: one full forward+backward; hooks only on detect layers (group_size=1).
        # Hooks are removed when compute_ag_batch_raw returns; unrelated to fisher_ag_group_size.
        self._set_forward_subspace(None)
        return compute_ag_batch_raw(
            model=self.model,
            layer_refs=self.detect_layers,
            batch=batch,
            loss_fn=loss_fn,
            use_autocast=self.args.use_bfloat16,
            autocast_dtype=torch.bfloat16 if self.args.use_bfloat16 else None,
            group_size=1,
        )

    def _compute_window_uv_full_selected(
        self, batches: List[dict], *, uv_rank: int
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        NEW/EXPAND full-layer Fisher:
        window K batches -> per-layer-chunk A/G accumulation -> one UV decomposition per chunk.
        """
        if not batches:
            return {}
        self._set_forward_subspace(None)
        rank = max(1, int(uv_rank))
        return compute_ag_window_grouped(
            model=self.model,
            layer_refs=self.selected_layers,
            batches=batches,
            loss_fn=loss_fn,
            use_autocast=self.args.use_bfloat16,
            autocast_dtype=torch.bfloat16 if self.args.use_bfloat16 else None,
            group_size=self.ag_group_size,
            uv_rank=rank,
        )

    def _subset_stats(
        self,
        stats_union: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        layer_names: List[str],
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        return {name: stats_union[name] for name in layer_names if name in stats_union}

    def _create_subspace(self, sid: str, uv: Dict[str, Tuple[torch.Tensor, torch.Tensor]]):
        for lname, (u, v) in uv.items():
            if lname in self.wrappers:
                self.wrappers[lname].add_subspace(sid, u[:, : self.rank_initial], v[:, : self.rank_initial])
        cap: Optional[int] = self._initial_max_optimizer_steps if self._budget_enabled else None
        self.subspaces[sid] = SubspaceState(
            sid=sid,
            rank=self.rank_initial,
            max_optimizer_steps=cap,
            optimizer_steps_done=0,
        )

    def _expand_subspace(self, sid: str, uv: Dict[str, Tuple[torch.Tensor, torch.Tensor]]):
        state = self.subspaces[sid]
        for lname, (u_new, v_new) in uv.items():
            if lname in self.wrappers:
                self.wrappers[lname].expand_subspace(
                    sid, u_new, v_new, self.expand_delta, orthogonal=self.expand_subspace_orthogonal
                )
        state.rank += self.expand_delta
        if self._budget_enabled and state.max_optimizer_steps is not None:
            state.max_optimizer_steps += self._expand_extra_optimizer_steps

    def _train_active_step(self, batch: dict) -> Optional[float]:
        if self.active_sid is None:
            return None
        if not self._sid_may_run_optimizer(self.active_sid):
            return None
        self.model.train()
        self._set_forward_subspace(self.active_sid)

        batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        inputs = {k: v for k, v in batch.items() if torch.is_tensor(v)}

        if self.optimizer is None:
            self.optimizer = self._create_optimizer_for_sid(self.active_sid)
            self.optimizer.zero_grad(set_to_none=True)
        if self._accum_count == 0:
            self.optimizer.zero_grad(set_to_none=True)
        if self.args.use_bfloat16:
            cast_ctx = torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
        else:
            cast_ctx = nullcontext()

        with cast_ctx:
            outputs = self.model(**inputs)
            loss = loss_fn(outputs, batch)

        (loss / self.grad_accum_steps).backward()
        self._accum_count += 1
        self._accum_sid = self.active_sid
        if self._accum_count >= self.grad_accum_steps:
            stepped_sid = self.active_sid
            self.optimizer.step()
            self._record_optimizer_step(stepped_sid)
            self.optimizer.zero_grad(set_to_none=True)
            self._accum_count = 0
            self._accum_sid = None
        return float(loss.detach().cpu().item())

    def run(self, all_batches: List[dict]) -> Tuple[List[dict], List[Optional[float]]]:
        decisions = []
        losses: List[Optional[float]] = []
        detect_interval = max(1, int(self.grad_accum_steps))
        last_decision_obj: Optional[Decision] = None
        detect_stats_accum: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        detect_accum_count = 0
        detect_batch_buffer: List[dict] = []

        no_prog = bool(getattr(self.args, "no_progress", False))
        pbar = (
            _tqdm(all_batches, desc="LS stream", total=len(all_batches), leave=True)
            if (_tqdm is not None and not no_prog)
            else None
        )
        iterable = pbar if pbar is not None else all_batches

        for step, batch in enumerate(iterable):
            detect_batch_buffer.append(dict(batch))
            stats_detect = self._compute_batch_stats_detect(batch)
            for lname, (a, g) in stats_detect.items():
                if lname not in detect_stats_accum:
                    detect_stats_accum[lname] = (a.detach().clone(), g.detach().clone())
                else:
                    a_acc, g_acc = detect_stats_accum[lname]
                    a_acc.add_(a)
                    g_acc.add_(g)
            detect_accum_count += 1

            detect_fired = False
            if detect_accum_count >= detect_interval:
                detect_fired = True
                avg_detect_window = {
                    lname: (a_sum / detect_accum_count, g_sum / detect_accum_count)
                    for lname, (a_sum, g_sum) in detect_stats_accum.items()
                }
                detect_stats_accum = {}
                detect_accum_count = 0

                detect_stats = self._subset_stats(avg_detect_window, self.detect_layers)
                detect_uv_full = get_UV(detect_stats, rank=self.rank_initial)
                uv_detect = {
                    name: (u[:, : self.detect_rank], v[:, : self.detect_rank]) for name, (u, v) in detect_uv_full.items()
                }
                window_uv = TaskUV(name=f"window_{step}", uv=uv_detect)
                decision_obj = self.engine.update(step=step, window_uv=window_uv)
                last_decision_obj = decision_obj

                if decision_obj.action == "NEW" and decision_obj.created_id is not None:
                    uv_full = self._compute_window_uv_full_selected(
                        detect_batch_buffer, uv_rank=self.rank_initial
                    )
                    self._create_subspace(decision_obj.created_id, uv_full)
                    self._switch_active_sid(decision_obj.created_id)
                elif decision_obj.action == "REUSE" and decision_obj.best_id in self.subspaces:
                    bid = decision_obj.best_id
                    if self._sid_may_run_optimizer(bid):
                        self._switch_active_sid(bid)
                    else:
                        # Budget exhausted for this window: do not attach optimizer / no further R updates.
                        self._switch_active_sid(None)
                elif decision_obj.action == "EXPAND" and decision_obj.best_id in self.subspaces:
                    expand_rank = (
                        self.expand_delta if self.expand_delta > 0 else self.rank_initial
                    )
                    uv_full = self._compute_window_uv_full_selected(
                        detect_batch_buffer, uv_rank=expand_rank
                    )
                    self._expand_subspace(decision_obj.best_id, uv_full)
                    # Keep boundary-matching prototype in detect feature space.
                    self.engine.subspaces[decision_obj.best_id] = TaskUV(name=decision_obj.best_id, uv=uv_detect)
                    # EXPAND replaces R parameter tensors; rebind optimizer/trainable flags even if sid unchanged.
                    self._switch_active_sid(decision_obj.best_id, force_rebind=True)

                detect_batch_buffer.clear()
            elif last_decision_obj is not None:
                decision_obj = last_decision_obj
            else:
                decision_obj = Decision(step=step, action="WAIT", best_id=None, best_sim=0.0, created_id=None)

            loss_val = self._train_active_step(batch)
            if loss_val is not None:
                losses.append(loss_val)
                decisions.append(
                    {
                        "step": step,
                        "action": decision_obj.action,
                        "best_id": decision_obj.best_id,
                        "best_sim": decision_obj.best_sim,
                        "created_id": decision_obj.created_id,
                        "active_sid": self.active_sid,
                        "detect_step": detect_fired,
                    }
                )
            if pbar is not None:
                lv = f"{loss_val:.4f}" if loss_val is not None else "n/a"
                pbar.set_postfix(
                    loss=lv,
                    act=decision_obj.action,
                    sid=str(self.active_sid) if self.active_sid is not None else "-",
                )
        self._flush_accumulated_step()
        return decisions, losses

    def total_r_parameters(self) -> int:
        """Sum of numel over all FiLoRA core R tensors in every wrapped layer."""
        n = 0
        for w in self.wrappers.values():
            for p in w.R_bank.values():
                n += int(p.numel())
        return n

    def save_subspace_meta(self, out_dir: Path):
        meta = {
            sid: {
                "rank": st.rank,
                "max_optimizer_steps": st.max_optimizer_steps,
                "optimizer_steps_done": st.optimizer_steps_done,
            }
            for sid, st in self.subspaces.items()
        }
        with (out_dir / "subspace_meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
