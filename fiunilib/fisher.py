from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

LayerRef = Union[str, nn.Linear]

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _unwrap_linear_module(mod: nn.Module) -> Optional[nn.Linear]:
    """
    Return the underlying nn.Linear for Fisher hooks. FiLoRA wrappers (e.g. BankedFiLoRALinear)
    expose the frozen backbone as `.base`.
    """
    if isinstance(mod, nn.Linear):
        return mod
    base = getattr(mod, "base", None)
    if isinstance(base, nn.Linear):
        return base
    return None


def _resolve_layer_to_linear(model: nn.Module, ref: LayerRef) -> Tuple[str, nn.Linear]:
    """Resolve string path or module ref to (name, nn.Linear) for activation hooks."""
    if isinstance(ref, nn.Linear):
        name = None
        for n, m in model.named_modules():
            if m is ref:
                name = n
                break
        lin = _unwrap_linear_module(ref)
        if lin is None:
            raise TypeError(f"{type(ref)} does not wrap or equal nn.Linear")
        return (name or f"<unnamed_linear_{id(ref)}>", lin)
    obj: nn.Module = model
    path = ref
    for p in path.split("."):
        obj = getattr(obj, p)
    lin = _unwrap_linear_module(obj)
    if lin is None:
        raise TypeError(f"{ref} is not nn.Linear (got {type(obj).__name__})")
    return path, lin


def _model_device(model: nn.Module) -> torch.device:
    """Device of model parameters (frozen models still have tensors on cuda)."""
    p = next(model.parameters(), None)
    return p.device if p is not None else torch.device("cpu")


def compute_ag(
    model: nn.Module,
    layer_refs: List[LayerRef],
    dataloader,
    loss_fn,
    use_autocast: bool = False,
    autocast_dtype: Optional[torch.dtype] = None,
    show_progress: bool = False,
    progress_desc: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    device = _model_device(model)
    model.eval()

    def _flat2d(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(-1, t.shape[-1]) if t.dim() > 2 else t

    def _to_device_batch(b: dict) -> dict:
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}

    layers = [_resolve_layer_to_linear(model, r) for r in layer_refs]
    results: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    handles = []
    for name, lin in layers:
        d_in, d_out = lin.in_features, lin.out_features
        a_sum = torch.zeros(d_in, d_in, device=device)
        g_sum = torch.zeros(d_out, d_out, device=device)
        n_t = torch.zeros(1, device=device)

        def fwd_hook(mod, inp, out, a_acc=a_sum, g_acc=g_sum, n_cnt=n_t):
            x = _flat2d(inp[0].detach())
            a_acc.add_(x.T @ x)
            n_cnt.add_(x.size(0))

            out.requires_grad_(True)

            def bwd_hook(grad_out):
                delta = _flat2d(grad_out)
                g_acc.add_(delta.T @ delta)

            out.register_hook(bwd_hook)

        handles.append((name, a_sum, g_sum, n_t, lin.register_forward_hook(fwd_hook)))

    it = dataloader
    if show_progress and tqdm is not None:
        try:
            total = len(dataloader)
        except Exception:
            total = None
        it = tqdm(dataloader, total=total, desc=progress_desc, leave=False)

    for batch in it:
        batch = _to_device_batch(batch)
        model.zero_grad(set_to_none=True)
        inputs = {k: v for k, v in batch.items() if torch.is_tensor(v)}

        if use_autocast and autocast_dtype is not None:
            cast_ctx = torch.autocast(device_type=device.type, dtype=autocast_dtype)
        else:
            cast_ctx = nullcontext()

        with cast_ctx:
            outputs = model(**inputs)
            loss = loss_fn(outputs, batch)
        loss.backward()

        del outputs, loss
        torch.cuda.empty_cache()

    for name, a_sum, g_sum, n_t, h in handles:
        h.remove()
        n = max(1.0, float(n_t.item()))
        a = 0.5 * ((a_sum / n) + (a_sum / n).T)
        g = 0.5 * ((g_sum / n) + (g_sum / n).T)
        results[name] = (a, g)

    del handles
    torch.cuda.empty_cache()
    return results


def _compute_ag_batch_raw_resolved(
    model: nn.Module,
    resolved: List[Tuple[str, nn.Linear]],
    batch: dict,
    loss_fn,
    use_autocast: bool = False,
    autocast_dtype: Optional[torch.dtype] = None,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Single forward+backward with hooks on ``resolved`` (name, Linear) pairs only."""
    device = _model_device(model)
    model.eval()

    def _flat2d(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(-1, t.shape[-1]) if t.dim() > 2 else t

    def _to_device_batch(b: dict) -> dict:
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}

    handles = []
    for name, lin in resolved:
        d_in, d_out = lin.in_features, lin.out_features
        a_sum = torch.zeros(d_in, d_in, device=device)
        g_sum = torch.zeros(d_out, d_out, device=device)
        n_t = torch.zeros(1, device=device)

        def fwd_hook(mod, inp, out, a_acc=a_sum, g_acc=g_sum, n_cnt=n_t):
            x = _flat2d(inp[0].detach())
            a_acc.add_(x.T @ x)
            n_cnt.add_(x.size(0))
            out.requires_grad_(True)

            def bwd_hook(grad_out):
                delta = _flat2d(grad_out)
                g_acc.add_(delta.T @ delta)

            out.register_hook(bwd_hook)

        handles.append((name, a_sum, g_sum, n_t, lin.register_forward_hook(fwd_hook)))

    batch = _to_device_batch(batch)
    model.zero_grad(set_to_none=True)
    inputs = {k: v for k, v in batch.items() if torch.is_tensor(v)}

    if use_autocast and autocast_dtype is not None:
        cast_ctx = torch.autocast(device_type=device.type, dtype=autocast_dtype)
    else:
        cast_ctx = nullcontext()

    with cast_ctx:
        outputs = model(**inputs)
        loss = loss_fn(outputs, batch)
    loss.backward()
    del outputs, loss
    torch.cuda.empty_cache()

    stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for name, a_sum, g_sum, n_t, h in handles:
        h.remove()
        n = max(1.0, float(n_t.item()))
        a = a_sum / n
        g = g_sum / n
        a = 0.5 * (a + a.T)
        g = 0.5 * (g + g.T)
        stats[name] = (a.detach().clone(), g.detach().clone())

    del handles
    torch.cuda.empty_cache()
    return stats


def compute_ag_batch_raw(
    model: nn.Module,
    layer_refs: List[LayerRef],
    batch: dict,
    loss_fn,
    use_autocast: bool = False,
    autocast_dtype: Optional[torch.dtype] = None,
    group_size: int = 1,
    eager_uv_rank: Optional[int] = None,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Single-batch Fisher/K-FAC statistics.
    Returns per-layer normalized and symmetrized (A, G), **unless** multi-pass eager UV is used.

    ``group_size`` means: split ``layer_refs`` into this many **contiguous partitions**
    (as equal in length as possible), run one full forward+backward per partition with
    hooks only on that partition’s layers, then merge dicts.

    * ``group_size == 1``: one pass, hooks on **all** listed layers (same cost shape as
      legacy single-pass AG).
    * ``group_size == 4``: up to four passes, each pass hooks roughly one quarter of the
      list (reduces peak A/G memory vs one pass).

    ``eager_uv_rank``: when set together with ``num_passes > 1``, after each partition
    immediately runs ``compute_UV_from_AG`` on that partition’s layers and **drops** full
    (A, G) matrices, keeping only low-rank (U, V) in ``merged``. This lowers peak VRAM
    between passes. The returned dict then holds (U, V) per layer (rank ``eager_uv_rank``),
    not (A, G). Callers must not pass these through ``get_UV`` again.

    This is **not** “only ``group_size`` modules per pass”.
    """
    resolved = [_resolve_layer_to_linear(model, r) for r in layer_refs]
    n = len(resolved)
    if n == 0:
        return {}
    requested = max(1, int(group_size))
    # Cannot have more non-empty partitions than layers; avoid empty chunks.
    num_passes = min(requested, n)
    merged: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    use_eager_uv = eager_uv_rank is not None and int(eager_uv_rank) > 0 and num_passes > 1
    rank_uv = int(eager_uv_rank) if use_eager_uv else 0

    for p in range(num_passes):
        start = (p * n) // num_passes
        end = ((p + 1) * n) // num_passes
        chunk = resolved[start:end]
        if not chunk:
            continue
        part = _compute_ag_batch_raw_resolved(
            model, chunk, batch, loss_fn, use_autocast=use_autocast, autocast_dtype=autocast_dtype
        )
        if use_eager_uv:
            for name, (a, g) in part.items():
                u, v = compute_UV_from_AG(a, g, rank_uv)
                merged[name] = (u.detach().clone(), v.detach().clone())
                del a, g
            part.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            merged.update(part)
    return merged


def compute_ag_window_grouped(
    model: nn.Module,
    layer_refs: List[LayerRef],
    batches: List[dict],
    loss_fn,
    use_autocast: bool = False,
    autocast_dtype: Optional[torch.dtype] = None,
    group_size: int = 1,
    uv_rank: Optional[int] = None,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Window Fisher/K-FAC statistics with layer chunking.

    For each chunk of layers:
      1) iterate all ``batches`` and accumulate chunk (A,G),
      2) average over window length,
      3) optionally decompose once to (U,V) with ``uv_rank``.

    This implements "accumulate over K batches first, then decompose once per chunk".
    """
    resolved = [_resolve_layer_to_linear(model, r) for r in layer_refs]
    n = len(resolved)
    if n == 0 or not batches:
        return {}

    requested = max(1, int(group_size))
    num_passes = min(requested, n)
    use_uv = uv_rank is not None and int(uv_rank) > 0
    rank_uv = int(uv_rank) if use_uv else 0
    merged: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    n_batches = len(batches)
    inv = 1.0 / float(n_batches)

    for p in range(num_passes):
        start = (p * n) // num_passes
        end = ((p + 1) * n) // num_passes
        chunk = resolved[start:end]
        if not chunk:
            continue

        acc: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for b in batches:
            part = _compute_ag_batch_raw_resolved(
                model, chunk, b, loss_fn, use_autocast=use_autocast, autocast_dtype=autocast_dtype
            )
            for name, (a, g) in part.items():
                if name not in acc:
                    acc[name] = (a.detach().clone(), g.detach().clone())
                else:
                    a0, g0 = acc[name]
                    a0.add_(a)
                    g0.add_(g)
            part.clear()

        for name, (a_sum, g_sum) in acc.items():
            a = a_sum * inv
            g = g_sum * inv
            a = 0.5 * (a + a.T)
            g = 0.5 * (g + g.T)
            if use_uv:
                u, v = compute_UV_from_AG(a, g, rank_uv)
                merged[name] = (u.detach().clone(), v.detach().clone())
            else:
                merged[name] = (a.detach().clone(), g.detach().clone())
        acc.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return merged


def merge_raw_window_to_ag(
    window_contribs: List[Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor]], float]],
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Merge per-batch raw sums into the same normalized A,G as compute_ag over those batches.
    """
    if not window_contribs:
        return {}

    layer_names: List[str] = []
    for raw_dict, _ in window_contribs:
        if raw_dict:
            layer_names = sorted(raw_dict.keys())
            break
    if not layer_names:
        return {}

    a_tot: Dict[str, torch.Tensor] = {}
    g_tot: Dict[str, torch.Tensor] = {}
    n_total = 0.0

    for raw_dict, n_b in window_contribs:
        n_total += n_b
        for name in layer_names:
            if name not in raw_dict:
                continue
            a_s, g_s = raw_dict[name]
            if name not in a_tot:
                a_tot[name] = a_s.clone()
                g_tot[name] = g_s.clone()
            else:
                a_tot[name].add_(a_s)
                g_tot[name].add_(g_s)

    n_total = max(1.0, n_total)
    results: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for name in layer_names:
        if name not in a_tot:
            continue
        a_sum = a_tot[name]
        g_sum = g_tot[name]
        a = 0.5 * ((a_sum / n_total) + (a_sum / n_total).T)
        g = 0.5 * ((g_sum / n_total) + (g_sum / n_total).T)
        results[name] = (a, g)
    return results


def compute_UV_from_AG(A: torch.Tensor, G: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    def _robust_top_eigvecs(M: torch.Tensor, k: int) -> torch.Tensor:
        if not torch.isfinite(M).all():
            M = torch.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

        M = 0.5 * (M + M.T)
        device = M.device
        M64 = M.to(dtype=torch.float64)
        n = M64.shape[0]
        kk = min(max(1, int(k)), n)

        eye = torch.eye(n, device=M64.device, dtype=M64.dtype)
        base_scale = torch.mean(torch.diag(M64).abs()).item()
        if not (base_scale > 0):
            base_scale = 1.0

        jitters = [0.0, 1e-8 * base_scale, 1e-6 * base_scale, 1e-4 * base_scale]
        last_err = None
        for eps in jitters:
            try:
                m_try = M64 if eps == 0.0 else (M64 + eps * eye)
                evals, evecs = torch.linalg.eigh(m_try)
                idx = torch.argsort(evals, descending=True)[:kk]
                return evecs[:, idx].to(device=device, dtype=M.dtype).contiguous()
            except RuntimeError as e:
                last_err = e

        try:
            m_cpu = M64.detach().cpu()
            evals, evecs = torch.linalg.eigh(m_cpu + (1e-4 * base_scale) * torch.eye(n, dtype=m_cpu.dtype))
            idx = torch.argsort(evals, descending=True)[:kk]
            return evecs[:, idx].to(device=device, dtype=M.dtype).contiguous()
        except RuntimeError:
            raise last_err

    v_r = _robust_top_eigvecs(A, rank)
    u_r = _robust_top_eigvecs(G, rank)
    return u_r, v_r


def get_UV(stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]], rank: int):
    results: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for layer_name, (a, g) in stats.items():
        u, v = compute_UV_from_AG(a, g, rank)
        results[layer_name] = (u, v)
    return results


# Backward-compatible aliases.
compute_uv_from_ag = compute_UV_from_AG
get_uv = get_UV
