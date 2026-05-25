import copy, math, re
import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F


class FiLoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, U: torch.Tensor, V: torch.Tensor):
        super().__init__()
        assert isinstance(linear, nn.Linear)
        assert U.size(1) == V.size(1), "U and V must have the same rank"

        self.filora = linear
        dev = linear.weight.device
        dt = linear.weight.dtype
        d_in, d_out = linear.in_features, linear.out_features
        r = U.size(1)

        self.register_buffer("U", U.to(device=dev, dtype=dt).contiguous())  # [d_out, r]
        self.register_buffer("V", V.to(device=dev, dtype=dt).contiguous())  # [d_in,  r]
        self.register_buffer("r", torch.tensor(r, device=dev, dtype=dt))

        sigma = 1e-5
        with torch.no_grad():
            RR = torch.randn(r, r, device=dev, dtype=dt) * sigma

        self.R = nn.Parameter(RR)  # [r, r]

    def forward(self, x: torch.Tensor):
        base_out = self.filora(x)  # [B, d_out]
        t = F.linear(x, self.V.t())       # [B, r]
        t = t.matmul(self.R.t())          # [B, r]
        delta_main = F.linear(t, self.U)  # [B, d_out]
        return base_out + delta_main


def replace_one_with_filora(model: nn.Module, layer_name: str, U, V):
    parts = layer_name.split(".")
    submodule = model
    for p in parts[:-1]:
        submodule = getattr(submodule, p)

    target = getattr(submodule, parts[-1])
    assert isinstance(target, nn.Linear), f"{layer_name} is not nn.Linear"
    new_layer = FiLoRALinear(target, U, V)
    setattr(submodule, parts[-1], new_layer)
    return model


@torch.no_grad()
def merge_filora_layers(model: nn.Module, inplace: bool = True, strip_name: bool = True) -> nn.Module:
    if not inplace:
        model = copy.deepcopy(model)

    F1 = globals().get("FiLoRALinear", None)
    F2 = globals().get("FiLoRALinear_new", None)
    WRAP_TYPES = tuple([t for t in (F1, F2) if isinstance(t, type)])

    def is_filora_wrapper(m: nn.Module) -> bool:
        if WRAP_TYPES and isinstance(m, WRAP_TYPES):
            return True
        has_core = all(hasattr(m, k) for k in ("filora", "U", "V"))
        has_s = hasattr(m, "S") or all(hasattr(m, k) for k in ("S1", "S2", "S3"))
        is_base_linear = isinstance(getattr(m, "filora", None), nn.Linear)
        return bool(has_core and has_s and is_base_linear)

    def _strip_filora(name: str) -> str:
        s = re.sub(r'(?i)(^|[._-])filora(?=([._-]|$))', lambda m: m.group(1), name)
        s = re.sub(r'[._-]{2,}', lambda m: m.group(0)[0], s)
        s = s.strip('._-')
        return s or name

    def _container_set(module: nn.Module, key: str, value: nn.Module, new_key: str = None):
        tgt = new_key if (new_key is not None) else key
        if isinstance(module, (nn.ModuleList, nn.Sequential)):
            idx = int(key)
            module[idx] = value
        elif isinstance(module, nn.ModuleDict):
            if new_key is not None and new_key != key:
                module[new_key] = value
                if key in module:
                    del module[key]
            else:
                module[key] = value
        else:
            setattr(module, tgt, value)
            if new_key is not None and new_key != key and key in module._modules:
                del module._modules[key]

    def _compute_S(child) -> torch.Tensor:
        if hasattr(child, "S"):
            return child.S
        elif all(hasattr(child, k) for k in ("S1", "S2", "S3")):
            return torch.diag(child.S1) @ child.S2 @ torch.diag(child.S3)
        else:
            raise RuntimeError("Unknown FiLoRA wrapper: cannot find S or (S1,S2,S3).")

    def _get_base_linear(child):
        base = getattr(child, "filora", None)
        if base is None:
            base = getattr(child, "linear", None)
        if not isinstance(base, nn.Linear):
            raise RuntimeError("Wrapper does not contain a base nn.Linear (expected attribute 'filora').")
        return base

    def recurse(module: nn.Module):
        for name, child in list(module._modules.items()):
            if is_filora_wrapper(child):
                base = _get_base_linear(child)
                dev, dt = base.weight.device, base.weight.dtype

                U, V = child.U, child.V
                S = _compute_S(child)

                delta_w = (U @ S @ V.transpose(0, 1)).to(device=dev, dtype=dt)
                merged_w = (base.weight + delta_w).to(device=dev, dtype=dt)
                merged_b = None if base.bias is None else base.bias.to(device=dev, dtype=dt)

                new_linear = nn.Linear(
                    in_features=base.in_features,
                    out_features=base.out_features,
                    bias=(merged_b is not None),
                    device=dev,
                    dtype=dt
                )
                new_linear.weight.copy_(merged_w)
                if merged_b is not None:
                    new_linear.bias.copy_(merged_b)

                if strip_name and re.search(r'(?i)filora', name):
                    new_name = _strip_filora(name)
                else:
                    new_name = name

                _container_set(module, key=name, value=new_linear, new_key=(new_name if new_name != name else None))
            else:
                recurse(child)

    recurse(model)

    for n, m in model.named_modules():
        if is_filora_wrapper(m):
            raise RuntimeError(f"Unmerged FiLoRA wrapper remains at: {n}")

    return model
