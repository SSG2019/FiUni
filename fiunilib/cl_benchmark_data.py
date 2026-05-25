"""
Load SC / LS continual-learning streams from the local processed ``CL`` tree.

Primary expected row schema (already formatted by upstream codebase):
  {"instruction": "...", "input": "...", "output": "..."}

We convert each row to:
  {"prompt": instruction + ("\\n" + input if input else ""), "answer": output}

Compatibility fallbacks are also supported:
  - {"prompt": "...", "answer": "..."}
  - {"sentence": "...", "label": "..."}
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset

from .data_utils import LS_ORDERS, SC_ORDERS, TaskData


def default_cl_root() -> Path:
    """Repo-root ``CL`` (sibling of ``fiunilib``)."""
    return Path(__file__).resolve().parent.parent / "CL"


def resolve_cl_root(explicit: Optional[str | Path] = None) -> Path:
    if explicit is not None and str(explicit).strip():
        return Path(explicit).expanduser().resolve()
    return default_cl_root()


def default_trace_root() -> Path:
    """Best-effort default TRACE root under repo `data/`."""
    base = Path(__file__).resolve().parent.parent / "data"
    candidates = [
        base / "TRACE-benchmark" / "LLM-CL-Benchmark",
        base / "TRACE-Benchmark" / "LLM-CL-Benchmark",
        base / "TRACE-Benchmark" / "LLM-CL-Benchmark_5000",
    ]
    for p in candidates:
        if p.is_dir():
            return p
    return candidates[0]


def resolve_trace_root(explicit: Optional[str | Path] = None) -> Path:
    if explicit is not None and str(explicit).strip():
        p = Path(explicit).expanduser().resolve()
        if p.is_dir():
            return p
        # Common TRACE naming variants.
        alt = str(p).replace("TRACE-benchmark", "TRACE-Benchmark")
        alt = alt.replace("LLM-CL-Benchmark", "LLM-CL-Benchmark_5000")
        p_alt = Path(alt)
        if p_alt.is_dir():
            return p_alt.resolve()
        return p
    return default_trace_root()


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list of objects, got {type(data)}")
    return data


def _row_to_example(ex: Dict[str, Any]) -> Dict[str, str]:
    if "instruction" in ex and "output" in ex:
        ins = str(ex["instruction"])
        inp = str(ex.get("input", "")).strip()
        prompt = ins if not inp else f"{ins}\n{inp}"
        return {"prompt": prompt, "answer": str(ex["output"]).strip()}
    if "prompt" in ex and "answer" in ex:
        return {"prompt": str(ex["prompt"]), "answer": str(ex["answer"]).strip()}
    if "sentence" in ex and "label" in ex:
        return {"prompt": str(ex["sentence"]), "answer": str(ex["label"]).strip()}
    raise KeyError(
        "Each row must have (instruction, output) or (prompt, answer) or (sentence, label); "
        f"got keys: {list(ex.keys())}"
    )


def _dataset_from_rows(rows: List[Dict[str, Any]]) -> Dataset:
    return Dataset.from_list([_row_to_example(r) for r in rows])


def _row_to_example_trace(ex: Dict[str, Any]) -> Dict[str, str]:
    """
    TRACE row conversion that preserves an explicit source field for SARI-style metrics.
    """
    base = _row_to_example(ex)
    source = None
    for k in ("source", "input", "text", "content", "sentence", "article"):
        if k in ex and str(ex.get(k, "")).strip():
            source = str(ex[k]).strip()
            break
    if source is not None:
        base["source"] = source
    return base


def _dataset_from_rows_trace(rows: List[Dict[str, Any]]) -> Dataset:
    return Dataset.from_list([_row_to_example_trace(r) for r in rows])


# Internal task key -> path segments under CL root
_CL_SC_PATHS: Dict[str, Tuple[str, ...]] = {
    "dbpedia": ("TC", "dbpedia"),
    "amazon": ("SC", "amazon"),
    "yahoo": ("TC", "yahoo"),
    "ag": ("TC", "agnews"),
}

_CL_LS_PATHS: Dict[str, Tuple[str, ...]] = {
    "mnli": ("NLI", "MNLI"),
    "cb": ("NLI", "CB"),
    "wic": ("WiC", "WiC"),
    "copa": ("COPA", "COPA"),
    "qqp": ("QQP", "QQP"),
    "boolqa": ("BoolQA", "BoolQA"),
    "rte": ("NLI", "RTE"),
    "imdb": ("SC", "IMDB"),
    "yelp": ("SC", "yelp"),
    "amazon": ("SC", "amazon"),
    "sst-2": ("SC", "SST-2"),
    "dbpedia": ("TC", "dbpedia"),
    "ag": ("TC", "agnews"),
    "multirc": ("MultiRC", "MultiRC"),
    "yahoo": ("TC", "yahoo"),
}

TRACE_ORDERS: Dict[int, List[str]] = {
    7: ["c-stance", "fomc", "meetingbank", "py150", "scienceqa", "numglue-cm", "numglue-ds", "20minuten"],
    8: ["meetingbank"],
}

_TRACE_TASK_PATHS: Dict[str, Tuple[str, ...]] = {
    "c-stance": ("C-STANCE",),
    "fomc": ("FOMC",),
    "meetingbank": ("MeetingBank",),
    "py150": ("Py150",),
    "scienceqa": ("ScienceQA",),
    "numglue-cm": ("NumGLUE-cm",),
    "numglue-ds": ("NumGLUE-ds",),
    "20minuten": ("20Minuten",),
}


def _load_one_cl_task(
    root: Path,
    path_parts: Tuple[str, ...],
    task_key: str,
    *,
    benchmark_tag: str,
    order_id: int,
    seed: int,
    train_file: str = "train.json",
    test_file: str = "test.json",
) -> TaskData:
    task_dir = root.joinpath(*path_parts)
    train_p = task_dir / train_file
    test_p = task_dir / test_file
    for p in (train_p, test_p):
        if not p.is_file():
            raise FileNotFoundError(f"Missing file for task {task_key}: {p}")

    train_rows = _load_json_list(train_p)
    test_rows = _load_json_list(test_p)

    train_ds = _dataset_from_rows(train_rows)
    test_ds = _dataset_from_rows(test_rows)

    meta: Dict[str, Any] = {
        "benchmark": benchmark_tag,
        "order_id": order_id,
        "cl_dir": str(task_dir.resolve()),
        "train_file": train_file,
        "test_file": test_file,
        "seed": seed,
    }

    return TaskData(
        name=task_key,
        train=train_ds,
        test=test_ds,
        validation=None,
        task_type="classification",
        metric="accuracy",
        meta=meta,
    )


def prepare_sc_tasks_from_cl_benchmark(
    cl_benchmark_root: Optional[str | Path] = None,
    order_id: int = 1,
    seed: int = 42,
    train_file: str = "train.json",
    test_file: str = "test.json",
) -> OrderedDict[str, TaskData]:
    """
    SC stream (orders 1–3) from local processed ``CL`` data.
    """
    if order_id not in SC_ORDERS:
        raise ValueError(f"order_id must be one of {list(SC_ORDERS.keys())}")
    root = resolve_cl_root(cl_benchmark_root)
    if not root.is_dir():
        raise FileNotFoundError(f"CL root not found: {root}")

    out: OrderedDict[str, TaskData] = OrderedDict()
    for task in SC_ORDERS[order_id]:
        parts = _CL_SC_PATHS[task]
        out[task] = _load_one_cl_task(
            root,
            parts,
            task,
            benchmark_tag="SC-CL",
            order_id=order_id,
            seed=seed,
            train_file=train_file,
            test_file=test_file,
        )
    return out


def prepare_ls_tasks_from_cl_benchmark(
    cl_benchmark_root: Optional[str | Path] = None,
    order_id: int = 4,
    seed: int = 42,
    train_file: str = "train.json",
    test_file: str = "test.json",
) -> OrderedDict[str, TaskData]:
    """
    LS stream (orders 4–6) from local processed ``CL`` data.
    """
    if order_id not in LS_ORDERS:
        raise ValueError(f"order_id must be one of {list(LS_ORDERS.keys())}")
    root = resolve_cl_root(cl_benchmark_root)
    if not root.is_dir():
        raise FileNotFoundError(f"CL root not found: {root}")

    out: OrderedDict[str, TaskData] = OrderedDict()
    for task in LS_ORDERS[order_id]:
        parts = _CL_LS_PATHS[task]
        out[task] = _load_one_cl_task(
            root,
            parts,
            task,
            benchmark_tag="LS-CL",
            order_id=order_id,
            seed=seed,
            train_file=train_file,
            test_file=test_file,
        )
    return out


def prepare_trace_tasks_from_cl_benchmark(
    trace_root: Optional[str | Path] = None,
    order_id: int = 7,
    seed: int = 42,
    train_file: str = "train.json",
    test_file: str = "test.json",
) -> OrderedDict[str, TaskData]:
    """
    TRACE stream (order 7) from local processed TRACE data.
    """
    if order_id not in TRACE_ORDERS:
        raise ValueError(f"order_id must be one of {list(TRACE_ORDERS.keys())}")
    root = resolve_trace_root(trace_root)
    if not root.is_dir():
        raise FileNotFoundError(f"TRACE root not found: {root}")

    out: OrderedDict[str, TaskData] = OrderedDict()
    for task in TRACE_ORDERS[order_id]:
        parts = _TRACE_TASK_PATHS[task]
        task_dir = root.joinpath(*parts)
        train_p = task_dir / train_file
        test_p = task_dir / test_file
        for p in (train_p, test_p):
            if not p.is_file():
                raise FileNotFoundError(f"Missing file for task {task}: {p}")

        train_rows = _load_json_list(train_p)
        test_rows = _load_json_list(test_p)
        train_ds = _dataset_from_rows_trace(train_rows)
        test_ds = _dataset_from_rows_trace(test_rows)
        out[task] = TaskData(
            name=task,
            train=train_ds,
            test=test_ds,
            validation=None,
            task_type="generative_or_mixed",
            metric=None,
            meta={
                "benchmark": "TRACE",
                "order_id": order_id,
                "cl_dir": str(task_dir.resolve()),
                "train_file": train_file,
                "test_file": test_file,
                "seed": seed,
            },
        )
    return out


def prepare_sc_tasks_from_cl(
    cl_root: Optional[str | Path] = None,
    order_id: int = 1,
    seed: int = 42,
    train_file: str = "train.json",
    test_file: str = "test.json",
) -> OrderedDict[str, TaskData]:
    """Alias of ``prepare_sc_tasks_from_cl_benchmark`` using ``cl_root`` name."""
    return prepare_sc_tasks_from_cl_benchmark(
        cl_benchmark_root=cl_root,
        order_id=order_id,
        seed=seed,
        train_file=train_file,
        test_file=test_file,
    )


def prepare_ls_tasks_from_cl(
    cl_root: Optional[str | Path] = None,
    order_id: int = 4,
    seed: int = 42,
    train_file: str = "train.json",
    test_file: str = "test.json",
) -> OrderedDict[str, TaskData]:
    """Alias of ``prepare_ls_tasks_from_cl_benchmark`` using ``cl_root`` name."""
    return prepare_ls_tasks_from_cl_benchmark(
        cl_benchmark_root=cl_root,
        order_id=order_id,
        seed=seed,
        train_file=train_file,
        test_file=test_file,
    )


def prepare_trace_tasks_from_cl(
    trace_root: Optional[str | Path] = None,
    order_id: int = 7,
    seed: int = 42,
    train_file: str = "train.json",
    test_file: str = "test.json",
) -> OrderedDict[str, TaskData]:
    """Alias of ``prepare_trace_tasks_from_cl_benchmark``."""
    return prepare_trace_tasks_from_cl_benchmark(
        trace_root=trace_root,
        order_id=order_id,
        seed=seed,
        train_file=train_file,
        test_file=test_file,
    )
