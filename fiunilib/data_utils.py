from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

from datasets import Dataset, load_dataset


# =========================
# ELLA benchmark constants
# =========================

SC_ORDERS = {
    1: ["dbpedia", "amazon", "yahoo", "ag"],
    2: ["dbpedia", "amazon", "ag", "yahoo"],
    3: ["yahoo", "amazon", "ag", "dbpedia"],
}

LS_ORDERS = {
    4: ["mnli", "cb", "wic", "copa", "qqp", "boolqa", "rte", "imdb",
        "yelp", "amazon", "sst-2", "dbpedia", "ag", "multirc", "yahoo"],
    5: ["multirc", "boolqa", "wic", "mnli", "cb", "copa", "qqp", "rte",
        "imdb", "sst-2", "dbpedia", "ag", "yelp", "amazon", "yahoo"],
    6: ["yelp", "amazon", "mnli", "cb", "copa", "qqp", "rte", "imdb",
        "sst-2", "dbpedia", "ag", "yahoo", "multirc", "boolqa", "wic"],
}

TRACE_ORDER = [
    "c-stance", "fomc", "meetingbank", "py150",
    "scienceqa", "numglue-cm", "numglue-ds", "20minuten"
]

TRACE_EPOCHS = {
    "c-stance": 5,
    "fomc": 3,
    "meetingbank": 7,
    "py150": 5,
    "scienceqa": 3,
    "numglue-cm": 5,
    "numglue-ds": 5,
    "20minuten": 7,
}

PROMPTS = {
    "NLI": 'What is the logical relationship between the "sentence 1" and the "sentence 2"?\nChoose one from the option.',
    "QQP": 'Whether the "first sentence" and the "second sentence" have the same meaning?\nChoose one from the option.',
    "SC": "What is the sentiment of the following paragraph? Choose one from the option.",
    "TC": "What is the topic of the following paragraph? Choose one from the option.",
    "BoolQA": "According to the following passage, is the question true or false?\nChoose one from the option.",
    "MultiRC": "According to the following passage and question, is the candidate answer true or false? Choose one from the option.",
    "WiC": "Given a word and two sentences, whether the word is used with the same sense in both sentence? Choose one from the option.",
    "FOMC": "What is the monetary policy stance for the following text?\nChoose one from the option.",
    "20Minuten": "Provide a simplified version of the following paragraph in German.",
    "ScienceQA": "Choose an answer for the following question and give your reasons.",
    "NumGLUE-cm": "Solve the following math problem.",
    "NumGLUE-ds": "Solve the following math problem.",
    "Py150": "Continue writing the code.",
    "MeetingBank": "Write a summary of the following meeting transcripts.",
    "C-STANCE": "Determine the attitude of the following text towards the specified object.\nChoose one from the option.",
}

HF_DEFAULTS = {
    "ag": ("ag_news", None),
    "dbpedia": ("dbpedia_14", None),
    "yahoo": ("yahoo_answers_topics", None),
    "amazon": ("vgaraujov/amazon_review_full", None),
    "yelp": ("yelp_review_full", None),
    "sst-2": ("glue", "sst2"),
    "mnli": ("glue", "mnli"),
    "qqp": ("glue", "qqp"),
    "rte": ("glue", "rte"),
    "wic": ("super_glue", "wic"),
    "cb": ("super_glue", "cb"),
    "copa": ("super_glue", "copa"),
    "boolqa": ("super_glue", "boolq"),
    "multirc": ("super_glue", "multirc"),
    "imdb": ("imdb", None),
}


# =========================
# Helper dataclass
# =========================

@dataclass
class TaskData:
    name: str
    train: Dataset
    test: Dataset
    validation: Optional[Dataset] = None
    task_type: Optional[str] = None
    metric: Optional[str] = None
    epochs: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None


# =========================
# Low-level helpers
# =========================

def _set_seed(seed: int) -> random.Random:
    return random.Random(seed)


def _dataset_name(task: str,
                  overrides: Optional[Dict[str, Tuple[str, Optional[str]]]] = None):
    if overrides and task in overrides:
        return overrides[task]
    return HF_DEFAULTS[task]


def _load_hf(task: str,
             cache_dir: Optional[str] = None,
             overrides: Optional[Dict[str, Tuple[str, Optional[str]]]] = None):
    ds_name, subset = _dataset_name(task, overrides)
    if subset is None:
        return load_dataset(ds_name, cache_dir=cache_dir)
    return load_dataset(ds_name, subset, cache_dir=cache_dir)


def _safe_text_join(*parts: Any) -> str:
    vals = [str(x).strip() for x in parts if x is not None and str(x).strip()]
    return "\n".join(vals)


def _build_instruction_example(task_instruction: str,
                               options: List[str],
                               text_block: str,
                               answer: str) -> Dict[str, str]:
    prompt = (
        f"Task Instruction: {task_instruction}\n"
        f"Options: {', '.join(options)}\n"
        f"Text: {text_block}\n"
        f"Answer:"
    )
    return {"prompt": prompt, "answer": answer}


def _map_dataset(dataset: Dataset,
                 mapper: Callable[[Dict[str, Any]], Dict[str, str]]) -> Dataset:
    cols = dataset.column_names
    return dataset.map(mapper, remove_columns=cols)


def _sample_dataset(dataset: Dataset,
                    max_samples: Optional[int],
                    seed: int) -> Dataset:
    """
    Randomly sample INSIDE the original split only.
    If max_samples is None or len(dataset) <= max_samples, return as-is.
    """
    if max_samples is None or len(dataset) <= max_samples:
        return dataset

    rng = _set_seed(seed)
    indices = rng.sample(range(len(dataset)), max_samples)
    indices.sort()
    return dataset.select(indices)


def _sample_dataset_per_class(dataset: Dataset,
                              label_key: str,
                              max_per_class: int,
                              seed: int) -> Dataset:
    """
    Sample INSIDE the original split only, with at most `max_per_class`
    examples for each class. If a class has fewer than max_per_class,
    keep them all.
    """
    if label_key not in dataset.column_names:
        raise ValueError(
            f"Label column '{label_key}' not found in dataset columns: {dataset.column_names}"
        )

    rng = _set_seed(seed)
    buckets: Dict[int, List[int]] = {}

    for i, y in enumerate(dataset[label_key]):
        # skip unlabeled examples if any
        if y == -1:
            continue
        buckets.setdefault(int(y), []).append(i)

    selected: List[int] = []
    for _, idxs in buckets.items():
        if len(idxs) <= max_per_class:
            selected.extend(idxs)
        else:
            selected.extend(rng.sample(idxs, max_per_class))

    selected.sort()
    return dataset.select(selected)


def _first_available_split(ds_dict,
                           split_names: List[str]) -> Optional[Dataset]:
    for name in split_names:
        if name in ds_dict:
            return ds_dict[name]
    return None


def _filter_unlabeled_split(dataset: Dataset,
                            label_key: str = "label") -> Dataset:
    if label_key not in dataset.column_names:
        return dataset

    valid_idx = [i for i, y in enumerate(dataset[label_key]) if y != -1]
    if len(valid_idx) == len(dataset):
        return dataset
    return dataset.select(valid_idx)


def _json_dataset(path: Path) -> Dataset:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    return Dataset.from_list(rows)


# =========================
# Task-specific constants
# =========================

DBPEDIA_LABELS = [
    "company", "educational institution", "artist", "athlete", "office holder",
    "mean of transportation", "building", "natural place", "village",
    "animal", "plant", "album", "film", "written work"
]

AG_LABELS = ["world", "sports", "business", "science and technology"]

YAHOO_LABELS = [
    "society and culture", "science and mathematics", "health", "education and reference",
    "computers and internet", "sports", "business and finance", "entertainment and music",
    "family and relationships", "politics and government"
]

BINARY_SENTIMENT = ["negative", "positive"]
AMAZON_FULL_LABELS = ["very negative", "negative", "neutral", "positive", "very positive"]
NLI_LABELS = ["entailment", "neutral", "contradiction"]
CB_LABELS = ["entailment", "contradiction", "neutral"]
RTE_LABELS = ["entailment", "not_entailment"]
BOOL_LABELS = ["false", "true"]
YES_NO_LABELS = ["no", "yes"]


# =========================
# Task-specific mappers
# =========================

def _map_ag(ex):
    return _build_instruction_example(
        task_instruction=PROMPTS["TC"],
        options=AG_LABELS,
        text_block=ex["text"],
        answer=AG_LABELS[int(ex["label"])],
    )


def _map_amazon(ex):
    text = _safe_text_join(ex.get("title"), ex.get("content"))
    return _build_instruction_example(
        task_instruction=PROMPTS["SC"],
        options=AMAZON_FULL_LABELS,
        text_block=text,
        answer=AMAZON_FULL_LABELS[int(ex["label"])],
    )


def _map_yelp(ex):
    return _build_instruction_example(
        task_instruction=PROMPTS["SC"],
        options=AMAZON_FULL_LABELS,
        text_block=ex["text"],
        answer=AMAZON_FULL_LABELS[int(ex["label"])],
    )


def _map_imdb(ex):
    return _build_instruction_example(
        task_instruction=PROMPTS["SC"],
        options=BINARY_SENTIMENT,
        text_block=ex["text"],
        answer=BINARY_SENTIMENT[int(ex["label"])],
    )


def _map_sst2(ex):
    return _build_instruction_example(
        task_instruction=PROMPTS["SC"],
        options=BINARY_SENTIMENT,
        text_block=ex["sentence"],
        answer=BINARY_SENTIMENT[int(ex["label"])],
    )


def _map_dbpedia(ex):
    text = _safe_text_join(ex.get("title"), ex.get("content"))
    return _build_instruction_example(
        task_instruction=PROMPTS["TC"],
        options=DBPEDIA_LABELS,
        text_block=text,
        answer=DBPEDIA_LABELS[int(ex["label"])],
    )


def _map_yahoo(ex):
    text = _safe_text_join(
        f"Question Title: {ex.get('question_title', '')}",
        f"Question Content: {ex.get('question_content', '')}",
        f"Best Answer: {ex.get('best_answer', '')}",
    )
    return _build_instruction_example(
        task_instruction=PROMPTS["TC"],
        options=YAHOO_LABELS,
        text_block=text,
        answer=YAHOO_LABELS[int(ex["topic"])],
    )


def _map_mnli(ex):
    text = _safe_text_join(
        f"sentence 1: {ex['premise']}",
        f"sentence 2: {ex['hypothesis']}",
    )
    return _build_instruction_example(
        task_instruction=PROMPTS["NLI"],
        options=NLI_LABELS,
        text_block=text,
        answer=NLI_LABELS[int(ex["label"])],
    )


def _map_cb(ex):
    text = _safe_text_join(
        f"sentence 1: {ex['premise']}",
        f"sentence 2: {ex['hypothesis']}",
    )
    return _build_instruction_example(
        task_instruction=PROMPTS["NLI"],
        options=CB_LABELS,
        text_block=text,
        answer=CB_LABELS[int(ex["label"])],
    )


def _map_rte(ex):
    text = _safe_text_join(
        f"sentence 1: {ex['sentence1']}",
        f"sentence 2: {ex['sentence2']}",
    )
    return _build_instruction_example(
        task_instruction=PROMPTS["NLI"],
        options=RTE_LABELS,
        text_block=text,
        answer=RTE_LABELS[int(ex["label"])],
    )


def _map_qqp(ex):
    text = _safe_text_join(
        f"first sentence: {ex['question1']}",
        f"second sentence: {ex['question2']}",
    )
    return _build_instruction_example(
        task_instruction=PROMPTS["QQP"],
        options=YES_NO_LABELS,
        text_block=text,
        answer=YES_NO_LABELS[int(ex["label"])],
    )


def _map_boolq(ex):
    text = _safe_text_join(
        f"Passage: {ex['passage']}",
        f"Question: {ex['question']}",
    )
    return _build_instruction_example(
        task_instruction=PROMPTS["BoolQA"],
        options=BOOL_LABELS,
        text_block=text,
        answer=BOOL_LABELS[int(ex["label"])],
    )


def _map_multirc(ex):
    text = _safe_text_join(
        f"Passage: {ex['paragraph']}",
        f"Question: {ex['question']}",
        f"Candidate answer: {ex['answer']}",
    )
    return _build_instruction_example(
        task_instruction=PROMPTS["MultiRC"],
        options=BOOL_LABELS,
        text_block=text,
        answer=BOOL_LABELS[int(ex["label"])],
    )


def _map_wic(ex):
    text = _safe_text_join(
        f"Word: {ex['word']}",
        f"Sentence 1: {ex['sentence1']}",
        f"Sentence 2: {ex['sentence2']}",
    )
    return _build_instruction_example(
        task_instruction=PROMPTS["WiC"],
        options=YES_NO_LABELS,
        text_block=text,
        answer=YES_NO_LABELS[int(ex["label"])],
    )


def _map_copa(ex):
    qtype = ex["question"]  # "cause" or "effect"
    text = _safe_text_join(
        f'Which sentence is the {qtype} of "{ex["premise"]}"?',
        f'A: {ex["choice1"]}',
        f'B: {ex["choice2"]}',
    )
    return _build_instruction_example(
        task_instruction="Choose one between A and B.",
        options=["A", "B"],
        text_block=text,
        answer="A" if int(ex["label"]) == 0 else "B",
    )


TASK_MAPPERS = {
    "ag": _map_ag,
    "amazon": _map_amazon,
    "yelp": _map_yelp,
    "imdb": _map_imdb,
    "sst-2": _map_sst2,
    "dbpedia": _map_dbpedia,
    "yahoo": _map_yahoo,
    "mnli": _map_mnli,
    "cb": _map_cb,
    "rte": _map_rte,
    "qqp": _map_qqp,
    "boolqa": _map_boolq,
    "multirc": _map_multirc,
    "wic": _map_wic,
    "copa": _map_copa,
}


# =========================
# Public function 1: SC
# =========================

def prepare_sc_tasks(order_id: int = 1,
                     seed: int = 42,
                     cache_dir: Optional[str] = None,
                     hf_overrides: Optional[Dict[str, Tuple[str, Optional[str]]]] = None,
                     train_total: int = 10000,
                     ) -> OrderedDict[str, TaskData]:
    """
    Standard CL Benchmark (SC):
      - tasks: AG News, Amazon Reviews, DBPedia, Yahoo
      - orders: 1/2/3
      - no repartitioning
      - train: sample INSIDE original train split only, at most train_total
      - use original official test split (or validation if no test split)
      - test remains full (no subsampling)
    """
    if order_id not in SC_ORDERS:
        raise ValueError(f"order_id must be one of {list(SC_ORDERS.keys())}")

    tasks = OrderedDict()
    for offset, task in enumerate(SC_ORDERS[order_id]):
        raw = _load_hf(task, cache_dir=cache_dir, overrides=hf_overrides)

        train_split = _first_available_split(raw, ["train"])
        test_split = _first_available_split(raw, ["test", "validation"])

        if train_split is None:
            raise ValueError(f"{task}: no train split available.")
        if test_split is None:
            raise ValueError(f"{task}: no test/validation split available.")

        train_split = _sample_dataset(train_split, train_total, seed + offset)

        mapper = TASK_MAPPERS[task]
        train_ds = _map_dataset(train_split, mapper)
        test_ds = _map_dataset(test_split, mapper)

        tasks[task] = TaskData(
            name=task,
            train=train_ds,
            test=test_ds,
            validation=None,
            task_type="classification",
            metric="accuracy",
            meta={
                "benchmark": "SC",
                "order_id": order_id,
                "train_max_samples": train_total,
                "seed": seed + offset,
            },
        )
    return tasks


# =========================
# Public function 2: LS
# =========================

def prepare_ls_tasks(order_id: int = 4,
                     seed: int = 42,
                     cache_dir: Optional[str] = None,
                     hf_overrides: Optional[Dict[str, Tuple[str, Optional[str]]]] = None,
                     train_total: int = 1000,
                     train_per_task: Optional[int] = None,
                     test_per_class: int = 500
                     ) -> OrderedDict[str, TaskData]:
    """
    Long Sequence Benchmark (LS), without any repartitioning:
      - 15 tasks
      - orders: 4/5/6
      - train: sample INSIDE original train split only, at most `train_total` per label
      - eval: sample INSIDE original eval split only, at most 500 per class
      - if original split is smaller than target, use the whole available data
      - no split merging, no pool reconstruction
    """
    # LS training now follows per-label sampling.
    # Keep train_per_task for compatibility; when set, it overrides train_total.
    train_per_label = train_total if train_per_task is None else train_per_task

    if order_id not in LS_ORDERS:
        raise ValueError(f"order_id must be one of {list(LS_ORDERS.keys())}")

    tasks = OrderedDict()
    for offset, task in enumerate(LS_ORDERS[order_id]):
        raw = _load_hf(task, cache_dir=cache_dir, overrides=hf_overrides)

        train_split = _first_available_split(raw, ["train"])
        if train_split is None:
            raise ValueError(f"{task}: no train split available.")

        # Prefer labeled validation-like splits first for GLUE/SuperGLUE tasks,
        # then fall back to official test if needed.
        eval_split = _first_available_split(
            raw,
            ["validation", "validation_matched", "validation_mismatched", "test"]
        )
        if eval_split is None:
            raise ValueError(
                f"{task}: no evaluation split found among "
                f"['validation', 'validation_matched', 'validation_mismatched', 'test']"
            )

        label_key = "topic" if task == "yahoo" else "label"

        train_split = _filter_unlabeled_split(train_split, label_key=label_key)
        eval_split = _filter_unlabeled_split(eval_split, label_key=label_key)

        train_split = _sample_dataset_per_class(
            train_split,
            label_key=label_key,
            max_per_class=train_per_label,
            seed=seed + offset,
        )
        eval_split = _sample_dataset_per_class(
            eval_split,
            label_key=label_key,
            max_per_class=test_per_class,
            seed=seed + 1000 + offset,
        )

        mapper = TASK_MAPPERS[task]
        train_ds = _map_dataset(train_split, mapper)
        test_ds = _map_dataset(eval_split, mapper)

        tasks[task] = TaskData(
            name=task,
            train=train_ds,
            test=test_ds,
            validation=None,
            task_type="classification",
            metric="accuracy",
            meta={
                "benchmark": "LS",
                "order_id": order_id,
                "train_max_samples_per_label": train_per_label,
                "train_max_samples_per_task": train_per_label,
                "train_max_samples": train_per_label,
                "test_max_per_class": test_per_class,
                "seed": seed + offset,
            },
        )
    return tasks


# =========================
# Public function 3: TRACE
# =========================

def prepare_trace_tasks(trace_root: str | Path,
                        include_epochs: bool = True
                        ) -> OrderedDict[str, TaskData]:
    """
    TRACE loader:
      - fixed task order: c-stance -> fomc -> meetingbank -> py150 ->
        scienceqa -> numglue-cm -> numglue-ds -> 20minuten
      - expects processed local files:
          trace_root/
            c-stance/{train.json, eval.json, test.json}
            fomc/{train.json, eval.json, test.json}
            ...
      - each json row:
          {"prompt": "...", "answer": "..."}
    """
    root = Path(trace_root)
    if not root.exists():
        raise FileNotFoundError(f"TRACE root not found: {root}")

    tasks = OrderedDict()
    for task in TRACE_ORDER:
        task_dir = root / task
        train_path = task_dir / "train.json"
        eval_path = task_dir / "eval.json"
        test_path = task_dir / "test.json"

        missing = [p.name for p in [train_path, eval_path, test_path] if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{task_dir} missing files: {missing}")

        train_ds = _json_dataset(train_path)
        val_ds = _json_dataset(eval_path)
        test_ds = _json_dataset(test_path)

        for split_name, ds in [("train", train_ds), ("eval", val_ds), ("test", test_ds)]:
            need = {"prompt", "answer"}
            got = set(ds.column_names)
            if not need.issubset(got):
                raise ValueError(f"{task}/{split_name} must contain {need}, got {got}")

        tasks[task] = TaskData(
            name=task,
            train=train_ds,
            validation=val_ds,
            test=test_ds,
            task_type="generative_or_mixed",
            metric=None,
            epochs=TRACE_EPOCHS.get(task) if include_epochs else None,
            meta={"benchmark": "TRACE", "order_id": 7},
        )
    return tasks

