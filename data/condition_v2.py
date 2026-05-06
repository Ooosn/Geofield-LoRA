from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import datasets as hf_datasets
from promptsource.templates import DatasetTemplates

_PROMPTSOURCE_TEMPLATE_CACHE: Dict[Tuple[str, Optional[str]], List[Any]] = {}

# Default task exclusions for task-conditioned routing experiments.
DEFAULT_EXCLUDED_TASK_IDS: Tuple[str, ...] = (
    "lambada",
    "scitail__tsv_format",
    "glue__rte",
)


def get_template_splits(
    templates: list,
    n_train: int = 3,
    n_held_out: int = 2,
) -> tuple[list, list]:
    if len(templates) < n_train + n_held_out:
        return templates, []
    return templates[:n_train], templates[-n_held_out:]


def _promptsource_key(dataset_name: str, subset_name: Optional[str]) -> str:
    return f"{dataset_name}/{subset_name}" if subset_name else dataset_name


def _load_promptsource_templates(ds_cfg: Dict[str, Any]) -> List[Any]:
    hf_name = ds_cfg["hf_name"]
    hf_config = ds_cfg.get("hf_config")
    cache_key = (hf_name, hf_config)
    if cache_key not in _PROMPTSOURCE_TEMPLATE_CACHE:
        ps_key = _promptsource_key(hf_name, hf_config)
        prompt_templates = DatasetTemplates(ps_key)
        _PROMPTSOURCE_TEMPLATE_CACHE[cache_key] = list(prompt_templates.templates.values())
    return _PROMPTSOURCE_TEMPLATE_CACHE[cache_key]


def _template_metric_names(template: Any) -> List[str]:
    meta = getattr(template, "metadata", None)
    metrics = getattr(meta, "metrics", None) or []
    return [str(metric).lower() for metric in metrics]


def _metric_compatible(
    task_metric: str,
    template_metric_names: Sequence[str],
) -> bool:
    metric = str(task_metric or "").strip().lower()
    if not metric:
        return True

    if metric == "accuracy":
        return "accuracy" in template_metric_names

    if metric == "rouge":
        return any(name.startswith("rouge") for name in template_metric_names)

    return metric in template_metric_names


def is_template_compatible(
    ds_cfg: Dict[str, Any],
    template: Any,
    filter_mode: str = "none",
) -> bool:
    mode = str(filter_mode or "none").lower()
    if mode in {"", "none", "off", "false"}:
        return True

    task_type = str(ds_cfg.get("task_type", "generation")).lower()
    task_metric = str(ds_cfg.get("metric", "")).lower()
    metric_names = _template_metric_names(template)
    has_answer_choices = getattr(template, "answer_choices", None) is not None

    if mode in {"metric_compatible", "metric_strict", "strict"}:
        if task_metric == "accuracy":
            return has_answer_choices and _metric_compatible(task_metric, metric_names)
        return _metric_compatible(task_metric, metric_names)

    if task_type == "classification":
        return has_answer_choices and ("accuracy" in metric_names)

    return True


def get_eligible_templates(
    ds_cfg: Dict[str, Any],
    filter_mode: str = "none",
) -> List[Any]:
    templates = _load_promptsource_templates(ds_cfg)
    eligible = [
        tmpl
        for tmpl in templates
        if is_template_compatible(ds_cfg, tmpl, filter_mode=filter_mode)
    ]
    return sorted(eligible, key=lambda tmpl: tmpl.name)


def filter_datasets_cfg_by_templates(
    datasets_cfg: Sequence[Dict[str, Any]],
    filter_mode: str = "none",
    min_valid_templates: int = 0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []

    for ds_cfg in datasets_cfg:
        eligible_templates = get_eligible_templates(ds_cfg, filter_mode=filter_mode)
        eligible_names = [tmpl.name for tmpl in eligible_templates]
        if min_valid_templates and len(eligible_templates) < min_valid_templates:
            dropped.append(
                {
                    "id": ds_cfg["id"],
                    "task_type": ds_cfg.get("task_type", "generation"),
                    "eligible_template_count": len(eligible_templates),
                    "eligible_template_names": eligible_names,
                }
            )
            continue

        updated = dict(ds_cfg)
        updated["_eligible_template_names"] = eligible_names
        kept.append(updated)

    return kept, dropped


@dataclass
class SupportExample:
    prompt: str
    answer: str
    template_name: Optional[str] = None


@dataclass
class TaskConditionRecord:
    task_id: str
    dataset_key: str
    hf_name: str
    hf_config: Optional[str]
    task_type: str
    task_description: str
    answer_schema: str
    support_examples: List[SupportExample] = field(default_factory=list)
    condition_text: str = ""
    source: str = "condition_v2"
    template_text_aux: Optional[str] = None


def serialize_condition_text(
    task_description: str,
    answer_schema: str,
    support_examples: Sequence[SupportExample],
    template_text_aux: Optional[str] = None,
) -> str:
    sections = [
        f"Task: {task_description.strip()}",
        f"Answer schema:\n{answer_schema.strip()}",
        "Support examples:",
    ]

    if support_examples:
        for idx, example in enumerate(support_examples, start=1):
            sections.append(
                f"Example {idx}\n"
                f"Prompt: {example.prompt.strip()}\n"
                f"Answer: {example.answer.strip()}"
            )
    else:
        sections.append("No support examples provided.")

    if template_text_aux:
        sections.append(f"Template auxiliary (do not rely on wording alone):\n{template_text_aux.strip()}")
    return "\n\n".join(sections)


def _load_manifest(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def get_excluded_task_ids(cfg) -> List[str]:
    configured = cfg.data.get("excluded_task_ids", None)
    if configured is None:
        return list(DEFAULT_EXCLUDED_TASK_IDS)
    return [str(task_id).strip() for task_id in configured if str(task_id).strip()]


def _apply_task_exclusions(
    datasets_cfg: Sequence[Dict[str, Any]],
    excluded_task_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    excluded = {str(task_id).strip() for task_id in excluded_task_ids if str(task_id).strip()}
    if not excluded:
        return list(datasets_cfg)
    return [ds_cfg for ds_cfg in datasets_cfg if ds_cfg["id"] not in excluded]


def resolve_datasets_cfg(cfg) -> List[Dict[str, Any]]:
    manifest_path = cfg.data.get("dataset_manifest_path")
    if manifest_path:
        records = _load_manifest(manifest_path)
        return _apply_task_exclusions(records, get_excluded_task_ids(cfg))

    configured = cfg.get("promptsource_select", {}).get("datasets")
    if configured:
        from .promptsource_select import DATASET_ID_MAP

        resolved = []
        for entry in configured:
            hf_name = entry.get("hf_name", entry.get("name"))
            hf_config = entry.get("hf_config", entry.get("config"))
            entry_id = entry.get("id")
            if not entry_id:
                normalized = _promptsource_key(hf_name, hf_config).replace("/", "__").replace(".", "_")
                for candidate_id, candidate in DATASET_ID_MAP.items():
                    if candidate.get("hf_name") == hf_name and candidate.get("hf_config") == hf_config:
                        entry_id = candidate_id
                        break
                if not entry_id:
                    entry_id = normalized

            default_meta = DATASET_ID_MAP.get(entry_id, {})
            resolved.append(
                {
                    "id": entry_id,
                    "hf_name": hf_name,
                    "hf_config": hf_config,
                    "task_type": entry.get("task_type", default_meta.get("task_type", "generation")),
                    "metric": entry.get("metric", default_meta.get("metric", "accuracy")),
                    "max_samples": entry.get("max_samples", default_meta.get("max_samples", 2000)),
                    "train_split": entry.get("train_split", default_meta.get("train_split", "train")),
                    "eval_split": entry.get("eval_split", default_meta.get("eval_split", "validation")),
                    "test_split": entry.get("test_split", default_meta.get("test_split", "test")),
                    "support_split": entry.get("support_split", default_meta.get("support_split")),
                    "label_map": entry.get("label_map", default_meta.get("label_map")),
                    "task_description": entry.get("task_description", default_meta.get("task_description")),
                    "answer_schema": entry.get("answer_schema", default_meta.get("answer_schema")),
                    "source": entry.get("source", "condition_v2"),
                }
            )
        return _apply_task_exclusions(resolved, get_excluded_task_ids(cfg))

    raise ValueError(
        "Condition v2 requires either data.dataset_manifest_path or an explicit "
        "promptsource_select.datasets list."
    )


def _default_task_description(ds_cfg: Dict[str, Any]) -> str:
    task_type = ds_cfg.get("task_type", "generation")
    dataset_key = _promptsource_key(ds_cfg["hf_name"], ds_cfg.get("hf_config"))
    pretty = dataset_key.replace("/", " / ").replace("_", " ")
    if task_type == "classification":
        return f"Read the input for {pretty} and predict the correct label."
    return f"Read the input for {pretty} and generate the correct task output."


def _default_answer_schema(ds_cfg: Dict[str, Any], observed_targets: Sequence[str]) -> str:
    label_map = ds_cfg.get("label_map")
    if label_map:
        labels = [str(v) for v in label_map.values()]
        return f"Choose exactly one label from: {', '.join(labels)}."

    unique_targets = []
    seen = set()
    for target in observed_targets:
        target = str(target).strip()
        if not target or target in seen:
            continue
        unique_targets.append(target)
        seen.add(target)
        if len(unique_targets) >= 8:
            break

    short_enough = unique_targets and all(len(target.split()) <= 4 for target in unique_targets)
    if 1 < len(unique_targets) <= 8 and short_enough:
        return f"Choose exactly one output from: {', '.join(unique_targets)}."

    if ds_cfg.get("task_type") == "generation":
        return "Generate the target text directly. Keep the output concise and faithful to the task."
    return "Produce the correct task output."


def _load_split_dataset(ds_cfg: Dict[str, Any], split_name: str, data_dir: str):
    kwargs = {
        "split": split_name,
        "cache_dir": data_dir,
    }
    if ds_cfg.get("hf_config") is None:
        ds = hf_datasets.load_dataset(ds_cfg["hf_name"], **kwargs)
    else:
        ds = hf_datasets.load_dataset(ds_cfg["hf_name"], ds_cfg.get("hf_config"), **kwargs)
    if isinstance(ds, hf_datasets.DatasetDict):
        ds = ds[split_name]
    return ds


def _build_support_examples(
    ds_cfg: Dict[str, Any],
    data_dir: str,
    seed: int,
    num_examples: int,
    num_templates: int,
    n_train_templates: int,
    n_held_out_templates: int,
    template_filter_mode: str = "none",
) -> List[SupportExample]:
    if num_examples <= 0:
        return []

    hf_datasets.disable_progress_bar()
    support_split = ds_cfg.get("support_split") or ds_cfg.get("train_split", "train")
    support_ds = _load_split_dataset(ds_cfg, support_split, data_dir)
    if len(support_ds) == 0:
        return []

    if len(support_ds) > 512:
        support_ds = support_ds.shuffle(seed=seed).select(range(512))

    all_templates = get_eligible_templates(ds_cfg, filter_mode=template_filter_mode)
    train_templates, _ = get_template_splits(
        all_templates,
        n_train=n_train_templates,
        n_held_out=n_held_out_templates,
    )
    support_templates = train_templates[: max(1, num_templates)] if train_templates else all_templates[:1]
    if not support_templates:
        return []

    examples: List[SupportExample] = []
    seen_pairs = set()
    rng = random.Random(seed)
    indices = list(range(len(support_ds)))
    rng.shuffle(indices)

    for raw_idx in indices:
        raw_example = support_ds[raw_idx]
        for tmpl in support_templates:
            try:
                applied = tmpl.apply(raw_example)
                if applied is None:
                    continue
                input_text, target_text = applied
            except Exception:
                continue

            input_text = str(input_text).strip()
            target_text = str(target_text).strip()
            if not input_text or not target_text:
                continue

            pair_key = (input_text, target_text)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            examples.append(
                SupportExample(
                    prompt=input_text,
                    answer=target_text,
                    template_name=getattr(tmpl, "name", None),
                )
            )
            if len(examples) >= num_examples:
                return examples
    return examples


def _build_template_text_aux(
    ds_cfg: Dict[str, Any],
    n_train_templates: int,
    n_held_out_templates: int,
    num_templates: int,
    template_filter_mode: str = "none",
) -> Optional[str]:
    all_templates = get_eligible_templates(ds_cfg, filter_mode=template_filter_mode)
    train_templates, _ = get_template_splits(
        all_templates,
        n_train=n_train_templates,
        n_held_out=n_held_out_templates,
    )
    chosen = train_templates[: max(1, num_templates)]
    aux_blocks = []
    for tmpl in chosen:
        jinja = tmpl.jinja if hasattr(tmpl, "jinja") else str(tmpl)
        aux_blocks.append(f"[{tmpl.name}]\n{jinja.strip()}")
    return "\n\n".join(aux_blocks) if aux_blocks else None


def build_condition_registry(
    datasets_cfg: Sequence[Dict[str, Any]],
    data_dir: str,
    seed: int,
    num_support_examples: int,
    num_support_templates: int,
    n_train_templates: int,
    n_held_out_templates: int,
    include_template_text_aux: bool = False,
    template_filter_mode: str = "none",
) -> Dict[str, TaskConditionRecord]:
    registry: Dict[str, TaskConditionRecord] = {}
    for ds_cfg in datasets_cfg:
        task_id = ds_cfg["id"]
        support_examples = _build_support_examples(
            ds_cfg=ds_cfg,
            data_dir=data_dir,
            seed=seed,
            num_examples=num_support_examples,
            num_templates=num_support_templates,
            n_train_templates=n_train_templates,
            n_held_out_templates=n_held_out_templates,
            template_filter_mode=template_filter_mode,
        )
        observed_targets = [example.answer for example in support_examples]
        task_description = ds_cfg.get("task_description") or _default_task_description(ds_cfg)
        answer_schema = ds_cfg.get("answer_schema") or _default_answer_schema(ds_cfg, observed_targets)
        template_text_aux = None
        if include_template_text_aux:
            try:
                template_text_aux = _build_template_text_aux(
                    ds_cfg=ds_cfg,
                    n_train_templates=n_train_templates,
                    n_held_out_templates=n_held_out_templates,
                    num_templates=num_support_templates,
                    template_filter_mode=template_filter_mode,
                )
            except Exception:
                template_text_aux = None

        condition_text = serialize_condition_text(
            task_description=task_description,
            answer_schema=answer_schema,
            support_examples=support_examples,
            template_text_aux=template_text_aux,
        )
        registry[task_id] = TaskConditionRecord(
            task_id=task_id,
            dataset_key=_promptsource_key(ds_cfg["hf_name"], ds_cfg.get("hf_config")),
            hf_name=ds_cfg["hf_name"],
            hf_config=ds_cfg.get("hf_config"),
            task_type=ds_cfg.get("task_type", "generation"),
            task_description=task_description,
            answer_schema=answer_schema,
            support_examples=support_examples,
            condition_text=condition_text,
            source=str(ds_cfg.get("source", "condition_v2")),
            template_text_aux=template_text_aux,
        )
    return registry


def registry_fingerprint(registry: Dict[str, TaskConditionRecord]) -> str:
    h = hashlib.sha1()
    for task_id in sorted(registry):
        record = registry[task_id]
        h.update(task_id.encode("utf-8"))
        h.update(b"\n")
        h.update(record.condition_text.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def save_condition_registry(registry: Dict[str, TaskConditionRecord], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for task_id in sorted(registry):
            record = registry[task_id]
            payload = asdict(record)
            payload["support_examples"] = [asdict(example) for example in record.support_examples]
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_condition_registry(path: str) -> Dict[str, TaskConditionRecord]:
    registry: Dict[str, TaskConditionRecord] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            payload["support_examples"] = [
                SupportExample(**example) for example in payload.get("support_examples", [])
            ]
            record = TaskConditionRecord(**payload)
            registry[record.task_id] = record
    return registry


def get_condition_v2_registry_path(cfg) -> str:
    cond_cfg = cfg.get("condition_v2", {})
    explicit = cond_cfg.get("registry_path")
    if explicit:
        return explicit
    return os.path.join(cfg.data.data_dir, "condition_registry_v2.jsonl")


def _condition_registry_meta_path(path: str) -> str:
    return f"{path}.meta.json"


def _requested_registry_meta(cfg) -> Dict[str, Any]:
    return {
        "seed": int(cfg.training.seed),
        "num_support_examples": int(cfg.get("condition_v2", {}).get("num_support_examples", 4)),
        "num_support_templates": int(cfg.get("condition_v2", {}).get("num_support_templates", 2)),
        "n_train_templates": int(cfg.data.get("n_train_templates", 3)),
        "n_held_out_templates": int(cfg.data.get("n_held_out_templates", 2)),
        "template_text_aux": bool(cfg.get("condition_v2", {}).get("template_text_aux", False)),
        "template_filter_mode": str(cfg.data.get("template_filter_mode", "none")),
        "min_valid_templates": int(cfg.data.get("min_valid_templates", 0)),
        "dataset_manifest_path": str(cfg.data.get("dataset_manifest_path", "")),
        "excluded_task_ids": sorted(get_excluded_task_ids(cfg)),
    }


def _load_registry_meta(path: str) -> Dict[str, Any] | None:
    meta_path = _condition_registry_meta_path(path)
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_registry_meta(path: str, meta: Dict[str, Any]):
    with open(_condition_registry_meta_path(path), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)


def load_or_build_condition_registry(cfg, datasets_cfg: Sequence[Dict[str, Any]]) -> Dict[str, TaskConditionRecord]:
    path = get_condition_v2_registry_path(cfg)
    force_rebuild = bool(cfg.get("condition_v2", {}).get("force_rebuild_registry", False))
    requested_meta = _requested_registry_meta(cfg)
    cached_meta = _load_registry_meta(path) if os.path.exists(path) else None
    if (
        os.path.exists(path)
        and not force_rebuild
        and cached_meta == requested_meta
    ):
        return load_condition_registry(path)

    registry = build_condition_registry(
        datasets_cfg=datasets_cfg,
        data_dir=cfg.data.data_dir,
        seed=int(cfg.training.seed),
        num_support_examples=int(cfg.get("condition_v2", {}).get("num_support_examples", 4)),
        num_support_templates=int(cfg.get("condition_v2", {}).get("num_support_templates", 2)),
        n_train_templates=int(cfg.data.get("n_train_templates", 3)),
        n_held_out_templates=int(cfg.data.get("n_held_out_templates", 2)),
        include_template_text_aux=bool(cfg.get("condition_v2", {}).get("template_text_aux", False)),
        template_filter_mode=str(cfg.data.get("template_filter_mode", "none")),
    )
    save_condition_registry(registry, path)
    _save_registry_meta(path, requested_meta)
    return registry
