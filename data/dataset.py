"""
GSLoRADataset: loads PromptSource-Select examples with template-applied text.

Each example in the dataset is:
  {
    "dataset_id":        str   – e.g. "sst2"
    "condition_text":    str   – the prompt template text (used as condition c)
    "condition_id":      str   – unique key for caching: "{dataset_id}__{template_name}"
    "input_text":        str   – template-applied input (template(example).input_target[0])
    "target_text":       str   – gold target string
    "template_name":     str   – name of the template
    "is_held_out":       bool  – True if from a held-out template
    "condition_text_2":  str | None  – second template's condition (for L_cons)
    "condition_id_2":    str | None
  }

Legacy mode uses the raw template TEXT (the Jinja template string) as
condition_text.

Condition V2 mode replaces that template-level condition with a task-level
condition built from:
  - canonical task description
  - answer schema
  - a small fixed set of instantiated support examples

In that mode, all templates for the same task share the same condition_id and
condition_text.

For L_cons, during training we pair each example with a second template's
condition_text. The input+target remain the same.
"""

import os
import random
import hashlib
import json

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Optional, List, Dict, Any, Tuple
from transformers import PreTrainedTokenizer

from .condition_v2 import (
    TaskConditionRecord,
    filter_datasets_cfg_by_templates,
    get_excluded_task_ids,
    get_eligible_templates,
    load_or_build_condition_registry,
    resolve_datasets_cfg,
)
from .condition_v2 import get_template_splits


CACHE_SCHEMA_VERSION = 4
CACHE_NAMESPACE = "strict_v2"


def _make_worker_init_fn(base_seed: int):
    def _seed_worker(worker_id: int):
        worker_seed = int(base_seed) + int(worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _seed_worker


class GSLoRADataset(Dataset):
    """
    Dataset that applies PromptSource templates to HuggingFace datasets.

    Args:
        split:           "train" | "validation" | "test"
        tokenizer:       HuggingFace tokenizer (for length checks)
        data_dir:        directory for caching downloaded HF datasets
        max_cond_len:    max condition text token length
        max_input_len:   max input text token length
        max_target_len:  max target text token length
        use_consistency: if True, also return a second template's condition
                         for the consistency loss
        datasets_cfg:    list of dataset configs to include (defaults to all)
        is_held_out:     if True, only include held-out template examples
        seed:            random seed for template assignment
        cache_dir:       directory for processed-example cache files
    """

    def __init__(
        self,
        split: str,
        tokenizer: PreTrainedTokenizer,
        data_dir: str,
        max_cond_len: int = 256,
        max_input_len: int = 256,
        max_target_len: int = 64,
        use_consistency: bool = True,
        datasets_cfg: Optional[List[Dict]] = None,
        is_held_out: bool = False,
        seed: int = 42,
        max_samples_per_dataset: Optional[int] = None,
        cache_dir: Optional[str] = None,
        n_train_templates: int = 3,
        n_held_out_templates: int = 2,
        condition_mode: str = "template_v1",
        condition_registry: Optional[Dict[str, TaskConditionRecord]] = None,
        template_filter_mode: str = "none",
        cache_ignore_seed: bool = False,
    ):
        self.split = split
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.max_cond_len = max_cond_len
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.condition_mode = condition_mode
        self.condition_registry = condition_registry or {}
        self.template_filter_mode = template_filter_mode
        self.use_consistency = (
            use_consistency
            and not is_held_out
            and condition_mode == "template_v1"
        )
        self.is_held_out = is_held_out
        self.seed = seed
        self.cache_ignore_seed = cache_ignore_seed
        self.n_train_templates = n_train_templates
        self.n_held_out_templates = n_held_out_templates
        self.max_samples_per_dataset = max_samples_per_dataset
        self.rng = random.Random(seed)
        self.cache_dir = cache_dir or os.path.join(data_dir, "cache", "processed_examples")
        os.makedirs(self.cache_dir, exist_ok=True)

        if datasets_cfg is None:
            raise ValueError(
                "GSLoRADataset now requires explicit datasets_cfg. "
                "Please provide an explicit dataset list."
            )
        self.datasets_cfg = datasets_cfg
        self.examples: List[Dict[str, Any]] = []
        self._template_lookup_by_ds: Dict[str, Dict[str, Any]] = {}
        self._train_template_names_by_ds: Dict[str, List[str]] = {}

        self._load_all()

    @staticmethod
    def _template_signature(templates: List[Any]) -> str:
        h = hashlib.sha1()
        for tmpl in templates:
            name = getattr(tmpl, "name", "")
            jinja = getattr(tmpl, "jinja", str(tmpl))
            h.update(f"{name}::{jinja}\n".encode("utf-8"))
        return h.hexdigest()

    def _cache_path(self, ds_id: str, requested_meta: Dict[str, Any]) -> str:
        meta_hash = hashlib.sha1(
            json.dumps(requested_meta, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        key = f"{CACHE_NAMESPACE}__{ds_id}__{meta_hash}"
        return os.path.join(self.cache_dir, f"{key}.pt")

    def _find_compatible_cache(
        self,
        ds_id: str,
        requested_meta: Dict[str, Any],
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        prefix = f"{CACHE_NAMESPACE}__{ds_id}__"
        try:
            filenames = sorted(os.listdir(self.cache_dir))
        except FileNotFoundError:
            return None, None

        for filename in filenames:
            if not filename.startswith(prefix) or not filename.endswith(".pt"):
                continue
            cache_path = os.path.join(self.cache_dir, filename)
            try:
                cached = torch.load(cache_path, map_location="cpu")
            except Exception:
                continue
            if self._matches_requested_metadata(cached.get("metadata", {}), requested_meta):
                return cache_path, cached
        return None, None

    def _requested_metadata(
        self,
        ds_cfg: Dict[str, Any],
        hf_split: str,
        max_samples: int,
        active_templates_sig: Optional[str] = None,
        train_templates_sig: Optional[str] = None,
    ) -> Dict[str, Any]:
        ds_cfg_key = {
            "id": ds_cfg["id"],
            "hf_name": ds_cfg["hf_name"],
            "hf_config": ds_cfg.get("hf_config"),
            "train_split": ds_cfg.get("train_split", "train"),
            "eval_split": ds_cfg.get("eval_split", "validation"),
            "test_split": ds_cfg.get("test_split", "test"),
            "max_samples": ds_cfg.get("max_samples", 2000),
        }
        ds_cfg_hash = hashlib.sha1(
            json.dumps(ds_cfg_key, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return {
            "schema": CACHE_SCHEMA_VERSION,
            "cache_namespace": CACHE_NAMESPACE,
            "split": self.split,
            "is_held_out": self.is_held_out,
            "seed": self.seed,
            "max_samples": max_samples,
            "n_train_templates": self.n_train_templates,
            "n_held_out_templates": self.n_held_out_templates,
            "hf_name": ds_cfg["hf_name"],
            "hf_config": ds_cfg.get("hf_config"),
            "hf_split": hf_split,
            "cache_key": ds_cfg_hash,
            "active_templates_sig": active_templates_sig,
            "train_templates_sig": train_templates_sig,
        }

    def _matches_requested_metadata(
        self,
        cached_meta: Dict[str, Any],
        requested_meta: Dict[str, Any],
    ) -> bool:
        return (
            cached_meta.get("schema") == requested_meta["schema"]
            and cached_meta.get("cache_namespace") == requested_meta["cache_namespace"]
            and cached_meta.get("cache_key") == requested_meta["cache_key"]
            and cached_meta.get("split") == requested_meta["split"]
            and cached_meta.get("is_held_out") == requested_meta["is_held_out"]
            and (
                self.cache_ignore_seed
                or cached_meta.get("seed") == requested_meta["seed"]
            )
            and cached_meta.get("max_samples") == requested_meta["max_samples"]
            and cached_meta.get("n_train_templates") == requested_meta["n_train_templates"]
            and cached_meta.get("n_held_out_templates") == requested_meta["n_held_out_templates"]
            and cached_meta.get("hf_name") == requested_meta["hf_name"]
            and cached_meta.get("hf_config") == requested_meta["hf_config"]
            and cached_meta.get("hf_split") == requested_meta["hf_split"]
            and cached_meta.get("active_templates_sig") == requested_meta["active_templates_sig"]
            and cached_meta.get("train_templates_sig") == requested_meta["train_templates_sig"]
        )

    @staticmethod
    def _metadata_mismatch_fields(
        cached_meta: Dict[str, Any],
        requested_meta: Dict[str, Any],
    ) -> List[str]:
        keys = [
            "schema",
            "cache_namespace",
            "cache_key",
            "split",
            "is_held_out",
            "seed",
            "max_samples",
            "n_train_templates",
            "n_held_out_templates",
            "hf_name",
            "hf_config",
            "hf_split",
            "active_templates_sig",
            "train_templates_sig",
        ]
        mismatches: List[str] = []
        for key in keys:
            if cached_meta.get(key) != requested_meta.get(key):
                mismatches.append(
                    f"{key}: cached={cached_meta.get(key)!r} requested={requested_meta.get(key)!r}"
                )
        return mismatches

    def _task_condition_for_dataset(self, ds_id: str) -> Tuple[str, str]:
        record = self.condition_registry.get(ds_id)
        if record is None:
            raise KeyError(
                f"Condition v2 registry missing task_id={ds_id}. "
                "Build the condition registry before loading the dataset."
            )
        return record.condition_text, f"task::{record.task_id}"

    def _template_condition_for_template(
        self,
        ds_id: str,
        template_name: str,
    ) -> Tuple[str, str]:
        tmpl = self._template_lookup_by_ds[ds_id][template_name]
        condition_text = tmpl.jinja if hasattr(tmpl, "jinja") else str(tmpl)
        return condition_text, f"{ds_id}__{template_name}"

    def _hydrate_examples(
        self,
        ds_id: str,
        base_examples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        hydrated: List[Dict[str, Any]] = []
        train_template_names = self._train_template_names_by_ds.get(ds_id, [])

        for base in base_examples:
            example = dict(base)
            if self.condition_mode == "task_v2":
                condition_text, condition_id = self._task_condition_for_dataset(ds_id)
                example["condition_text"] = condition_text
                example["condition_id"] = condition_id
                example["condition_text_2"] = None
                example["condition_id_2"] = None
            elif self.condition_mode == "task_plus_template_v1":
                task_text, task_id = self._task_condition_for_dataset(ds_id)
                template_text, template_id = self._template_condition_for_template(
                    ds_id=ds_id,
                    template_name=example["template_name"],
                )
                example["condition_text"] = task_text
                example["condition_id"] = task_id
                example["condition_text_2"] = template_text
                example["condition_id_2"] = template_id
            else:
                condition_text, condition_id = self._template_condition_for_template(
                    ds_id=ds_id,
                    template_name=example["template_name"],
                )
                example["condition_text"] = condition_text
                example["condition_id"] = condition_id
                example["condition_text_2"] = None
                example["condition_id_2"] = None

            if (
                self.condition_mode == "template_v1"
                and self.use_consistency
                and len(train_template_names) > 1
                and not example["is_held_out"]
            ):
                other_template_names = [
                    name for name in train_template_names
                    if name != example["template_name"]
                ]
                if other_template_names:
                    tmpl_name_2 = self.rng.choice(other_template_names)
                    cond_text_2, cond_id_2 = self._template_condition_for_template(
                        ds_id=ds_id,
                        template_name=tmpl_name_2,
                    )
                    example["condition_text_2"] = cond_text_2
                    example["condition_id_2"] = cond_id_2

            hydrated.append(example)

        return hydrated

    def _load_all(self):
        import datasets as hf_datasets

        hf_datasets.disable_progress_bar()

        for ds_cfg in self.datasets_cfg:
            ds_id = ds_cfg["id"]
            hf_name = ds_cfg["hf_name"]
            hf_config = ds_cfg.get("hf_config")

            # Determine split name
            if self.split == "train":
                hf_split = ds_cfg.get("train_split", "train")
            elif self.split == "validation":
                hf_split = ds_cfg.get("eval_split", "validation")
            else:
                hf_split = ds_cfg.get("test_split", "test")

            # Cap samples for task balance (training) or fast eval (validation)
            if self.split == "train":
                max_samples = ds_cfg.get("max_samples", 2000)
            else:
                max_samples = self.max_samples_per_dataset or ds_cfg.get("max_samples", 2000)

            # Load PromptSource templates before cache validation so cache hits are
            # invalidated whenever template ordering/splits change.
            try:
                all_templates = get_eligible_templates(
                    ds_cfg,
                    filter_mode=self.template_filter_mode,
                )
            except Exception as e:
                ps_key = f"{hf_name}/{hf_config}" if hf_config else hf_name
                print(f"[Dataset] WARNING: No compatible templates for {ps_key}: {e}")
                all_templates = []

            if len(all_templates) < 2:
                print(f"[Dataset] WARNING: Only {len(all_templates)} templates for {ds_id}, skipping")
                continue

            train_templates, held_out_templates = get_template_splits(
                all_templates,
                n_train=self.n_train_templates,
                n_held_out=self.n_held_out_templates,
            )
            self._template_lookup_by_ds[ds_id] = {
                tmpl.name: tmpl for tmpl in all_templates
            }
            self._train_template_names_by_ds[ds_id] = [
                tmpl.name for tmpl in train_templates
            ]

            if self.is_held_out:
                active_templates = held_out_templates
            else:
                active_templates = train_templates

            if not active_templates:
                continue

            active_templates_sig = self._template_signature(active_templates)
            train_templates_sig = self._template_signature(train_templates)

            # Incremental processed cache (per dataset)
            requested_meta = self._requested_metadata(
                ds_cfg,
                hf_split,
                max_samples,
                active_templates_sig=active_templates_sig,
                train_templates_sig=train_templates_sig,
            )
            cache_path = self._cache_path(ds_id, requested_meta)
            if os.path.exists(cache_path):
                try:
                    cached = torch.load(cache_path, map_location="cpu")
                    if self._matches_requested_metadata(cached.get("metadata", {}), requested_meta):
                        ds_examples = self._hydrate_examples(
                            ds_id,
                            cached.get("examples", []),
                        )
                        self.examples.extend(ds_examples)
                        print(f"[Dataset] Cache hit: {ds_id}, +{len(ds_examples)} examples")
                        continue
                    mismatches = self._metadata_mismatch_fields(
                        cached.get("metadata", {}),
                        requested_meta,
                    )
                    if mismatches:
                        print(f"[Dataset] Cache miss: {ds_id}")
                        for item in mismatches:
                            print(f"  - {item}")
                except Exception as e:
                    print(f"[Dataset] WARNING: cache load failed for {ds_id}: {e}")
            else:
                compat_path, compat_cached = self._find_compatible_cache(ds_id, requested_meta)
                if compat_cached is not None:
                    ds_examples = self._hydrate_examples(
                        ds_id,
                        compat_cached.get("examples", []),
                    )
                    self.examples.extend(ds_examples)
                    print(
                        f"[Dataset] Cache hit: {ds_id}, +{len(ds_examples)} examples "
                        f"(compat: {os.path.basename(compat_path)})"
                    )
                    continue

            # Load HuggingFace dataset only when cache miss requires rebuild.
            try:
                ds_kwargs = {
                    "split": hf_split,
                    "cache_dir": self.data_dir,
                }
                if hf_config is None:
                    ds = hf_datasets.load_dataset(hf_name, **ds_kwargs)
                else:
                    ds = hf_datasets.load_dataset(hf_name, hf_config, **ds_kwargs)
                if isinstance(ds, hf_datasets.DatasetDict):
                    ds = ds[hf_split]
            except Exception as e:
                print(f"[Dataset] WARNING: Failed to load {hf_name}/{hf_config}: {e}")
                continue

            if len(ds) > max_samples:
                ds = ds.shuffle(seed=self.seed).select(range(max_samples))

            metadata = {
                **requested_meta,
                "dataset_fingerprint": getattr(ds, "_fingerprint", None),
            }

            # Apply templates to examples
            base_examples: List[Dict[str, Any]] = []
            for raw_example in ds:
                for tmpl in active_templates:
                    try:
                        applied = tmpl.apply(raw_example)
                        if applied is None:
                            continue
                        input_text, target_text = applied
                        if not input_text or not target_text:
                            continue
                    except Exception:
                        continue

                    example = {
                        "dataset_id": ds_id,
                        "input_text": input_text,
                        "target_text": target_text,
                        "template_name": tmpl.name,
                        "is_held_out": self.is_held_out,
                    }
                    base_examples.append(example)

            ds_examples = self._hydrate_examples(ds_id, base_examples)
            self.examples.extend(ds_examples)
            tmp_cache_path = f"{cache_path}.tmp.{os.getpid()}.{os.environ.get('RANK', '0')}"
            try:
                torch.save({"metadata": metadata, "examples": base_examples}, tmp_cache_path)
                os.replace(tmp_cache_path, cache_path)
                print(f"[Dataset] Cache write: {ds_id}, {len(base_examples)} examples")
            except Exception as e:
                try:
                    if os.path.exists(tmp_cache_path):
                        os.remove(tmp_cache_path)
                except OSError:
                    pass
                print(f"[Dataset] WARNING: cache write failed for {ds_id}: {e}")

        print(
            f"[Dataset] Split={self.split}, held_out={self.is_held_out}, "
            f"examples={len(self.examples)}"
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return dict(self.examples[idx])


class GSLoRACollator:
    """
    Collate GSLoRA examples into batched tensors.

    For each example, returns tokenized:
      - condition_input_ids / condition_attention_mask  (for condition encoder)
      - input_ids / attention_mask                      (for backbone)
      - labels                                          (for LM loss)
      - condition_input_ids_2 / ...                     (optional, for L_cons)
      - condition_ids                                   (str list, for embedding cache)
      - dataset_ids                                     (str list)
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_cond_len: int = 256,
        max_input_len: int = 256,
        max_target_len: int = 64,
        label_pad_id: int = -100,
    ):
        self.tokenizer = tokenizer
        self.max_cond_len = max_cond_len
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.label_pad_id = label_pad_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Tokenize condition texts
        cond_texts = [ex["condition_text"] for ex in batch]
        cond_enc = self.tokenizer(
            cond_texts,
            max_length=self.max_cond_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize input + target as a seq2seq-style pack:
        # input_ids = [condition tokens + input tokens + target tokens]
        # labels    = [-100 * (cond+input length) + target tokens]
        full_input_ids = []
        full_attention_masks = []
        full_labels = []

        for ex in batch:
            # Encode input and target separately
            input_enc = self.tokenizer(
                ex["input_text"],
                max_length=self.max_input_len,
                truncation=True,
                add_special_tokens=True,
            )
            target_enc = self.tokenizer(
                ex["target_text"],
                max_length=self.max_target_len,
                truncation=True,
                add_special_tokens=False,
            )

            # Concatenate: [input tokens] [target tokens] [eos]
            ids = input_enc["input_ids"] + target_enc["input_ids"] + [self.tokenizer.eos_token_id]
            mask = [1] * len(ids)

            # Labels: -100 for input tokens, target tokens for supervision
            input_len = len(input_enc["input_ids"])
            labels = [self.label_pad_id] * input_len + target_enc["input_ids"] + [self.tokenizer.eos_token_id]

            full_input_ids.append(ids)
            full_attention_masks.append(mask)
            full_labels.append(labels)

        # Pad to same length
        max_len = max(len(ids) for ids in full_input_ids)
        max_len = min(max_len, self.max_input_len + self.max_target_len + 1)

        padded_ids = []
        padded_masks = []
        padded_labels = []
        for ids, mask, lbl in zip(full_input_ids, full_attention_masks, full_labels):
            pad_len = max_len - len(ids)
            padded_ids.append(ids[:max_len] + [self.tokenizer.pad_token_id] * max(0, pad_len))
            padded_masks.append(mask[:max_len] + [0] * max(0, pad_len))
            padded_labels.append(lbl[:max_len] + [self.label_pad_id] * max(0, pad_len))

        result = {
            "condition_input_ids": cond_enc["input_ids"],
            "condition_attention_mask": cond_enc["attention_mask"],
            "condition_ids": [ex["condition_id"] for ex in batch],
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "dataset_ids": [ex["dataset_id"] for ex in batch],
            "template_names": [ex["template_name"] for ex in batch],
            "is_held_out": [ex["is_held_out"] for ex in batch],
        }

        # Second template for consistency loss
        has_cond2 = any(ex["condition_text_2"] is not None for ex in batch)
        if has_cond2:
            cond2_texts = [
                ex["condition_text_2"] if ex["condition_text_2"] else ex["condition_text"]
                for ex in batch
            ]
            cond2_enc = self.tokenizer(
                cond2_texts,
                max_length=self.max_cond_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            result["condition_input_ids_2"] = cond2_enc["input_ids"]
            result["condition_attention_mask_2"] = cond2_enc["attention_mask"]
            result["condition_ids_2"] = [ex.get("condition_id_2") for ex in batch]

        return result


def build_dataloaders(
    cfg,
    tokenizer: PreTrainedTokenizer,
    num_workers: int = 0,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    build_eval_loaders: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, validation, and held-out-template dataloaders.

    Returns:
        (train_loader, val_seen_loader, val_heldout_loader)
    """
    collator = GSLoRACollator(
        tokenizer=tokenizer,
        max_cond_len=cfg.data.max_condition_length,
        max_input_len=cfg.data.max_input_length,
        max_target_len=cfg.data.max_target_length,
    )

    n_train_templates = cfg.data.get("n_train_templates", 3)
    n_held_out_templates = cfg.data.get("n_held_out_templates", 2)
    template_filter_mode = str(cfg.data.get("template_filter_mode", "none"))
    min_valid_templates = int(cfg.data.get("min_valid_templates", 0))
    pin_memory = cfg.training.get("pin_memory", True)
    excluded_task_ids = get_excluded_task_ids(cfg)
    datasets_cfg_raw = resolve_datasets_cfg(cfg)
    if excluded_task_ids:
        print(
            f"[Tasks] Excluding {len(excluded_task_ids)} task(s) by default: "
            f"{', '.join(excluded_task_ids)}"
        )
    datasets_cfg, dropped_tasks = filter_datasets_cfg_by_templates(
        datasets_cfg_raw,
        filter_mode=template_filter_mode,
        min_valid_templates=min_valid_templates,
    )
    if dropped_tasks:
        print(
            f"[Templates] Dropped {len(dropped_tasks)} task(s) with fewer than "
            f"{min_valid_templates} compatible templates"
        )
        for item in dropped_tasks:
            print(
                f"  - {item['id']}: kept={item['eligible_template_count']} "
                f"templates {item['eligible_template_names']}"
            )
    if not datasets_cfg:
        raise ValueError("No datasets remain after template compatibility filtering.")
    if cfg.get("template_delta_encoder", {}).get("enabled", False):
        condition_mode = "task_plus_template_v1"
    else:
        condition_mode = "task_v2" if cfg.get("condition_v2", {}).get("enabled", False) else "template_v1"
    condition_registry = (
        load_or_build_condition_registry(cfg, datasets_cfg)
        if condition_mode in {"task_v2", "task_plus_template_v1"}
        else None
    )

    train_ds = GSLoRADataset(
        split="train",
        tokenizer=tokenizer,
        data_dir=cfg.data.data_dir,
        cache_dir=cfg.data.get("cache_dir", os.path.join(cfg.data.data_dir, "cache")),
        max_cond_len=cfg.data.max_condition_length,
        max_input_len=cfg.data.max_input_length,
        max_target_len=cfg.data.max_target_length,
        use_consistency=True,
        seed=cfg.training.seed,
        datasets_cfg=datasets_cfg,
        n_train_templates=n_train_templates,
        n_held_out_templates=n_held_out_templates,
        condition_mode=condition_mode,
        condition_registry=condition_registry,
        template_filter_mode=template_filter_mode,
        cache_ignore_seed=bool(cfg.data.get("cache_ignore_seed", False)),
    )

    train_sampler = None
    train_seed = int(cfg.training.seed)
    train_generator = torch.Generator()
    train_generator.manual_seed(train_seed)
    train_worker_init_fn = _make_worker_init_fn(train_seed)
    if distributed:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
            seed=train_seed,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=train_sampler is None,
        generator=train_generator if train_sampler is None else None,
        worker_init_fn=train_worker_init_fn if num_workers > 0 else None,
    )

    val_seen_loader = None
    val_heldout_loader = None
    if build_eval_loaders:
        # Cap validation samples per dataset to keep eval fast
        max_eval_samples = cfg.data.get("max_eval_samples", 500)
        # Fixed evaluation must not silently drift with training.seed.
        eval_seed = int(cfg.data.get("eval_seed", cfg.training.seed))
        eval_generator = torch.Generator()
        eval_generator.manual_seed(eval_seed)
        eval_worker_init_fn = _make_worker_init_fn(eval_seed)

        val_seen_ds = GSLoRADataset(
            split="validation",
            tokenizer=tokenizer,
            data_dir=cfg.data.data_dir,
            cache_dir=cfg.data.get("cache_dir", os.path.join(cfg.data.data_dir, "cache")),
            max_cond_len=cfg.data.max_condition_length,
            max_input_len=cfg.data.max_input_length,
            max_target_len=cfg.data.max_target_length,
            use_consistency=False,
            is_held_out=False,
            seed=eval_seed,
            max_samples_per_dataset=max_eval_samples,
            datasets_cfg=datasets_cfg,
            n_train_templates=n_train_templates,
            n_held_out_templates=n_held_out_templates,
            condition_mode=condition_mode,
            condition_registry=condition_registry,
            template_filter_mode=template_filter_mode,
            cache_ignore_seed=bool(cfg.data.get("cache_ignore_seed", False)),
        )

        val_heldout_ds = GSLoRADataset(
            split="validation",
            tokenizer=tokenizer,
            data_dir=cfg.data.data_dir,
            cache_dir=cfg.data.get("cache_dir", os.path.join(cfg.data.data_dir, "cache")),
            max_cond_len=cfg.data.max_condition_length,
            max_input_len=cfg.data.max_input_length,
            max_target_len=cfg.data.max_target_length,
            use_consistency=False,
            is_held_out=True,
            seed=eval_seed,
            max_samples_per_dataset=max_eval_samples,
            datasets_cfg=datasets_cfg,
            n_train_templates=n_train_templates,
            n_held_out_templates=n_held_out_templates,
            condition_mode=condition_mode,
            condition_registry=condition_registry,
            template_filter_mode=template_filter_mode,
            cache_ignore_seed=bool(cfg.data.get("cache_ignore_seed", False)),
        )

        val_seen_loader = DataLoader(
            val_seen_ds,
            batch_size=cfg.training.batch_size * 2,
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            generator=eval_generator,
            worker_init_fn=eval_worker_init_fn,
        )

        val_heldout_loader = DataLoader(
            val_heldout_ds,
            batch_size=cfg.training.batch_size * 2,
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            generator=eval_generator,
            worker_init_fn=eval_worker_init_fn,
        )

    return train_loader, val_seen_loader, val_heldout_loader
