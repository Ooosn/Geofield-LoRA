"""
Pre-compute and cache condition embeddings.

Run this ONCE before training to avoid encoding condition texts
with the backbone during every training step.

Usage:
    python cache_conditions.py --config configs/stage1_gs_condition_v2_qwen3_seed43.yaml

Output:
    runtime/data/condition_embeddings.pt
"""

import os
import argparse
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


DEFAULT_EXTERNAL_INSTRUCTION = (
    "Represent this prompt template by its task intent, expected answer format, and wording style "
    "so that semantically similar templates are close together."
)

DEFAULT_CONDITION_V2_INSTRUCTION = (
    "Represent this task condition by its task intent, answer schema, and few-shot behavioral "
    "signature so that behaviorally similar tasks are close together."
)


def last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    seq_lens = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
    return last_hidden_state[batch_indices, seq_lens]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--encoder_key", default="condition_encoder")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    encoder_key = str(args.encoder_key)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from model.condition_encoder import (
        CachedConditionEmbeddings,
        get_condition_cache_path,
        get_named_condition_cache_path,
        get_condition_source,
    )
    from data.condition_v2 import (
        filter_datasets_cfg_by_templates,
        get_excluded_task_ids,
        get_eligible_templates,
        load_or_build_condition_registry,
        resolve_datasets_cfg,
    )

    cond_cfg = cfg.get(encoder_key, {})
    if encoder_key == "condition_encoder":
        condition_source = get_condition_source(cfg)
    else:
        condition_source = str(cond_cfg.get("source", "backbone_pooled"))

    model = None
    tokenizer = None
    external_model = None
    external_instruction = None
    external_max_length = None

    if condition_source == "backbone_pooled":
        print("Loading model...")
        from model import GSLoRAModel
        model = GSLoRAModel.from_pretrained(
            cfg,
            device_map=cfg.model.get("device_map", None),
            device=device,
        )
        model.eval()
        tokenizer = model.tokenizer
    elif condition_source == "external_embedding":
        model_name = cond_cfg.get("external_model_name")
        if not model_name:
            raise ValueError(f"{encoder_key}.external_model_name is required for external_embedding mode.")
        default_instruction = DEFAULT_EXTERNAL_INSTRUCTION
        if encoder_key == "condition_encoder" and cfg.get("condition_v2", {}).get("enabled", False):
            default_instruction = DEFAULT_CONDITION_V2_INSTRUCTION
        external_instruction = cond_cfg.get("external_instruction", default_instruction)
        external_max_length = int(cond_cfg.get("external_max_length", max(cfg.data.max_condition_length, 512)))

        print(f"Loading external condition encoder: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        external_model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        ).to(device)
        external_model.eval()
    else:
        raise ValueError(f"Unknown {encoder_key}.source={condition_source}")

    print("Collecting unique condition texts...")
    excluded_task_ids = get_excluded_task_ids(cfg)
    datasets_cfg_raw = resolve_datasets_cfg(cfg)
    if excluded_task_ids:
        print(
            f"Excluding {len(excluded_task_ids)} task(s) by default: "
            f"{', '.join(excluded_task_ids)}"
        )
    template_filter_mode = str(cfg.data.get("template_filter_mode", "none"))
    min_valid_templates = int(cfg.data.get("min_valid_templates", 0))
    datasets_cfg, dropped_tasks = filter_datasets_cfg_by_templates(
        datasets_cfg_raw,
        filter_mode=template_filter_mode,
        min_valid_templates=min_valid_templates,
    )
    if dropped_tasks:
        print(
            f"Dropped {len(dropped_tasks)} task(s) with fewer than "
            f"{min_valid_templates} compatible templates"
        )
        for item in dropped_tasks:
            print(
                f"  - {item['id']}: kept={item['eligible_template_count']} "
                f"templates {item['eligible_template_names']}"
            )
    condition_texts: dict[str, str] = {}  # condition_id → condition_text

    if encoder_key == "condition_encoder" and cfg.get("condition_v2", {}).get("enabled", False):
        registry = load_or_build_condition_registry(cfg, datasets_cfg)
        for task_id, record in registry.items():
            condition_texts[f"task::{task_id}"] = record.condition_text
    else:
        for ds_cfg in datasets_cfg:
            ds_id = ds_cfg["id"]
            try:
                templates = get_eligible_templates(
                    ds_cfg,
                    filter_mode=template_filter_mode,
                )
            except Exception as e:
                hf_name = ds_cfg["hf_name"]
                hf_config = ds_cfg.get("hf_config")
                ps_key = f"{hf_name}/{hf_config}" if hf_config else hf_name
                print(f"  WARNING: No compatible templates for {ps_key}: {e}")
                continue

            for tmpl in templates:
                cid = f"{ds_id}__{tmpl.name}"
                condition_texts[cid] = tmpl.jinja if hasattr(tmpl, "jinja") else str(tmpl)

    print(f"Found {len(condition_texts)} unique conditions")

    # Tokenize all condition texts
    cids = list(condition_texts.keys())
    texts = [condition_texts[cid] for cid in cids]

    # Process in batches
    cache = CachedConditionEmbeddings()

    for i in tqdm(range(0, len(cids), args.batch_size), desc="Encoding conditions"):
        batch_cids = cids[i:i + args.batch_size]
        batch_texts = texts[i:i + args.batch_size]

        if condition_source == "backbone_pooled":
            enc = tokenizer(
                batch_texts,
                max_length=cfg.data.max_condition_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            with torch.no_grad():
                pooled = model.encode_conditions_batch(input_ids, attention_mask)
        else:
            prompt_texts = [
                f"Instruct: {external_instruction}\nQuery: {text}"
                for text in batch_texts
            ]
            enc = tokenizer(
                prompt_texts,
                max_length=external_max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                outputs = external_model(**enc)
                pooled = last_token_pool(outputs.last_hidden_state, enc["attention_mask"])
            pooled = pooled.float()

        for cid, emb in zip(batch_cids, pooled):
            cache.set(cid, emb)

    # Save cache
    if encoder_key == "condition_encoder":
        save_path = get_condition_cache_path(cfg)
    else:
        save_path = get_named_condition_cache_path(cfg, encoder_key)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cache.save(save_path)
    print(f"Saved condition embeddings → {save_path}")


if __name__ == "__main__":
    main()
