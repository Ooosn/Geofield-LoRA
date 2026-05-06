"""
Evaluation Script.

Evaluates on seen-template and held-out-template splits.
Computes:
  - Per-dataset accuracy / ROUGE-L
  - Task-balanced mean score
  - Held-out-template gap
  - Primitive consistency (alpha distribution similarity across templates)
  - Primitive utilization (entropy, gini, per-primitive usage frequency)

Usage:
    python eval.py --config configs/stage1_gs_condition_v2_qwen3_seed43.yaml --adapter_path path/to/adapter
"""

import os
import sys
import json
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from typing import Optional, Dict, Any

from omegaconf import OmegaConf


# ── Metric helpers ────────────────────────────────────────────────────────────

def load_dataset_metadata(cfg) -> Dict[str, Dict[str, Any]]:
    manifest_meta: Dict[str, Dict[str, Any]] = {}
    manifest_path = cfg.data.get("dataset_manifest_path") if getattr(cfg, "data", None) else None
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                manifest_meta[row["id"]] = row

    from data.promptsource_select import DATASET_ID_MAP

    combined = dict(DATASET_ID_MAP)
    combined.update(manifest_meta)
    return combined

def compute_accuracy(predictions: list[str], references: list[str]) -> float:
    if not predictions:
        return 0.0
    correct = sum(p.strip().lower() == r.strip().lower() for p, r in zip(predictions, references))
    return correct / len(predictions)


def compute_rouge_l(predictions: list[str], references: list[str]) -> float:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [scorer.score(ref, pred)["rougeL"].fmeasure
                  for pred, ref in zip(predictions, references)]
        return float(np.mean(scores)) if scores else 0.0
    except ImportError:
        # Simple token overlap fallback
        scores = []
        for pred, ref in zip(predictions, references):
            pred_toks = set(pred.lower().split())
            ref_toks = set(ref.lower().split())
            if not ref_toks:
                scores.append(1.0 if not pred_toks else 0.0)
                continue
            lcs = len(pred_toks & ref_toks)
            prec = lcs / max(len(pred_toks), 1)
            rec = lcs / max(len(ref_toks), 1)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            scores.append(f1)
        return float(np.mean(scores)) if scores else 0.0


def get_metric_fn(task_type: str):
    if task_type == "classification":
        return compute_accuracy
    elif task_type == "generation":
        return compute_rouge_l
    else:
        return compute_accuracy


def decode_predictions(logits: torch.Tensor, tokenizer, labels: torch.Tensor) -> tuple[list[str], list[str]]:
    """
    Extract prediction strings and reference strings from logits and labels.

    Causal LM offset: logits[t] predicts token at position t+1.
    So to get the predicted token for label position t, we use logits[t-1].
    """
    # Shift: logits[:, :-1] predicts labels[:, 1:]
    shifted_logits = logits[:, :-1, :]          # (B, seq-1, vocab)
    shifted_labels = labels[:, 1:]              # (B, seq-1)
    pred_ids = shifted_logits.argmax(dim=-1)    # (B, seq-1)

    predictions, references = [], []

    for i in range(pred_ids.shape[0]):
        target_mask = shifted_labels[i] != -100
        if not target_mask.any():
            predictions.append("")
            references.append("")
            continue

        pred_toks = pred_ids[i][target_mask]
        ref_toks = shifted_labels[i][target_mask]

        pred_str = tokenizer.decode(pred_toks, skip_special_tokens=True)
        ref_str = tokenizer.decode(ref_toks, skip_special_tokens=True)
        predictions.append(pred_str)
        references.append(ref_str)

    return predictions, references


def _as_torch_device(candidate):
    if candidate is None:
        return None
    if isinstance(candidate, torch.device):
        return candidate
    if isinstance(candidate, int):
        return torch.device(f"cuda:{candidate}")
    if isinstance(candidate, str):
        try:
            return torch.device(candidate)
        except (RuntimeError, TypeError):
            return None
    return None


def _find_adapter_model_config(adapter_path: str) -> Optional[str]:
    adapter_dir = Path(adapter_path).resolve()
    if not adapter_dir.exists():
        return None

    direct_cfg = adapter_dir / "config.yaml"
    if direct_cfg.is_file():
        return str(direct_cfg)

    sibling_cfg = adapter_dir.parent.with_suffix(".config.yaml")
    if sibling_cfg.is_file():
        return str(sibling_cfg)

    return None


def _merge_model_side_cfg(eval_cfg, adapter_cfg):
    merged = OmegaConf.create(OmegaConf.to_container(eval_cfg, resolve=False))

    for section in ("model", "gslora", "template_delta_encoder"):
        if section in adapter_cfg:
            merged[section] = OmegaConf.create(
                OmegaConf.to_container(adapter_cfg[section], resolve=False)
            )

    if "condition_encoder" in adapter_cfg:
        cond_cfg = OmegaConf.create(
            OmegaConf.to_container(merged.get("condition_encoder", {}), resolve=False)
        )
        adapter_cond_cfg = adapter_cfg["condition_encoder"]
        # Keep eval-side cache routing on the target dataset, but inherit
        # adapter-side model behavior such as head type and dimensions.
        for key, value in adapter_cond_cfg.items():
            if key in {"cache_name", "cache_path"}:
                continue
            cond_cfg[key] = value
        merged["condition_encoder"] = cond_cfg

    if "training" in adapter_cfg and "model_type" in adapter_cfg.training:
        merged.training.model_type = adapter_cfg.training.model_type

    return merged


def _resolve_runtime_cfg(eval_cfg, adapter_path: str) -> tuple[Any, Optional[str]]:
    cfg_path = _find_adapter_model_config(adapter_path)
    if cfg_path is None:
        return eval_cfg, None

    adapter_cfg = OmegaConf.load(cfg_path)
    merged_cfg = _merge_model_side_cfg(eval_cfg, adapter_cfg)
    return merged_cfg, cfg_path


# ── Main evaluation function ──────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model,
    dataloader,
    cache,
    cfg,
    device: torch.device,
    dtype: torch.dtype,
    split: str = "seen",
    max_batches: Optional[int] = None,
    template_cache=None,
) -> Dict[str, Any]:
    """
    Run evaluation over a dataloader.

    Returns:
        metrics dict with:
          - per_dataset_scores: {dataset_id: score}
          - mean_score: task-balanced mean
          - primitive_stats: entropy, gini, consistency, etc.
    """
    from model.condition_encoder import get_condition_source
    dataset_meta = load_dataset_metadata(cfg)

    # Accumulators
    dataset_preds: Dict[str, list] = defaultdict(list)
    dataset_refs: Dict[str, list] = defaultdict(list)
    all_alphas: Dict[str, list] = defaultdict(list)   # condition_id → list of alpha

    # For primitive consistency: group by (dataset_id, input_text) to compare
    # alpha across templates. We use condition_id as a proxy per-template.
    template_alphas: Dict[str, torch.Tensor] = {}  # condition_id → mean alpha over batch
    total_batches = len(dataloader)
    total_examples = len(dataloader.dataset)
    batch_size = getattr(dataloader, "batch_size", None)
    interactive = sys.stderr.isatty()

    print(
        f"[Eval {split}] examples={total_examples:,}, batches={total_batches:,}, "
        f"batch_size={batch_size}"
    )

    for batch_idx, batch in enumerate(
        tqdm(
            dataloader,
            desc=f"Eval [{split}]",
            total=total_batches,
            unit="batch",
            dynamic_ncols=True,
            disable=not interactive,
        )
    ):
        if max_batches and batch_idx >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        # Get condition embeddings
        cids = batch["condition_ids"]
        if all(cache.has(cid) for cid in cids):
            cond_h = cache.get_batch(cids, device).to(dtype)
        else:
            if get_condition_source(cfg) != "backbone_pooled":
                missing = [cid for cid in cids if not cache.has(cid)]
                raise RuntimeError(
                    "External condition encoder mode requires a complete condition cache during eval. "
                    f"Missing cached condition ids: {missing[:4]}"
                )
            cond_h = model.encode_conditions_batch(
                batch["condition_input_ids"].to(device),
                batch["condition_attention_mask"].to(device),
            ).to(dtype)

        cond_h_2 = None
        if "condition_ids_2" in batch and any(c is not None for c in batch["condition_ids_2"]):
            cids_2 = batch["condition_ids_2"]
            secondary_cache = template_cache if template_cache is not None else cache
            if all(c is not None and secondary_cache.has(c) for c in cids_2):
                cond_h_2 = secondary_cache.get_batch(cids_2, device).to(dtype)
            elif "condition_input_ids_2" in batch:
                if get_condition_source(cfg) != "backbone_pooled":
                    missing = [cid for cid in cids_2 if cid is not None and not secondary_cache.has(cid)]
                    raise RuntimeError(
                        "External condition encoder mode requires a complete secondary condition cache during eval. "
                        f"Missing cached condition ids: {missing[:4]}"
                    )
                cond_h_2 = model.encode_conditions_batch(
                    batch["condition_input_ids_2"].to(device),
                    batch["condition_attention_mask_2"].to(device),
                ).to(dtype)

        # Forward
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            condition_hidden=cond_h,
            task_ids=batch["dataset_ids"],
            condition_hidden_2=cond_h_2,
            labels=labels,
            return_logits=True,
        )

        logits = output["logits"]    # (B, seq, vocab)
        alpha = output["alpha"]      # (B, K)
        layer_alphas = output.get("layer_alphas")

        # Decode predictions
        preds, refs = decode_predictions(logits, model.tokenizer, labels)

        # Accumulate by dataset
        for i, ds_id in enumerate(batch["dataset_ids"]):
            dataset_preds[ds_id].append(preds[i])
            dataset_refs[ds_id].append(refs[i])

        # Accumulate alphas by condition_id
        for i, cid in enumerate(cids):
            alpha_for_stats = layer_alphas[i] if layer_alphas is not None else alpha[i]
            all_alphas[cid].append(alpha_for_stats.cpu())

        # 与 pilot_gslora_*.log 一致：非 TTY 时 tqdm 关闭，用少量 batch 行表示进度
        if not interactive and ((batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches):
            print(f"[Eval {split}] batch={batch_idx + 1}/{total_batches}")

    # ── Compute per-dataset scores ────────────────────────────────────────
    per_dataset_scores = {}
    for ds_id in dataset_preds:
        ds_cfg = dataset_meta.get(ds_id, {})
        task_type = ds_cfg.get("task_type", "classification")
        metric_fn = get_metric_fn(task_type)
        score = metric_fn(dataset_preds[ds_id], dataset_refs[ds_id])
        per_dataset_scores[ds_id] = score

    mean_score = float(np.mean(list(per_dataset_scores.values()))) if per_dataset_scores else 0.0

    # ── Primitive statistics ──────────────────────────────────────────────
    # Average alpha per condition_id
    cid_mean_alpha: Dict[str, torch.Tensor] = {}
    for cid, alpha_list in all_alphas.items():
        cid_mean_alpha[cid] = torch.stack(alpha_list).mean(dim=0)

    if cid_mean_alpha:
        all_mean_alphas = torch.stack(list(cid_mean_alpha.values()))
        alpha_flat = all_mean_alphas.reshape(-1, all_mean_alphas.shape[-1])

        # Entropy
        entropy = -(alpha_flat * (alpha_flat + 1e-8).log()).sum(dim=-1)
        mean_entropy = entropy.mean().item()

        # Global average utilization
        global_avg = alpha_flat.mean(dim=0)   # (K,)
        sorted_alpha, _ = global_avg.sort()
        K = sorted_alpha.shape[0]
        gini_num = (2 * torch.arange(1, K + 1).float() - K - 1) * sorted_alpha
        gini = (gini_num.sum() / (K * sorted_alpha.sum() + 1e-8)).item()

        primitive_stats = {
            "mean_entropy": mean_entropy,
            "gini": gini,
            "avg_utilization": global_avg.tolist(),
        }
    else:
        primitive_stats = {}

    return {
        "per_dataset_scores": per_dataset_scores,
        "mean_score": mean_score,
        "primitive_stats": primitive_stats,
        "split": split,
        "n_datasets": len(per_dataset_scores),
    }


@torch.no_grad()
def evaluate_shared(
    model,
    tokenizer,
    dataloader,
    cfg,
    device: torch.device,
    split: str = "seen",
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    dataset_meta = load_dataset_metadata(cfg)
    dataset_preds: Dict[str, list] = defaultdict(list)
    dataset_refs: Dict[str, list] = defaultdict(list)
    total_batches = len(dataloader)
    total_examples = len(dataloader.dataset)
    batch_size = getattr(dataloader, "batch_size", None)
    interactive = sys.stderr.isatty()

    print(
        f"[Eval {split}] examples={total_examples:,}, batches={total_batches:,}, "
        f"batch_size={batch_size}"
    )

    for batch_idx, batch in enumerate(
        tqdm(
            dataloader,
            desc=f"Eval [{split}]",
            total=total_batches,
            unit="batch",
            dynamic_ncols=True,
            disable=not interactive,
        )
    ):
        if max_batches and batch_idx >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_logits=True,
            use_cache=False,
        )
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        preds, refs = decode_predictions(logits, tokenizer, labels)
        for i, ds_id in enumerate(batch["dataset_ids"]):
            dataset_preds[ds_id].append(preds[i])
            dataset_refs[ds_id].append(refs[i])

        if not interactive and ((batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches):
            print(f"[Eval {split}] batch={batch_idx + 1}/{total_batches}")

    per_dataset_scores = {}
    for ds_id in dataset_preds:
        ds_cfg = dataset_meta.get(ds_id, {})
        task_type = ds_cfg.get("task_type", "classification")
        metric_fn = get_metric_fn(task_type)
        per_dataset_scores[ds_id] = metric_fn(dataset_preds[ds_id], dataset_refs[ds_id])

    mean_score = float(np.mean(list(per_dataset_scores.values()))) if per_dataset_scores else 0.0
    return {
        "per_dataset_scores": per_dataset_scores,
        "mean_score": mean_score,
        "primitive_stats": {},
        "split": split,
        "n_datasets": len(per_dataset_scores),
    }


# ── Standalone evaluation ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate adapter")
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter_path", required=True, help="Path to adapter checkpoint")
    parser.add_argument("--split", default="both", choices=["seen", "held_out", "both"])
    parser.add_argument("--model_type", default=None, choices=["gs", "mlp", "mixlora", "shared", "base"])
    parser.add_argument("--eval_seed", type=int, default=None, help="Override the eval template/data split seed")
    parser.add_argument("--output_file", default=None, help="JSON output path")
    args = parser.parse_args()

    eval_cfg = OmegaConf.load(args.config)
    if args.eval_seed is not None:
        eval_cfg.data.eval_seed = int(args.eval_seed)
    model_type = args.model_type or str(eval_cfg.training.get("model_type", "gs")).strip().lower()
    cfg, resolved_model_cfg_path = _resolve_runtime_cfg(eval_cfg, args.adapter_path)
    if args.model_type is None:
        model_type = str(cfg.training.get("model_type", model_type)).strip().lower()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if cfg.training.bf16 else torch.float32

    print("Loading model and adapter...")
    if resolved_model_cfg_path is not None:
        print(
            "[Eval] Using adapter-side model settings from "
            f"{resolved_model_cfg_path} while preserving eval data/cache settings from "
            f"{os.path.abspath(args.config)}"
        )
    tokenizer = None
    if model_type == "gs":
        from model import GSLoRAModel
        model = GSLoRAModel.from_pretrained(
            cfg,
            device_map=cfg.model.get("device_map", None),
            device=device,
        )
        tokenizer = model.tokenizer
    elif model_type == "mlp":
        from model import MLPGatedLoRAModel
        model = MLPGatedLoRAModel.from_pretrained(
            cfg,
            device_map=cfg.model.get("device_map", None),
            device=device,
        )
        tokenizer = model.tokenizer
    elif model_type == "mixlora":
        from model import MixLoRAModel
        model = MixLoRAModel.from_pretrained(
            cfg,
            device_map=cfg.model.get("device_map", None),
            device=device,
        )
        tokenizer = model.tokenizer
    elif model_type == "shared":
        from model import SharedLoRAModel

        model = SharedLoRAModel.from_pretrained(
            cfg,
            device_map=cfg.model.get("device_map", None),
            device=device,
            attach_trainable_lora=False,
        )
        tokenizer = model.tokenizer
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.model.backbone, use_fast=True)
        base = AutoModelForCausalLM.from_pretrained(
            cfg.model.backbone,
            torch_dtype=dtype,
            device_map=cfg.model.get("device_map", None) or "auto",
        )

        class BaseModelWrapper(torch.nn.Module):
            def __init__(self, inner, tok):
                super().__init__()
                self.inner = inner
                self.tokenizer = tok

            def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
                output = self.inner(input_ids=input_ids, attention_mask=attention_mask)
                logits = output.logits
                alpha = torch.ones(
                    (input_ids.shape[0], 1),
                    device=logits.device,
                    dtype=logits.dtype,
                )
                return {"logits": logits, "alpha": alpha}

        model = BaseModelWrapper(base, tokenizer)

    if model_type in {"gs", "mlp", "mixlora", "shared"}:
        model.load_adapter(args.adapter_path)
    model.eval()

    print("Building dataloaders...")
    from data import build_dataloaders
    num_workers = cfg.training.get("num_workers", 0)
    _, val_seen_loader, val_heldout_loader = build_dataloaders(
        eval_cfg, tokenizer, num_workers=num_workers
    )

    results = {}

    if args.split in ("seen", "both"):
        print("\nEvaluating on SEEN templates...")
        if model_type == "shared":
            seen_metrics = evaluate_shared(model, tokenizer, val_seen_loader, eval_cfg, device, split="seen")
        else:
            print("Loading condition cache...")
            from model.condition_encoder import (
                CachedConditionEmbeddings,
                get_condition_cache_path,
                get_named_condition_cache_path,
            )
            cache_path = get_condition_cache_path(eval_cfg)
            cache = CachedConditionEmbeddings.load(cache_path)
            template_cache = None
            if bool(cfg.get("template_delta_encoder", {}).get("enabled", False)):
                template_cache = CachedConditionEmbeddings.load(
                    get_named_condition_cache_path(eval_cfg, "template_delta_encoder")
                )
            seen_metrics = evaluate(
                model,
                val_seen_loader,
                cache,
                cfg,
                device,
                dtype,
                split="seen",
                template_cache=template_cache,
            )
        results["seen"] = seen_metrics
        print(f"Seen template mean score: {seen_metrics['mean_score']:.4f}")
        print("Per-dataset scores:")
        for ds_id, score in sorted(seen_metrics["per_dataset_scores"].items()):
            print(f"  {ds_id:20s}: {score:.4f}")

    if args.split in ("held_out", "both"):
        print("\nEvaluating on HELD-OUT templates...")
        if model_type == "shared":
            held_metrics = evaluate_shared(model, tokenizer, val_heldout_loader, eval_cfg, device, split="held_out")
        else:
            if "cache" not in locals():
                print("Loading condition cache...")
                from model.condition_encoder import (
                    CachedConditionEmbeddings,
                    get_condition_cache_path,
                    get_named_condition_cache_path,
                )
                cache_path = get_condition_cache_path(eval_cfg)
                cache = CachedConditionEmbeddings.load(cache_path)
                template_cache = None
                if bool(cfg.get("template_delta_encoder", {}).get("enabled", False)):
                    template_cache = CachedConditionEmbeddings.load(
                        get_named_condition_cache_path(eval_cfg, "template_delta_encoder")
                    )
            held_metrics = evaluate(
                model,
                val_heldout_loader,
                cache,
                cfg,
                device,
                dtype,
                split="held_out",
                template_cache=template_cache,
            )
        results["held_out"] = held_metrics
        print(f"Held-out template mean score: {held_metrics['mean_score']:.4f}")
        print("Per-dataset scores:")
        for ds_id, score in sorted(held_metrics["per_dataset_scores"].items()):
            print(f"  {ds_id:20s}: {score:.4f}")

    if args.split == "both" and "seen" in results and "held_out" in results:
        gap = results["seen"]["mean_score"] - results["held_out"]["mean_score"]
        results["template_gap"] = gap
        print(f"\nHeld-out template gap: {gap:.4f}")
        print("(Smaller gap = better template robustness)")

    results["metadata"] = {
        "config_path": os.path.abspath(args.config),
        "resolved_model_config_path": (
            os.path.abspath(resolved_model_cfg_path) if resolved_model_cfg_path else None
        ),
        "adapter_path": os.path.abspath(args.adapter_path),
        "model_type": model_type,
        "eval_seed": int(eval_cfg.data.get("eval_seed", eval_cfg.training.seed)),
        "training_seed": int(cfg.training.seed),
        "n_train_templates": int(eval_cfg.data.get("n_train_templates", 3)),
        "n_held_out_templates": int(eval_cfg.data.get("n_held_out_templates", 2)),
        "max_eval_samples": int(eval_cfg.data.get("max_eval_samples", 500)),
        "dataset_manifest_path": str(eval_cfg.data.get("dataset_manifest_path", "")),
    }

    # Print primitive stats
    if "seen" in results and results["seen"].get("primitive_stats"):
        ps = results["seen"]["primitive_stats"]
        print(f"\nPrimitive stats (seen):")
        print(f"  Mean entropy:   {ps.get('mean_entropy', 0):.3f}")
        print(f"  Gini coeff:     {ps.get('gini', 0):.3f}")
        print(f"  Utilization:    {[f'{v:.3f}' for v in ps.get('avg_utilization', [])]}")

    # Save results
    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")

    return results


if __name__ == "__main__":
    main()
