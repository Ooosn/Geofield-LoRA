"""
Geofield-LoRA Training Script.

Usage:
    python train.py --config configs/stage1_gs_condition_v2_qwen3_seed43.yaml [--resume checkpoint/path]

Training loop:
  1. Load pre-computed condition embeddings from cache (fast, no backbone re-forward)
  2. For each batch:
     a. Look up condition_hidden from cache
     b. Compute per-layer alpha_l = PrimitiveBank_l(ConditionEncoder(condition_hidden))
     c. Forward backbone with Geofield-LoRA delta weighted by alpha_l
     d. L_task + structural regularizers
  3. Evaluate on seen and held-out templates every eval_steps
  4. Save adapter (not backbone) checkpoints
"""

import os
import sys
import argparse
import math
import random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from model import GSLoRAModel, MLPGatedLoRAModel, MixLoRAModel, SharedLoRAModel
from model.condition_encoder import (
    CachedConditionEmbeddings,
    get_condition_cache_path,
    get_named_condition_cache_path,
    get_condition_source,
)
from data import build_dataloaders
from data.condition_v2 import filter_datasets_cfg_by_templates, resolve_datasets_cfg
from losses import compute_structural_losses
from eval import evaluate, evaluate_shared


def parse_args():
    parser = argparse.ArgumentParser(description="Train Geofield-LoRA")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="Path to adapter checkpoint to resume")
    parser.add_argument("--run_name", default=None, help="Experiment name for logs. Defaults to the config stem.")
    parser.add_argument("--local_rank", type=int, default=-1, help="torchrun local rank")
    parser.add_argument("--local-rank", dest="local_rank", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def infer_model_type(cfg) -> str:
    model_type = str(cfg.training.get("model_type", "")).strip().lower()
    if model_type:
        return model_type
    if cfg.get("mlp_gated") is not None:
        return "mlp"
    return "gs"


def get_active_task_ids(cfg) -> list[str]:
    datasets_cfg_raw = resolve_datasets_cfg(cfg)
    template_filter_mode = str(cfg.data.get("template_filter_mode", "none"))
    min_valid_templates = int(cfg.data.get("min_valid_templates", 0))
    datasets_cfg, _ = filter_datasets_cfg_by_templates(
        datasets_cfg_raw,
        filter_mode=template_filter_mode,
        min_valid_templates=min_valid_templates,
    )
    return [str(ds_cfg["id"]) for ds_cfg in datasets_cfg]


def reduce_mean(value: float, device: torch.device) -> float:
    if not is_distributed():
        return value
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return tensor.item()


def load_condition_cache(cfg) -> CachedConditionEmbeddings:
    cache_path = get_condition_cache_path(cfg)
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Condition embedding cache not found at {cache_path}. "
            "Run `python cache_conditions.py` first."
        )
    print(f"[Train] Loading condition cache from {cache_path}")
    return CachedConditionEmbeddings.load(cache_path)


def load_template_delta_cache(cfg) -> CachedConditionEmbeddings | None:
    if not bool(cfg.get("template_delta_encoder", {}).get("enabled", False)):
        return None
    cache_path = get_named_condition_cache_path(cfg, "template_delta_encoder")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Template-delta embedding cache not found at {cache_path}. "
            "Build it before training this stage."
        )
    print(f"[Train] Loading template-delta cache from {cache_path}")
    return CachedConditionEmbeddings.load(cache_path)


def set_condition_encoder_trainable(model_module: GSLoRAModel, enabled: bool):
    for param in model_module.condition_encoder.parameters():
        param.requires_grad = enabled


def set_condition_encoder_base_trainable(model_module: GSLoRAModel, enabled: bool):
    for name, param in model_module.condition_encoder.named_parameters():
        if name.startswith("task_embeddings."):
            continue
        if name.startswith("task_radius_offsets."):
            continue
        param.requires_grad = enabled


def set_task_view_embeddings_trainable(model_module: GSLoRAModel, enabled: bool):
    task_embeddings = getattr(model_module.condition_encoder, "task_embeddings", None)
    if task_embeddings is None:
        return
    for param in task_embeddings.parameters():
        param.requires_grad = enabled


def set_task_radii_trainable(model_module: GSLoRAModel, enabled: bool):
    task_radius_offsets = getattr(model_module.condition_encoder, "task_radius_offsets", None)
    if task_radius_offsets is None:
        return
    for param in task_radius_offsets.parameters():
        param.requires_grad = enabled


def _iter_routing_banks(model_module: GSLoRAModel):
    primitive_banks = getattr(model_module, "primitive_banks", None)
    if primitive_banks is not None:
        yield from primitive_banks.values()
    router_banks = getattr(model_module, "router_banks", None)
    if router_banks is not None:
        yield from router_banks.values()


def set_primitive_bank_base_trainable(model_module: GSLoRAModel, enabled: bool):
    for primitive_bank in _iter_routing_banks(model_module):
        for name, param in primitive_bank.named_parameters():
            if name == "lower_offdiag":
                continue
            param.requires_grad = enabled


def set_covariance_offdiag_trainable(model_module: GSLoRAModel, enabled: bool):
    for primitive_bank in _iter_routing_banks(model_module):
        lower_offdiag = getattr(primitive_bank, "lower_offdiag", None)
        if lower_offdiag is not None:
            lower_offdiag.requires_grad = enabled


def set_template_side_trainable(model_module: GSLoRAModel, enabled: bool):
    if model_module.template_delta_encoder is not None:
        for param in model_module.template_delta_encoder.parameters():
            param.requires_grad = enabled
    if model_module.template_bias_mlp is not None:
        for param in model_module.template_bias_mlp.parameters():
            param.requires_grad = enabled
    if model_module.template_gamma is not None:
        model_module.template_gamma.requires_grad = enabled


def set_task_side_trainable(model_module: GSLoRAModel, enabled: bool):
    for param in model_module.condition_encoder.parameters():
        param.requires_grad = enabled
    set_primitive_bank_base_trainable(model_module, enabled)
    for param in model_module.gs_lora_list.parameters():
        param.requires_grad = enabled


def build_optimizer(raw_model: GSLoRAModel, cfg):
    base_lr = float(cfg.training.learning_rate)
    cond_lr_scale = float(cfg.training.get("condition_encoder_lr_scale", 1.0))
    task_view_lr_scale = float(
        cfg.training.get("task_view_embedding_lr_scale", cond_lr_scale)
    )
    task_radius_lr_scale = float(
        cfg.training.get("task_radius_lr_scale", cond_lr_scale)
    )
    primitive_lr_scale = float(cfg.training.get("primitive_bank_lr_scale", 1.0))
    primitive_mu_lr_scale = float(cfg.training.get("primitive_mu_lr_scale", primitive_lr_scale))
    primitive_logdiag_lr_scale = float(
        cfg.training.get("primitive_logdiag_lr_scale", primitive_lr_scale)
    )
    primitive_log_s_lr_scale = float(
        cfg.training.get("primitive_log_s_lr_scale", primitive_lr_scale)
    )
    primitive_ray_origin_lr_scale = float(
        cfg.training.get("primitive_ray_origin_lr_scale", primitive_lr_scale)
    )
    covariance_offdiag_lr_scale = float(
        cfg.training.get("covariance_offdiag_lr_scale", primitive_lr_scale)
    )
    lora_atom_lr_scale = float(cfg.training.get("lora_atom_lr_scale", 1.0))
    condition_weight_decay = cfg.training.get("condition_encoder_weight_decay", None)
    primitive_weight_decay = cfg.training.get("primitive_bank_weight_decay", None)
    named_params = [
        (name, param)
        for name, param in raw_model.named_parameters()
        if (not name.startswith("backbone.")) or param.requires_grad
    ]
    task_view_params = [
        param for name, param in named_params
        if name.startswith("condition_encoder.task_embeddings.")
    ]
    task_radius_params = [
        param for name, param in named_params
        if name.startswith("condition_encoder.task_radius_offsets.")
    ]
    cond_params = [
        param for name, param in named_params
        if name.startswith("condition_encoder.")
        and not name.startswith("condition_encoder.task_embeddings.")
        and not name.startswith("condition_encoder.task_radius_offsets.")
    ]
    primitive_param_groups = {
        "mu": [],
        "logdiag": [],
        "logs": [],
        "offdiag": [],
        "origin": [],
        "primitive_other": [],
        "router_banks": [],
    }
    for name, param in named_params:
        if name.startswith("primitive_banks."):
            bucket = _geom_param_bucket(name) or "primitive_other"
            primitive_param_groups.setdefault(bucket, []).append(param)
        elif name.startswith("router_banks."):
            primitive_param_groups["router_banks"].append(param)
    lora_atom_params = [
        param for name, param in named_params
        if _geom_param_bucket(name) in {"lora_U", "lora_V"}
    ]
    other_params = [
        param for name, param in named_params
        if not name.startswith("condition_encoder.")
        and not name.startswith("primitive_banks.")
        and not name.startswith("router_banks.")
        and _geom_param_bucket(name) not in {"lora_U", "lora_V"}
    ]

    param_groups = []
    if other_params:
        param_groups.append({
            "params": other_params,
            "lr": base_lr,
            "group_name": "default",
        })
    if cond_params:
        group = {
            "params": cond_params,
            "lr": base_lr * cond_lr_scale,
            "group_name": "condition_encoder",
        }
        if condition_weight_decay is not None:
            group["weight_decay"] = float(condition_weight_decay)
        param_groups.append(group)
    if task_view_params:
        param_groups.append({
            "params": task_view_params,
            "lr": base_lr * task_view_lr_scale,
            "group_name": "task_view_embeddings",
        })
    if task_radius_params:
        param_groups.append({
            "params": task_radius_params,
            "lr": base_lr * task_radius_lr_scale,
            "group_name": "task_radii",
        })
    primitive_group_specs = [
        ("primitive_mu", primitive_param_groups.get("mu", []), primitive_mu_lr_scale),
        (
            "primitive_logdiag",
            primitive_param_groups.get("logdiag", []),
            primitive_logdiag_lr_scale,
        ),
        ("primitive_log_s", primitive_param_groups.get("logs", []), primitive_log_s_lr_scale),
        (
            "covariance_offdiag",
            primitive_param_groups.get("offdiag", []),
            covariance_offdiag_lr_scale,
        ),
        (
            "ray_origin",
            primitive_param_groups.get("origin", []),
            primitive_ray_origin_lr_scale,
        ),
        (
            "primitive_other",
            primitive_param_groups.get("primitive_other", []),
            primitive_lr_scale,
        ),
        ("router_banks", primitive_param_groups.get("router_banks", []), primitive_lr_scale),
    ]
    for group_name, params, lr_scale in primitive_group_specs:
        if params:
            group = {
                "params": params,
                "lr": base_lr * lr_scale,
                "group_name": group_name,
            }
            if primitive_weight_decay is not None:
                group["weight_decay"] = float(primitive_weight_decay)
            param_groups.append(group)
    if lora_atom_params:
        param_groups.append({
            "params": lora_atom_params,
            "lr": base_lr * lora_atom_lr_scale,
            "group_name": "lora_atoms",
        })

    expected_by_id = {id(param): name for name, param in named_params}
    optimizer_param_ids = []
    for group in param_groups:
        optimizer_param_ids.extend(id(param) for param in group["params"])

    seen_ids = set()
    duplicate_ids = set()
    for pid in optimizer_param_ids:
        if pid in seen_ids:
            duplicate_ids.add(pid)
        seen_ids.add(pid)

    expected_ids = set(expected_by_id.keys())
    missing_ids = expected_ids - seen_ids
    unexpected_ids = seen_ids - expected_ids
    if missing_ids or unexpected_ids or duplicate_ids:
        def _names(param_ids):
            return [expected_by_id.get(pid, f"<unexpected:{pid}>") for pid in sorted(param_ids)]

        raise RuntimeError(
            "Optimizer parameter partition mismatch.\n"
            f"missing={_names(missing_ids)[:8]}\n"
            f"unexpected={_names(unexpected_ids)[:8]}\n"
            f"duplicate={_names(duplicate_ids)[:8]}"
        )

    return AdamW(
        param_groups,
        weight_decay=cfg.training.weight_decay,
        betas=(0.9, 0.999),
    )


def build_scheduler(optimizer, cfg, total_steps: int):
    total_steps = max(1, int(total_steps))
    warmup_ratio = float(cfg.training.warmup_ratio)
    warmup_steps = int(total_steps * warmup_ratio)

    if total_steps == 1:
        return LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=1)

    if warmup_steps <= 0:
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    warmup_steps = min(warmup_steps, total_steps - 1)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-6,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps),
        eta_min=1e-6,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )


def _geom_param_bucket(name: str) -> str | None:
    if name.startswith("primitive_banks."):
        if name.endswith(".ray_origin"):
            return "origin"
        if name.endswith(".mu"):
            return "mu"
        if name.endswith(".log_diag"):
            return "logdiag"
        if name.endswith(".log_s"):
            return "logs"
        if name.endswith(".lower_offdiag"):
            return "offdiag"
        return "primitive_other"
    if name.startswith("condition_encoder."):
        return "cond"
    if name.endswith(".U"):
        return "lora_U"
    if name.endswith(".V"):
        return "lora_V"
    return None


def build_geometry_diagnostics(raw_model: GSLoRAModel, cfg) -> dict | None:
    diag_cfg = cfg.training.get("geometry_diagnostics", {})
    if isinstance(diag_cfg, bool):
        enabled = bool(diag_cfg)
        interval = int(cfg.training.get("geometry_diagnostics_interval", 10))
    else:
        enabled = bool(diag_cfg.get("enabled", False))
        interval = int(diag_cfg.get("interval", 10))
    if not enabled:
        return None

    snapshot_buckets = {"mu", "logdiag", "logs", "offdiag", "origin", "cond"}
    params: list[tuple[str, str, torch.nn.Parameter]] = []
    snapshots: dict[str, torch.Tensor] = {}
    init_snapshots: dict[str, torch.Tensor] = {}
    for name, param in raw_model.named_parameters():
        bucket = _geom_param_bucket(name)
        if bucket is None or not param.requires_grad:
            continue
        params.append((name, bucket, param))
        if bucket in snapshot_buckets:
            snap = param.detach().float().cpu().clone()
            snapshots[name] = snap.clone()
            init_snapshots[name] = snap
    print(
        "[GeomDiag] enabled: "
        f"interval={interval}, tracked={len(params)}, "
        f"snapshots={len(snapshots)}"
    )
    return {
        "interval": max(1, interval),
        "params": params,
        "snapshots": snapshots,
        "init_snapshots": init_snapshots,
    }


def _norm_stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    sq = sum(v * v for v in values)
    return math.sqrt(sq), max(values)


def collect_geometry_grad_stats(geom_diag: dict | None) -> dict[str, float]:
    if geom_diag is None:
        return {}
    by_bucket: dict[str, list[float]] = {}
    param_norms: dict[str, list[float]] = {}
    for _, bucket, param in geom_diag["params"]:
        param_norms.setdefault(bucket, []).append(float(param.detach().float().norm().item()))
        if param.grad is None:
            continue
        by_bucket.setdefault(bucket, []).append(float(param.grad.detach().float().norm().item()))

    stats: dict[str, float] = {}
    for bucket in sorted({b for _, b, _ in geom_diag["params"]}):
        grad_norm, grad_max = _norm_stats(by_bucket.get(bucket, []))
        param_norm, _ = _norm_stats(param_norms.get(bucket, []))
        stats[f"{bucket}_grad"] = grad_norm
        stats[f"{bucket}_grad_max"] = grad_max
        stats[f"{bucket}_norm"] = param_norm
    return stats


def collect_geometry_delta_stats(geom_diag: dict | None) -> dict[str, float]:
    if geom_diag is None:
        return {}
    deltas: dict[str, list[float]] = {}
    init_deltas: dict[str, list[float]] = {}
    for name, bucket, param in geom_diag["params"]:
        if name not in geom_diag["snapshots"]:
            continue
        current = param.detach().float().cpu()
        prev = geom_diag["snapshots"][name]
        init = geom_diag["init_snapshots"][name]
        deltas.setdefault(bucket, []).append(float((current - prev).norm().item()))
        init_deltas.setdefault(bucket, []).append(float((current - init).norm().item()))
        geom_diag["snapshots"][name] = current.clone()

    stats: dict[str, float] = {}
    for bucket in sorted(deltas):
        delta_norm, delta_max = _norm_stats(deltas[bucket])
        init_delta_norm, _ = _norm_stats(init_deltas.get(bucket, []))
        stats[f"{bucket}_delta"] = delta_norm
        stats[f"{bucket}_delta_max"] = delta_max
        stats[f"{bucket}_init_delta"] = init_delta_norm
    return stats


def format_geometry_stats(step: int, grad_stats: dict[str, float], delta_stats: dict[str, float]) -> str:
    parts = [f"[GeomDiag step={step}]"]
    for bucket in ("mu", "logdiag", "logs", "offdiag", "origin", "cond", "lora_U", "lora_V"):
        keys = (
            f"{bucket}_grad",
            f"{bucket}_delta",
            f"{bucket}_init_delta",
            f"{bucket}_norm",
        )
        vals = []
        if keys[0] in grad_stats:
            vals.append(f"g={grad_stats[keys[0]]:.3e}")
        if keys[1] in delta_stats:
            vals.append(f"d={delta_stats[keys[1]]:.3e}")
        if keys[2] in delta_stats:
            vals.append(f"D={delta_stats[keys[2]]:.3e}")
        if keys[3] in grad_stats:
            vals.append(f"n={grad_stats[keys[3]]:.3e}")
        if vals:
            parts.append(f"{bucket}(" + ",".join(vals) + ")")
    return " ".join(parts)


def get_condition_hidden(
    batch: dict,
    cache: CachedConditionEmbeddings,
    template_cache: CachedConditionEmbeddings | None,
    model: GSLoRAModel,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Retrieve pre-computed condition embeddings from cache.
    Fall back to online encoding if not cached.

    Returns:
        (condition_hidden, condition_hidden_2)  – both (B, H) or None for _2
    """
    cids = batch["condition_ids"]

    # Try cache first
    if all(cache.has(cid) for cid in cids):
        cond_h = cache.get_batch(cids, device).to(dtype)
    else:
        if get_condition_source(model.cfg) != "backbone_pooled":
            missing = [cid for cid in cids if not cache.has(cid)]
            raise RuntimeError(
                "External condition encoder mode requires a complete condition cache. "
                f"Missing cached condition ids: {missing[:4]}"
            )
        # Online fallback (slow path)
        cond_h = model.encode_conditions_batch(
            batch["condition_input_ids"].to(device),
            batch["condition_attention_mask"].to(device),
        ).to(dtype)
        for cid, emb in zip(cids, cond_h):
            cache.set(cid, emb)

    # Second condition for consistency loss
    cond_h_2 = None
    if "condition_ids_2" in batch and any(c is not None for c in batch["condition_ids_2"]):
        cids_2 = batch["condition_ids_2"]
        secondary_cache = template_cache if template_cache is not None else cache
        if all(c is not None and secondary_cache.has(c) for c in cids_2):
            cond_h_2 = secondary_cache.get_batch(cids_2, device).to(dtype)
        elif "condition_input_ids_2" in batch:
            if get_condition_source(model.cfg) != "backbone_pooled":
                missing = [cid for cid in cids_2 if cid is not None and not secondary_cache.has(cid)]
                raise RuntimeError(
                    "External condition encoder mode requires a complete secondary condition cache. "
                    f"Missing cached condition ids: {missing[:4]}"
                )
            cond_h_2 = model.encode_conditions_batch(
                batch["condition_input_ids_2"].to(device),
                batch["condition_attention_mask_2"].to(device),
            ).to(dtype)

    return cond_h, cond_h_2


def train_step(
    batch: dict,
    model,
    cache: CachedConditionEmbeddings | None,
    template_cache: CachedConditionEmbeddings | None,
    cfg,
    device: torch.device,
    dtype: torch.dtype,
    routing_regularizer_scale: float = 1.0,
) -> dict:
    """Single training step. Returns dict of scalar losses."""
    model_module = unwrap_model(model)
    # Move task tensors to device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    if infer_model_type(cfg) == "shared":
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_logits=False,
        )
        l_task = output["loss"]
        zero = torch.zeros((), device=l_task.device, dtype=l_task.dtype)
        return {
            "loss": l_task,
            "loss_task": l_task.detach(),
            "loss_cons": zero.detach(),
            "loss_sparse": zero.detach(),
            "loss_balance": zero.detach(),
            "alpha": None,
            "layer_alphas": None,
            "z_norm_mean": 0.0,
            "z_norm_max": 0.0,
            "z_task_norm_mean": 0.0,
            "z_task_norm_max": 0.0,
            "condition_rho": 0.0,
            "task_rho_mean": 0.0,
            "task_rho_max": 0.0,
            "template_runtime_enabled": 0.0,
            "template_gamma": 0.0,
            "template_delta_max": 0.0,
            "template_z_gap_max": 0.0,
        }

    cond_h, cond_h_2 = get_condition_hidden(batch, cache, template_cache, model_module, device, dtype)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        condition_hidden=cond_h,
        task_ids=batch["dataset_ids"],
        labels=labels,
        condition_hidden_2=cond_h_2,
        return_logits=False,
    )

    template_runtime_enabled = bool(
        getattr(model_module, "template_runtime_enabled", getattr(model_module, "template_delta_enabled", False))
    )
    template_debug = {
        "z_norm_mean": 0.0,
        "z_norm_max": 0.0,
        "z_task_norm_mean": 0.0,
        "z_task_norm_max": 0.0,
        "condition_rho": 0.0,
        "task_rho_mean": 0.0,
        "task_rho_max": 0.0,
        "template_runtime_enabled": float(template_runtime_enabled),
        "template_gamma": 0.0,
        "template_delta_max": 0.0,
        "template_z_gap_max": 0.0,
    }
    if getattr(model_module, "template_gamma", None) is not None:
        template_debug["template_gamma"] = float(model_module.template_gamma.detach().item())
    if getattr(model_module, "condition_encoder", None) is not None:
        template_debug["condition_rho"] = float(
            model_module.condition_encoder.output_radius_value()
        )
        task_radii = model_module.condition_encoder.task_radius_values(
            batch["dataset_ids"],
            device=device,
            dtype=torch.float32,
            include_global_radius=True,
        )
        if task_radii is not None:
            template_debug["task_rho_mean"] = float(task_radii.mean().item())
            template_debug["task_rho_max"] = float(task_radii.max().item())
    if bool(getattr(model_module, "template_delta_enabled", False)) and not template_runtime_enabled:
        if "delta_template" in output:
            raise RuntimeError(
                "Template branch produced delta_template even though template_runtime_enabled=False."
            )
        z = output.get("z")
        z_task = output.get("z_task")
        if z is None or z_task is None:
            raise RuntimeError(
                "Template no-op check failed because z/z_task were missing from the model output."
            )
        max_abs_diff = (z - z_task).abs().max().item()
        template_debug["template_z_gap_max"] = max_abs_diff
        if max_abs_diff != 0.0:
            raise RuntimeError(
                f"Template branch is not a no-op before enable epoch: max_abs(z-z_task)={max_abs_diff:.6e}"
            )
    elif "delta_template" in output:
        template_debug["template_delta_max"] = float(output["delta_template"].abs().max().item())
        z = output.get("z")
        z_task = output.get("z_task")
        if z is not None and z_task is not None:
            template_debug["template_z_gap_max"] = float((z - z_task).abs().max().item())

    z = output.get("z")
    if z is not None:
        z_norm = z.norm(dim=-1)
        template_debug["z_norm_mean"] = float(z_norm.mean().item())
        template_debug["z_norm_max"] = float(z_norm.max().item())
    z_task = output.get("z_task")
    if z_task is not None:
        z_task_norm = z_task.norm(dim=-1)
        template_debug["z_task_norm_mean"] = float(z_task_norm.mean().item())
        template_debug["z_task_norm_max"] = float(z_task_norm.max().item())

    l_task = output["loss"]
    alpha = output["alpha"]
    alpha_2 = output.get("alpha_2")
    layer_alphas = output.get("layer_alphas", alpha)
    layer_alphas_2 = output.get("layer_alphas_2", alpha_2)

    struct_losses = compute_structural_losses(
        alpha=layer_alphas,
        alpha_2=layer_alphas_2,
        lambda_cons=cfg.loss.lambda_cons,
        lambda_sparse=cfg.loss.lambda_sparse * routing_regularizer_scale,
        lambda_balance=cfg.loss.lambda_balance * routing_regularizer_scale,
        consistency_metric=cfg.loss.consistency_metric,
    )

    total_loss = l_task + struct_losses["loss_struct"]

    return {
        "loss": total_loss,
        "loss_task": l_task.detach(),
        "loss_cons": struct_losses["loss_cons"].detach(),
        "loss_sparse": struct_losses["loss_sparse"].detach(),
        "loss_balance": struct_losses["loss_balance"].detach(),
        "alpha": alpha.detach(),
        "layer_alphas": layer_alphas.detach(),
        "routing_regularizer_scale": routing_regularizer_scale,
        **template_debug,
    }


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if not args.run_name:
        args.run_name = Path(args.config).stem
    model_type = infer_model_type(cfg)

    if cfg.training.get("gradient_checkpointing", False) and model_type in {"gs", "mlp", "mixlora"}:
        print(
            "[Train] WARNING: gradient_checkpointing=true is incompatible with the "
            "current Geofield-LoRA dynamic alpha path; overriding to false for this run."
        )
        cfg.training.gradient_checkpointing = False

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Match the original training behavior for single-run reproduction.
    torch.manual_seed(cfg.training.seed)
    torch.cuda.manual_seed_all(cfg.training.seed)

    dtype = torch.bfloat16 if cfg.training.bf16 else (
        torch.float16 if cfg.training.fp16 else torch.float32
    )

    # Dirs
    run_dir = os.path.join(cfg.training.output_dir, args.run_name)
    if is_main_process():
        os.makedirs(run_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(run_dir, "config.yaml"))

    writer = SummaryWriter(log_dir=os.path.join(run_dir, "tb_logs")) if is_main_process() else None

    active_task_ids: list[str] = []
    if bool(cfg.get("condition_encoder", {}).get("task_embedding_enabled", False)) or bool(
        cfg.get("condition_encoder", {}).get("task_radius_enabled", False)
    ):
        active_task_ids = get_active_task_ids(cfg)
        if is_main_process():
            print(
                f"[Train] task-specific condition parameters enabled for "
                f"{len(active_task_ids)} tasks"
            )

    # ── Model ─────────────────────────────────────────────────────────────
    if distributed and model_type == "shared":
        raise RuntimeError("Shared LoRA is currently supported only on a single process.")

    if is_main_process():
        print(f"Loading model... (type={model_type})")
    if model_type == "gs":
        model_cls = GSLoRAModel
    elif model_type == "mlp":
        model_cls = MLPGatedLoRAModel
    elif model_type == "mixlora":
        model_cls = MixLoRAModel
    elif model_type == "shared":
        model_cls = SharedLoRAModel
    else:
        raise ValueError(f"Unsupported model_type={model_type}")

    raw_model = model_cls.from_pretrained(
        cfg,
        device_map=cfg.model.get("device_map", None),
        device=device,
    )
    if active_task_ids and hasattr(raw_model, "register_task_ids"):
        raw_model.register_task_ids(active_task_ids)
    if args.resume:
        raw_model.load_adapter(args.resume)
    # Keep the backbone in train mode so HF gradient checkpointing can activate.
    # Parameters remain frozen because requires_grad=False is set in GSLoRAModel.
    raw_model.train()

    # For gradient checkpointing in backbone (if configured)
    if cfg.training.gradient_checkpointing:
        if hasattr(raw_model.backbone, "enable_input_require_grads"):
            raw_model.backbone.enable_input_require_grads()
        if hasattr(raw_model.backbone, "config"):
            raw_model.backbone.config.use_cache = False
        raw_model.backbone.gradient_checkpointing_enable()

    model = raw_model
    if distributed:
        ddp_find_unused_parameters = bool(
            cfg.training.get("freeze_condition_encoder", False)
            or int(cfg.training.get("condition_encoder_freeze_steps", 0)) > 0
            or int(cfg.training.get("task_view_embedding_freeze_steps", 0)) > 0
            or int(cfg.training.get("task_radius_freeze_steps", 0)) > 0
            or int(cfg.training.get("task_side_freeze_steps", 0)) > 0
            or int(
                cfg.training.get(
                    "template_enable_epoch",
                    1 if bool(getattr(raw_model, "template_delta_enabled", False)) else 0,
                )
            ) > 1
            or int(cfg.training.get("covariance_enable_epoch", 1)) > 1
            or float(cfg.training.get("aux_stage_task_freeze_fraction", 0.0)) > 0.0
        )
        if is_main_process() and ddp_find_unused_parameters:
            print(
                "[Train] DDP find_unused_parameters=True "
                "(staged trainability schedule detected)"
            )
        model = DDP(
            raw_model,
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=False,
            find_unused_parameters=ddp_find_unused_parameters,
        )

    # ── Condition cache ────────────────────────────────────────────────────
    cache = load_condition_cache(cfg) if model_type != "shared" else None
    template_cache = load_template_delta_cache(cfg) if model_type != "shared" else None

    # ── Data-driven initialization (optional) ─────────────────────────────
    if (
        cache is not None
        and model_type == "gs"
        and str(cfg.gslora.get("init_mode", "random")) == "data_driven"
        and not args.resume
    ):
        init_gain = float(cfg.gslora.get("init_gain", 1.0))
        init_task_ids = set(f"task::{tid}" for tid in get_active_task_ids(cfg))
        init_cache = {k: v for k, v in cache._cache.items() if k in init_task_ids}
        if is_main_process():
            print(
                "[Train] Applying data-driven initialization "
                f"(PCA + K-means, gain={init_gain}, tasks={len(init_cache)})"
            )
        raw_model.initialize_from_condition_data(init_cache, gain=init_gain)

    # ── Data ──────────────────────────────────────────────────────────────
    if is_main_process():
        print("Loading datasets...")
    num_workers = cfg.training.get("num_workers", 0)
    train_loader, val_seen_loader, val_heldout_loader = build_dataloaders(
        cfg,
        raw_model.tokenizer,
        num_workers=num_workers,
        distributed=distributed,
        rank=get_rank(),
        world_size=get_world_size(),
        build_eval_loaders=is_main_process(),
    )
    train_examples = len(train_loader.dataset)
    seen_examples = len(val_seen_loader.dataset) if val_seen_loader is not None else 0
    heldout_examples = len(val_heldout_loader.dataset) if val_heldout_loader is not None else 0
    effective_batch_size = cfg.training.batch_size * cfg.training.grad_accum_steps
    if is_main_process():
        print(
            f"[Train] train_examples={train_examples:,}, train_batches_per_rank={len(train_loader):,}, "
            f"world_size={get_world_size()}, batch_size_per_rank={cfg.training.batch_size}, "
            f"grad_accum={cfg.training.grad_accum_steps}, "
            f"effective_global_batch={effective_batch_size * get_world_size()}"
        )
        print(
            f"[Train] train_loader_rng_seed={int(cfg.training.seed)} "
            "(independent generator)"
        )
        print(
            f"[Eval] seen_examples={seen_examples:,}, seen_batches={len(val_seen_loader):,}, "
            f"held_out_examples={heldout_examples:,}, held_out_batches={len(val_heldout_loader):,}"
        )

    # ── Optimizer & Scheduler ─────────────────────────────────────────────
    permanently_freeze_condition_encoder = bool(
        cfg.training.get("freeze_condition_encoder", False)
    ) and model_type != "shared"
    total_steps = (
        math.ceil(len(train_loader) / cfg.training.grad_accum_steps)
        * cfg.training.num_epochs
    )
    steps_per_epoch = math.ceil(len(train_loader) / cfg.training.grad_accum_steps)

    condition_encoder_freeze_steps = int(cfg.training.get("condition_encoder_freeze_steps", 0))
    task_view_embedding_freeze_steps = int(
        cfg.training.get("task_view_embedding_freeze_steps", condition_encoder_freeze_steps)
    )
    task_radius_freeze_steps = int(
        cfg.training.get("task_radius_freeze_steps", condition_encoder_freeze_steps)
    )
    task_side_freeze_steps = int(cfg.training.get("task_side_freeze_steps", 0))
    has_task_view_embeddings = bool(
        model_type != "shared"
        and getattr(raw_model.condition_encoder, "has_task_embeddings", lambda: False)()
    )
    has_task_radii = bool(
        model_type != "shared"
        and getattr(raw_model.condition_encoder, "has_task_radii", lambda: False)()
    )
    has_template_side = model_type != "shared" and bool(getattr(raw_model, "template_delta_enabled", False))
    template_enable_epoch = int(
        cfg.training.get("template_enable_epoch", 1 if has_template_side else 0)
    )
    primitive_banks = getattr(raw_model, "primitive_banks", None)
    has_covariance_offdiag = (
        primitive_banks is not None
        and any(
            getattr(bank, "lower_offdiag", None) is not None
            for bank in primitive_banks.values()
        )
    )
    covariance_enable_epoch = int(
        cfg.training.get("covariance_enable_epoch", 1 if has_covariance_offdiag else 0)
    )
    aux_stage_task_freeze_fraction = float(
        cfg.training.get("aux_stage_task_freeze_fraction", 0.0)
    )
    aux_stage_task_freeze_fraction = min(max(aux_stage_task_freeze_fraction, 0.0), 1.0)
    reset_optimizer_on_aux_stage_start = bool(
        cfg.training.get("reset_optimizer_on_aux_stage_start", False)
    )
    reset_optimizer_on_task_reenable = bool(
        cfg.training.get("reset_optimizer_on_task_reenable", False)
    )
    optimizer_reset_steps = sorted(
        {
            int(step)
            for step in cfg.training.get("optimizer_reset_steps", [])
            if int(step) > 0
        }
    )
    routing_regularizer_warmup_steps = int(
        cfg.training.get(
            "routing_regularizer_warmup_steps",
            cfg.training.get("sparse_balance_warmup_steps", 0),
        )
    )

    aux_enable_steps = []
    if has_template_side and template_enable_epoch > 1:
        aux_enable_steps.append((template_enable_epoch - 1) * steps_per_epoch)
    if has_covariance_offdiag and covariance_enable_epoch > 1:
        aux_enable_steps.append((covariance_enable_epoch - 1) * steps_per_epoch)

    aux_stage_start_step = None
    aux_stage_end_step = None
    if aux_enable_steps and aux_stage_task_freeze_fraction > 0.0:
        aux_stage_start_step = min(aux_enable_steps)
        aux_stage_duration_steps = max(1, math.ceil(steps_per_epoch * aux_stage_task_freeze_fraction))
        aux_stage_end_step = min(total_steps, aux_stage_start_step + aux_stage_duration_steps)

    if is_main_process():
        if permanently_freeze_condition_encoder:
            print("[Train] condition_encoder permanently frozen for this run")
        if task_side_freeze_steps > 0 and model_type != "shared":
            print(
                f"[Train] task-side modules frozen for the first "
                f"{task_side_freeze_steps} optimizer steps"
            )
        if condition_encoder_freeze_steps > 0 and not permanently_freeze_condition_encoder:
            print(
                f"[Train] condition_encoder frozen for the first "
                f"{condition_encoder_freeze_steps} optimizer steps"
            )
        if has_task_view_embeddings and task_view_embedding_freeze_steps > 0:
            print(
                f"[Train] task-view embeddings frozen for the first "
                f"{task_view_embedding_freeze_steps} optimizer steps"
            )
        if has_task_radii and task_radius_freeze_steps > 0:
            print(
                f"[Train] task-specific rho frozen for the first "
                f"{task_radius_freeze_steps} optimizer steps"
            )
        if has_template_side:
            if template_enable_epoch <= 1:
                print("[Train] template-side modules enabled from epoch 1")
            else:
                print(
                    f"[Train] template-side modules frozen until epoch {template_enable_epoch}"
                )
        if has_covariance_offdiag:
            if covariance_enable_epoch <= 1:
                print("[Train] covariance off-diagonal terms enabled from epoch 1")
            else:
                print(
                    f"[Train] covariance off-diagonal terms frozen until epoch {covariance_enable_epoch}"
                )
        if aux_stage_start_step is not None:
            print(
                f"[Train] aux-stage schedule: start_step={aux_stage_start_step}, "
                f"end_step={aux_stage_end_step}, "
                f"freeze_task_fraction={aux_stage_task_freeze_fraction:.2f}, "
                f"reset_on_start={reset_optimizer_on_aux_stage_start}, "
                f"reset_on_task_reenable={reset_optimizer_on_task_reenable}"
            )
        if optimizer_reset_steps:
            print(
                f"[Train] scheduled optimizer resets at steps {optimizer_reset_steps}"
            )
        if routing_regularizer_warmup_steps > 0:
            print(
                "[Train] routing sparsity/balance regularizers warm up for "
                f"{routing_regularizer_warmup_steps} optimizer steps"
            )

    def in_aux_only_stage(step: int) -> bool:
        return (
            aux_stage_start_step is not None
            and aux_stage_end_step is not None
            and aux_stage_start_step <= step < aux_stage_end_step
        )

    def desired_task_side_trainable(step: int) -> bool:
        enabled = step >= task_side_freeze_steps
        if in_aux_only_stage(step):
            enabled = False
        return enabled

    def desired_condition_encoder_trainable(step: int, task_enabled: bool) -> bool:
        return (
            task_enabled
            and not permanently_freeze_condition_encoder
            and step >= condition_encoder_freeze_steps
        )

    def desired_task_view_embeddings_trainable(step: int, task_enabled: bool) -> bool:
        return (
            has_task_view_embeddings
            and task_enabled
            and not permanently_freeze_condition_encoder
            and step >= task_view_embedding_freeze_steps
        )

    def desired_task_radii_trainable(step: int, task_enabled: bool) -> bool:
        return (
            has_task_radii
            and task_enabled
            and not permanently_freeze_condition_encoder
            and step >= task_radius_freeze_steps
        )

    def desired_template_side_trainable(epoch_index: int) -> bool:
        return has_template_side and epoch_index >= template_enable_epoch

    def desired_covariance_offdiag_trainable(epoch_index: int) -> bool:
        return has_covariance_offdiag and epoch_index >= covariance_enable_epoch

    task_side_trainable = None
    condition_encoder_trainable = None
    task_view_embeddings_trainable = None
    task_radii_trainable = None
    template_side_trainable = None
    covariance_offdiag_trainable = None

    def apply_training_state(epoch_index: int, step: int):
        nonlocal task_side_trainable
        nonlocal condition_encoder_trainable
        nonlocal task_view_embeddings_trainable
        nonlocal task_radii_trainable
        nonlocal template_side_trainable
        nonlocal covariance_offdiag_trainable

        if model_type == "shared":
            return

        should_train_task_side = desired_task_side_trainable(step)
        if should_train_task_side != task_side_trainable:
            set_task_side_trainable(raw_model, should_train_task_side)
            task_side_trainable = should_train_task_side
            if is_main_process():
                state = "unfrozen" if should_train_task_side else "frozen"
                print(f"[Train] task-side modules {state} at step {step}")

        should_train_condition_encoder = desired_condition_encoder_trainable(
            step, should_train_task_side
        )
        if should_train_condition_encoder != condition_encoder_trainable:
            set_condition_encoder_base_trainable(raw_model, should_train_condition_encoder)
            condition_encoder_trainable = should_train_condition_encoder
            if is_main_process():
                state = "unfrozen" if should_train_condition_encoder else "frozen"
                print(f"[Train] condition_encoder {state} at step {step}")

        should_train_task_view_embeddings = desired_task_view_embeddings_trainable(
            step, should_train_task_side
        )
        if should_train_task_view_embeddings != task_view_embeddings_trainable:
            set_task_view_embeddings_trainable(raw_model, should_train_task_view_embeddings)
            task_view_embeddings_trainable = should_train_task_view_embeddings
            if is_main_process() and has_task_view_embeddings:
                state = "unfrozen" if should_train_task_view_embeddings else "frozen"
                print(f"[Train] task-view embeddings {state} at step {step}")

        should_train_task_radii = desired_task_radii_trainable(
            step, should_train_task_side
        )
        if should_train_task_radii != task_radii_trainable:
            set_task_radii_trainable(raw_model, should_train_task_radii)
            task_radii_trainable = should_train_task_radii
            if is_main_process() and has_task_radii:
                state = "unfrozen" if should_train_task_radii else "frozen"
                print(f"[Train] task-specific rho {state} at step {step}")

        if has_template_side:
            should_train_template_side = desired_template_side_trainable(epoch_index)
            if should_train_template_side != template_side_trainable:
                set_template_side_trainable(raw_model, should_train_template_side)
                template_side_trainable = should_train_template_side
                raw_model.template_runtime_enabled = should_train_template_side
                if is_main_process():
                    state = "enabled" if should_train_template_side else "frozen"
                    print(
                        f"[Train] template-side modules {state} at "
                        f"epoch {epoch_index}, step {step}"
                    )
            else:
                raw_model.template_runtime_enabled = bool(template_side_trainable)

        if has_covariance_offdiag:
            should_train_covariance_offdiag = desired_covariance_offdiag_trainable(epoch_index)
            if should_train_covariance_offdiag != covariance_offdiag_trainable:
                set_covariance_offdiag_trainable(raw_model, should_train_covariance_offdiag)
                covariance_offdiag_trainable = should_train_covariance_offdiag
                if is_main_process():
                    state = "enabled" if should_train_covariance_offdiag else "frozen"
                    print(
                        f"[Train] covariance off-diagonal terms {state} at "
                        f"epoch {epoch_index}, step {step}"
                    )

    global_step = 0
    apply_training_state(epoch_index=1, step=global_step)
    aux_stage_start_reset_done = False
    aux_stage_end_reset_done = False
    scheduled_optimizer_reset_done: set[int] = set()

    optimizer = build_optimizer(raw_model, cfg)
    if is_main_process():
        group_summaries = [
            f"{group.get('group_name', idx)}:lr={group['lr']:.2e},n={sum(p.numel() for p in group['params']):,}"
            for idx, group in enumerate(optimizer.param_groups)
        ]
        print(f"[Train] optimizer_groups: {' | '.join(group_summaries)}")
    scheduler = build_scheduler(optimizer, cfg, total_steps)
    geom_diag = build_geometry_diagnostics(raw_model, cfg) if is_main_process() else None

    def rebuild_optimizer_and_scheduler(reason: str):
        nonlocal optimizer, scheduler
        remaining_steps = max(1, total_steps - global_step)
        optimizer = build_optimizer(raw_model, cfg)
        scheduler = build_scheduler(optimizer, cfg, remaining_steps)
        optimizer.zero_grad()
        if is_main_process():
            print(
                f"[Train] optimizer/scheduler reset at step {global_step}: "
                f"{reason} (remaining_steps={remaining_steps})"
            )

    # ── Training Loop ─────────────────────────────────────────────────────
    best_seen_score = -float("inf")
    best_heldout_score = -float("inf")
    max_train_steps = int(cfg.training.get("max_steps", total_steps))
    if max_train_steps <= 0:
        max_train_steps = total_steps
    max_train_steps = min(max_train_steps, total_steps)
    if is_main_process() and max_train_steps < total_steps:
        print(f"[Train] max_steps override: stopping after {max_train_steps}/{total_steps} optimizer steps")

    def run_validation(eval_suffix: str = ""):
        """Run seen/held-out eval on main; DDP barrier before/after. eval_suffix e.g. ' (final)'."""
        nonlocal best_seen_score, best_heldout_score
        model.eval()
        if distributed:
            dist.barrier()
        if is_main_process() and val_seen_loader is not None:
            if model_type == "shared":
                seen_metrics = evaluate_shared(
                    raw_model,
                    raw_model.tokenizer,
                    val_seen_loader,
                    cfg,
                    device,
                    split="seen",
                )
                held_metrics = evaluate_shared(
                    raw_model,
                    raw_model.tokenizer,
                    val_heldout_loader,
                    cfg,
                    device,
                    split="held_out",
                )
            else:
                seen_metrics = evaluate(
                    raw_model,
                    val_seen_loader,
                    cache,
                    cfg,
                    device,
                    dtype,
                    split="seen",
                    template_cache=template_cache,
                )
                held_metrics = evaluate(
                    raw_model,
                    val_heldout_loader,
                    cache,
                    cfg,
                    device,
                    dtype,
                    split="held_out",
                    template_cache=template_cache,
                )

            val_score = seen_metrics.get("mean_score", 0.0)
            heldout_score = held_metrics.get("mean_score", 0.0)
            gap = val_score - heldout_score

            writer.add_scalar("val/seen_mean", val_score, global_step)
            writer.add_scalar("val/held_out_mean", heldout_score, global_step)
            writer.add_scalar("val/template_gap", gap, global_step)

            def _tb_ds_tag(split_prefix: str, ds_id: str) -> str:
                safe = ds_id.replace("/", "_").replace(" ", "_")
                return f"val/{split_prefix}_per_ds/{safe}"

            seen_per = seen_metrics.get("per_dataset_scores") or {}
            held_per = held_metrics.get("per_dataset_scores") or {}
            for ds_id, score in sorted(seen_per.items()):
                writer.add_scalar(_tb_ds_tag("seen", ds_id), score, global_step)
            for ds_id, score in sorted(held_per.items()):
                writer.add_scalar(_tb_ds_tag("held_out", ds_id), score, global_step)

            # Log format aligned with pilot_gslora_*.log (runtime/checkpoints/pilot_gslora_1gpu.log)
            print(
                f"\n[Eval step={global_step}{eval_suffix}] "
                f"seen={val_score:.4f}, held_out={heldout_score:.4f}, gap={gap:.4f}"
            )
            print("  [seen] per-dataset:")
            for ds_id, score in sorted(seen_per.items()):
                print(f"    {ds_id:20s}: {score:.4f}")
            print("  [held-out] per-dataset:")
            for ds_id, score in sorted(held_per.items()):
                print(f"    {ds_id:20s}: {score:.4f}")

            if val_score > best_seen_score:
                best_seen_score = val_score
                raw_model.save_adapter(os.path.join(run_dir, "best_seen_adapter"))
                print(f"  → New best seen: {best_seen_score:.4f}")

            if heldout_score > best_heldout_score:
                best_heldout_score = heldout_score
                raw_model.save_adapter(os.path.join(run_dir, "best_heldout_adapter"))
                print(f"  → New best held_out: {best_heldout_score:.4f}")

        model.train()
        if model_type != "shared" and hasattr(raw_model, "backbone"):
            raw_model.backbone.eval()
        if distributed:
            dist.barrier()

    if bool(cfg.training.get("eval_before_train", False)):
        if is_main_process():
            print("\n[Train] Running pre-train evaluation at step 0")
        run_validation(eval_suffix=" pretrain")

    stop_training = False
    for epoch in range(cfg.training.num_epochs):
        sampler = getattr(train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        epoch_index = epoch + 1
        apply_training_state(epoch_index=epoch_index, step=global_step)
        optimizer.zero_grad()
        interactive = sys.stderr.isatty()
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{cfg.training.num_epochs}",
            total=len(train_loader),
            unit="batch",
            dynamic_ncols=True,
            disable=(not interactive) or (not is_main_process()),
        )
        log_window = {
            "loss": 0.0,
            "loss_task": 0.0,
            "loss_cons": 0.0,
            "loss_sparse": 0.0,
            "loss_balance": 0.0,
            "entropy": 0.0,
            "z_norm_mean": 0.0,
            "z_norm_max": 0.0,
            "z_task_norm_mean": 0.0,
            "z_task_norm_max": 0.0,
            "condition_rho": 0.0,
            "task_rho_mean": 0.0,
            "task_rho_max": 0.0,
            "template_runtime_enabled": 0.0,
            "template_gamma": 0.0,
            "template_delta_max": 0.0,
            "template_z_gap_max": 0.0,
            "routing_regularizer_scale": 0.0,
        }
        log_window_batches = 0
        epoch_examples_seen = 0

        for step_in_epoch, batch in enumerate(pbar):
            if (
                aux_stage_start_step is not None
                and global_step == aux_stage_start_step
                and not aux_stage_start_reset_done
            ):
                apply_training_state(epoch_index=epoch_index, step=global_step)
                if reset_optimizer_on_aux_stage_start:
                    rebuild_optimizer_and_scheduler("aux stage start")
                aux_stage_start_reset_done = True
            elif (
                aux_stage_end_step is not None
                and global_step == aux_stage_end_step
                and not aux_stage_end_reset_done
            ):
                apply_training_state(epoch_index=epoch_index, step=global_step)
                if reset_optimizer_on_task_reenable:
                    rebuild_optimizer_and_scheduler("task-side re-enable")
                aux_stage_end_reset_done = True
            else:
                apply_training_state(epoch_index=epoch_index, step=global_step)

            if (
                optimizer_reset_steps
                and global_step in optimizer_reset_steps
                and global_step not in scheduled_optimizer_reset_done
            ):
                rebuild_optimizer_and_scheduler(
                    f"scheduled reset step {global_step}"
                )
                scheduled_optimizer_reset_done.add(global_step)

            # Train step
            if routing_regularizer_warmup_steps > 0:
                routing_regularizer_scale = min(
                    1.0,
                    float(global_step) / float(max(1, routing_regularizer_warmup_steps)),
                )
            else:
                routing_regularizer_scale = 1.0
            step_losses = train_step(
                batch,
                model,
                cache,
                template_cache,
                cfg,
                device,
                dtype,
                routing_regularizer_scale=routing_regularizer_scale,
            )
            batch_size = batch["input_ids"].shape[0]
            epoch_examples_seen += batch_size
            batch_stats = (
                raw_model.get_primitive_stats(step_losses["layer_alphas"])
                if step_losses["layer_alphas"] is not None
                else {}
            )
            log_window["loss"] += step_losses["loss"].item()
            log_window["loss_task"] += step_losses["loss_task"].item()
            log_window["loss_cons"] += step_losses["loss_cons"].item()
            log_window["loss_sparse"] += step_losses["loss_sparse"].item()
            log_window["loss_balance"] += step_losses["loss_balance"].item()
            log_window["entropy"] += batch_stats.get("entropy_mean", 0.0)
            log_window["z_norm_mean"] += step_losses.get("z_norm_mean", 0.0)
            log_window["z_norm_max"] += step_losses.get("z_norm_max", 0.0)
            log_window["z_task_norm_mean"] += step_losses.get("z_task_norm_mean", 0.0)
            log_window["z_task_norm_max"] += step_losses.get("z_task_norm_max", 0.0)
            log_window["condition_rho"] += step_losses.get("condition_rho", 0.0)
            log_window["task_rho_mean"] += step_losses.get("task_rho_mean", 0.0)
            log_window["task_rho_max"] += step_losses.get("task_rho_max", 0.0)
            log_window["template_runtime_enabled"] += step_losses.get("template_runtime_enabled", 0.0)
            log_window["template_gamma"] += step_losses.get("template_gamma", 0.0)
            log_window["template_delta_max"] += step_losses.get("template_delta_max", 0.0)
            log_window["template_z_gap_max"] += step_losses.get("template_z_gap_max", 0.0)
            log_window["routing_regularizer_scale"] += step_losses.get("routing_regularizer_scale", 1.0)
            log_window_batches += 1

            loss = step_losses["loss"] / cfg.training.grad_accum_steps
            loss.backward()

            if (step_in_epoch + 1) % cfg.training.grad_accum_steps == 0:
                next_global_step = global_step + 1
                geom_grad_stats = {}
                if (
                    geom_diag is not None
                    and next_global_step % int(geom_diag["interval"]) == 0
                ):
                    geom_grad_stats = collect_geometry_grad_stats(geom_diag)
                torch.nn.utils.clip_grad_norm_(
                    raw_model.trainable_parameters(),
                    cfg.training.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                global_step += 1
                if (
                    geom_diag is not None
                    and global_step % int(geom_diag["interval"]) == 0
                ):
                    geom_delta_stats = collect_geometry_delta_stats(geom_diag)
                    print(format_geometry_stats(global_step, geom_grad_stats, geom_delta_stats))
                    if writer is not None:
                        for key, value in geom_grad_stats.items():
                            writer.add_scalar(f"geom/{key}", value, global_step)
                        for key, value in geom_delta_stats.items():
                            writer.add_scalar(f"geom/{key}", value, global_step)
                optimizer.zero_grad()

                # Logging
                if global_step % cfg.training.logging_steps == 0:
                    lr = scheduler.get_last_lr()[0]
                    avg_loss = reduce_mean(log_window["loss"] / max(log_window_batches, 1), device)
                    avg_loss_task = reduce_mean(log_window["loss_task"] / max(log_window_batches, 1), device)
                    avg_loss_cons = reduce_mean(log_window["loss_cons"] / max(log_window_batches, 1), device)
                    avg_loss_sparse = reduce_mean(log_window["loss_sparse"] / max(log_window_batches, 1), device)
                    avg_loss_balance = reduce_mean(log_window["loss_balance"] / max(log_window_batches, 1), device)
                    avg_entropy = reduce_mean(log_window["entropy"] / max(log_window_batches, 1), device)
                    avg_z_norm_mean = reduce_mean(
                        log_window["z_norm_mean"] / max(log_window_batches, 1), device
                    )
                    avg_z_norm_max = reduce_mean(
                        log_window["z_norm_max"] / max(log_window_batches, 1), device
                    )
                    avg_z_task_norm_mean = reduce_mean(
                        log_window["z_task_norm_mean"] / max(log_window_batches, 1), device
                    )
                    avg_z_task_norm_max = reduce_mean(
                        log_window["z_task_norm_max"] / max(log_window_batches, 1), device
                    )
                    avg_condition_rho = reduce_mean(
                        log_window["condition_rho"] / max(log_window_batches, 1), device
                    )
                    avg_task_rho_mean = reduce_mean(
                        log_window["task_rho_mean"] / max(log_window_batches, 1), device
                    )
                    avg_task_rho_max = reduce_mean(
                        log_window["task_rho_max"] / max(log_window_batches, 1), device
                    )
                    avg_template_runtime_enabled = reduce_mean(
                        log_window["template_runtime_enabled"] / max(log_window_batches, 1), device
                    )
                    avg_template_gamma = reduce_mean(
                        log_window["template_gamma"] / max(log_window_batches, 1), device
                    )
                    avg_template_delta_max = reduce_mean(
                        log_window["template_delta_max"] / max(log_window_batches, 1), device
                    )
                    avg_template_z_gap_max = reduce_mean(
                        log_window["template_z_gap_max"] / max(log_window_batches, 1), device
                    )
                    avg_routing_regularizer_scale = reduce_mean(
                        log_window["routing_regularizer_scale"] / max(log_window_batches, 1), device
                    )
                    stats = (
                        raw_model.get_primitive_stats(step_losses["layer_alphas"])
                        if step_losses["layer_alphas"] is not None
                        else {}
                    )
                    if is_main_process():
                        writer.add_scalar("train/loss", avg_loss, global_step)
                        writer.add_scalar("train/loss_task", avg_loss_task, global_step)
                        writer.add_scalar("train/loss_cons", avg_loss_cons, global_step)
                        writer.add_scalar("train/loss_sparse", avg_loss_sparse, global_step)
                        writer.add_scalar("train/loss_balance", avg_loss_balance, global_step)
                        writer.add_scalar("train/lr", lr, global_step)
                        writer.add_scalar("condition/z_norm_mean", avg_z_norm_mean, global_step)
                        writer.add_scalar("condition/z_norm_max", avg_z_norm_max, global_step)
                        writer.add_scalar("condition/z_task_norm_mean", avg_z_task_norm_mean, global_step)
                        writer.add_scalar("condition/z_task_norm_max", avg_z_task_norm_max, global_step)
                        writer.add_scalar("condition/rho", avg_condition_rho, global_step)
                        writer.add_scalar("condition/task_rho_mean", avg_task_rho_mean, global_step)
                        writer.add_scalar("condition/task_rho_max", avg_task_rho_max, global_step)
                        writer.add_scalar("template/runtime_enabled", avg_template_runtime_enabled, global_step)
                        writer.add_scalar("template/gamma", avg_template_gamma, global_step)
                        writer.add_scalar("template/delta_max", avg_template_delta_max, global_step)
                        writer.add_scalar("template/z_gap_max", avg_template_z_gap_max, global_step)
                        writer.add_scalar(
                            "train/routing_regularizer_scale",
                            avg_routing_regularizer_scale,
                            global_step,
                        )

                        # Primitive stats
                        if "gini" in stats:
                            writer.add_scalar("primitives/entropy_mean", avg_entropy, global_step)
                            writer.add_scalar("primitives/gini", stats["gini"], global_step)

                        if interactive:
                            postfix = {
                                "step": f"{global_step}/{total_steps}",
                                "loss": f"{avg_loss:.4f}",
                                "lr": f"{lr:.2e}",
                            }
                            if "gini" in stats:
                                postfix["entropy"] = f"{avg_entropy:.2f}"
                            pbar.set_postfix(postfix)

                        msg = (
                            f"[Train step={global_step}/{total_steps}] "
                            f"epoch={epoch + 1}/{cfg.training.num_epochs}, "
                            f"batch={step_in_epoch + 1}/{len(train_loader)}, "
                            f"examples≈{epoch_examples_seen * get_world_size():,} (all-ranks est.) / "
                            f"rank0={epoch_examples_seen:,}, dataset_n={train_examples:,}, "
                            f"loss={avg_loss:.4f}, task={avg_loss_task:.4f}, "
                            f"cons={avg_loss_cons:.4f}, sparse={avg_loss_sparse:.4f}, "
                            f"balance={avg_loss_balance:.4f}, lr={lr:.2e}, "
                            f"rho={avg_condition_rho:.2f}, "
                            f"task_rho={avg_task_rho_mean:.2f}/{avg_task_rho_max:.2f}, "
                            f"z={avg_z_norm_mean:.2f}/{avg_z_norm_max:.2f}"
                        )
                        if "gini" in stats:
                            msg += f", entropy={avg_entropy:.2f}"
                        if getattr(raw_model, "template_delta_enabled", False):
                            msg += (
                                f", tpl_rt={avg_template_runtime_enabled:.2f}, "
                                f"tpl_gamma={avg_template_gamma:.3e}, "
                                f"tpl_delta_max={avg_template_delta_max:.3e}, "
                                f"tpl_z_gap={avg_template_z_gap_max:.3e}"
                            )
                        if routing_regularizer_warmup_steps > 0:
                            msg += f", route_reg={avg_routing_regularizer_scale:.2f}"
                        print(msg)

                    for key in log_window:
                        log_window[key] = 0.0
                    log_window_batches = 0

                # Evaluation (also see end-of-training catch-up below)
                if global_step % cfg.training.eval_steps == 0:
                    run_validation("")

                # Save checkpoint
                if global_step % cfg.training.save_steps == 0 and is_main_process():
                    ckpt_dir = os.path.join(run_dir, f"checkpoint_{global_step}")
                    raw_model.save_adapter(ckpt_dir)

                if global_step >= max_train_steps:
                    stop_training = True
                    break
        if stop_training:
            break

    # If training stopped on a step not divisible by eval_steps / save_steps, the last
    # optimizer step never triggered eval or periodic checkpoint; align with user expectation.
    if global_step > 0:
        if global_step % cfg.training.eval_steps != 0:
            if is_main_process() and val_seen_loader is not None:
                print(
                    f"\n[Train] End-of-training eval (step {global_step} not on eval_steps="
                    f"{cfg.training.eval_steps} grid)"
                )
            run_validation(" (final)")
        if global_step % cfg.training.save_steps != 0 and is_main_process():
            ckpt_dir = os.path.join(run_dir, f"checkpoint_{global_step}")
            raw_model.save_adapter(ckpt_dir)
            print(f"[Train] End-of-training checkpoint: {ckpt_dir}")

    # Final save
    if is_main_process():
        raw_model.save_adapter(os.path.join(run_dir, "final_adapter"))
        writer.close()
        print(
            f"\nTraining complete. "
            f"Best seen score: {best_seen_score:.4f}, "
            f"best held_out score: {best_heldout_score:.4f}"
        )
        print(f"Checkpoints saved to: {run_dir}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
