"""
Condition Encoder: maps condition text embedding → view embedding z.

The condition text c (e.g., prompt template, instruction) is first encoded
by the frozen backbone to get a pooled hidden state, then projected through
a small MLP to obtain z ∈ R^view_dim.

We support two modes:
  1. online: encode c via backbone at training time (slow but flexible)
  2. cached: load pre-computed embeddings from disk (fast, recommended)
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence


class ConditionEncoder(nn.Module):
    """
    Two-layer MLP that maps a pooled backbone hidden state to view embedding z.

    Args:
        input_dim:  backbone hidden size (e.g. 1536 for Qwen2.5-1.5B)
        hidden_dim: MLP intermediate size
        view_dim:   output view embedding dimension
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        view_dim: int,
        head_type: str = "mlp",
        output_normalize: bool = False,
        output_radius_init: float = 1.0,
        output_radius_learnable: bool = False,
        task_embedding_enabled: bool = False,
        task_embedding_init_std: float = 0.0,
        task_radius_enabled: bool = False,
        task_radius_init: float = 1.0,
        task_radius_init_std: float = 0.0,
    ):
        super().__init__()
        self.head_type = head_type
        self.output_normalize = bool(output_normalize)
        self.output_radius_learnable = bool(output_radius_learnable)
        self.task_embedding_enabled = bool(task_embedding_enabled)
        self.task_embedding_init_std = float(task_embedding_init_std)
        self.task_radius_enabled = bool(task_radius_enabled)
        self.task_radius_init = float(task_radius_init)
        self.task_radius_init_std = float(task_radius_init_std)
        self._build_modules_without_advancing_rng(input_dim, hidden_dim, view_dim, head_type)
        self.view_dim = view_dim
        self.task_embeddings: nn.Embedding | None = None
        self.task_radius_offsets: nn.Embedding | None = None
        self._task_id_to_index: dict[str, int] = {}
        self._init_output_radius(output_radius_init)
        self._init_weights()

    def _init_output_radius(self, output_radius_init: float) -> None:
        radius = float(output_radius_init)
        if radius <= 0.0:
            raise ValueError(
                f"output_radius_init must be positive, got {output_radius_init!r}"
            )
        if self.output_radius_learnable:
            self.log_output_radius = nn.Parameter(
                torch.tensor(math.log(radius), dtype=torch.float32)
            )
            self.register_buffer(
                "fixed_output_radius",
                torch.tensor(1.0, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.register_buffer(
                "fixed_output_radius",
                torch.tensor(radius, dtype=torch.float32),
                persistent=True,
            )
            self.register_parameter("log_output_radius", None)

    def _build_modules_without_advancing_rng(
        self,
        input_dim: int,
        hidden_dim: int,
        view_dim: int,
        head_type: str,
    ) -> None:
        cpu_rng_state = torch.get_rng_state()
        try:
            if head_type == "linear":
                self.mlp = nn.Linear(input_dim, view_dim)
            elif head_type == "mlp":
                self.mlp = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, view_dim),
                )
            elif head_type == "res_mlp":
                self.base_linear = nn.Linear(input_dim, view_dim)
                self.residual_mlp = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, view_dim),
                )
            else:
                raise ValueError(
                    f"Unsupported condition encoder head_type={head_type!r}. "
                    "Expected one of: 'mlp', 'linear', 'res_mlp'."
                )
        finally:
            # Keep the global RNG stream unchanged so swapping the condition head
            # does not perturb downstream Gaussian / LoRA initialization.
            torch.set_rng_state(cpu_rng_state)

    def _init_weights(self):
        init_gen = torch.Generator(device="cpu")
        init_gen.manual_seed(int(torch.initial_seed()))

        if self.head_type in {"linear", "mlp"}:
            for module in self.mlp.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.1, generator=init_gen)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            return

        if self.head_type == "res_mlp":
            nn.init.xavier_uniform_(self.base_linear.weight, gain=0.1, generator=init_gen)
            if self.base_linear.bias is not None:
                nn.init.zeros_(self.base_linear.bias)

            residual_linears = [
                module
                for module in self.residual_mlp.modules()
                if isinstance(module, nn.Linear)
            ]
            for module in residual_linears[:-1]:
                nn.init.xavier_uniform_(module.weight, gain=0.1, generator=init_gen)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            final_linear = residual_linears[-1]
            nn.init.zeros_(final_linear.weight)
            if final_linear.bias is not None:
                nn.init.zeros_(final_linear.bias)
            return

    def register_task_ids(self, task_ids: Sequence[str]) -> None:
        if not (self.task_embedding_enabled or self.task_radius_enabled):
            return

        unique_task_ids = [str(task_id) for task_id in task_ids]
        unique_task_ids = sorted(dict.fromkeys(unique_task_ids))
        if not unique_task_ids:
            raise ValueError(
                "task-specific condition parameters require at least one task id"
            )

        if self.task_embeddings is not None or self.task_radius_offsets is not None:
            if sorted(self._task_id_to_index.keys()) != unique_task_ids:
                raise ValueError(
                    "ConditionEncoder task ids were already registered with a different set."
                )
            return

        self._task_id_to_index = {
            task_id: idx for idx, task_id in enumerate(unique_task_ids)
        }
        module_device = next(self.parameters()).device
        module_dtype = next(self.parameters()).dtype
        if self.task_embedding_enabled:
            self.task_embeddings = nn.Embedding(
                len(unique_task_ids),
                self.view_dim,
                device=module_device,
                dtype=module_dtype,
            )
            nn.init.zeros_(self.task_embeddings.weight)
            if self.task_embedding_init_std > 0.0:
                with torch.no_grad():
                    self.task_embeddings.weight.normal_(
                        mean=0.0, std=self.task_embedding_init_std
                    )

        if self.task_radius_enabled:
            if self.task_radius_init <= 0.0:
                raise ValueError(
                    f"task_radius_init must be positive, got {self.task_radius_init!r}"
                )
            self.task_radius_offsets = nn.Embedding(
                len(unique_task_ids),
                1,
                device=module_device,
                dtype=module_dtype,
            )
            init_log_radius = math.log(self.task_radius_init)
            nn.init.constant_(self.task_radius_offsets.weight, init_log_radius)
            if self.task_radius_init_std > 0.0:
                with torch.no_grad():
                    self.task_radius_offsets.weight.normal_(
                        mean=init_log_radius,
                        std=self.task_radius_init_std,
                    )

    def has_task_embeddings(self) -> bool:
        return self.task_embeddings is not None

    def has_task_radii(self) -> bool:
        return self.task_radius_offsets is not None

    def _task_index_tensor(
        self,
        task_ids: Optional[Sequence[str]],
        device: torch.device,
    ) -> torch.Tensor:
        if task_ids is None:
            raise ValueError(
                "ConditionEncoder has task-specific parameters enabled, but no task_ids were provided."
            )
        try:
            indices = [self._task_id_to_index[str(task_id)] for task_id in task_ids]
        except KeyError as exc:
            raise KeyError(
                f"Unknown task_id for task-specific condition parameter: {exc.args[0]!r}"
            ) from exc
        return torch.tensor(indices, device=device, dtype=torch.long)

    def _lookup_task_delta(
        self,
        task_ids: Optional[Sequence[str]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if self.task_embeddings is None:
            return None
        index_tensor = self._task_index_tensor(task_ids, device)
        return self.task_embeddings(index_tensor).to(dtype=dtype)

    def _lookup_task_radius_scale(
        self,
        task_ids: Optional[Sequence[str]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if self.task_radius_offsets is None:
            return None
        index_tensor = self._task_index_tensor(task_ids, device)
        return self.task_radius_offsets(index_tensor).exp().to(dtype=dtype)

    def task_radius_values(
        self,
        task_ids: Sequence[str],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        include_global_radius: bool = True,
    ) -> Optional[torch.Tensor]:
        if self.task_radius_offsets is None:
            return None
        target_device = device
        if target_device is None:
            target_device = self.task_radius_offsets.weight.device
        target_dtype = dtype or self.task_radius_offsets.weight.dtype
        scales = self._lookup_task_radius_scale(task_ids, target_device, target_dtype)
        if scales is None:
            return None
        if include_global_radius:
            scales = scales * self.output_radius().to(device=target_device, dtype=target_dtype)
        return scales.squeeze(-1)

    def forward(
        self,
        pooled_hidden: torch.Tensor,
        task_ids: Optional[Sequence[str]] = None,
    ) -> torch.Tensor:
        """
        Args:
            pooled_hidden: (batch, input_dim) – mean-pooled hidden states
                           from the frozen backbone over condition tokens
        Returns:
            z: (batch, view_dim)
        """
        if self.head_type == "res_mlp":
            z = self.base_linear(pooled_hidden) + self.residual_mlp(pooled_hidden)
        else:
            z = self.mlp(pooled_hidden)
        task_delta = self._lookup_task_delta(task_ids, device=z.device, dtype=z.dtype)
        task_radius_scale = self._lookup_task_radius_scale(
            task_ids, device=z.device, dtype=z.dtype
        )
        if self.output_normalize:
            z = F.normalize(z, p=2, dim=-1, eps=1e-8)
            if task_delta is not None:
                z = F.normalize(z + task_delta, p=2, dim=-1, eps=1e-8)
        elif task_delta is not None:
            z = z + task_delta
        if self.output_normalize:
            z = z * self.output_radius().to(device=z.device, dtype=z.dtype)
        if task_radius_scale is not None:
            z = z * task_radius_scale
        return z

    def output_radius(self) -> torch.Tensor:
        if self.output_radius_learnable and self.log_output_radius is not None:
            return self.log_output_radius.exp()
        return self.fixed_output_radius

    def output_radius_value(self) -> float:
        return float(self.output_radius().detach().cpu().item())


def pool_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Mean-pool hidden_states over non-padding tokens.

    Args:
        hidden_states:  (batch, seq_len, hidden_dim)
        attention_mask: (batch, seq_len) with 1 for real tokens, 0 for padding
    Returns:
        pooled: (batch, hidden_dim)
    """
    mask = attention_mask.unsqueeze(-1).float()           # (B, S, 1)
    sum_hidden = (hidden_states * mask).sum(dim=1)        # (B, H)
    count = mask.sum(dim=1).clamp(min=1e-9)              # (B, 1)
    return sum_hidden / count


class CachedConditionEmbeddings:
    """
    In-memory store for pre-computed condition embeddings.
    Maps condition_id → pooled hidden state tensor.
    Avoids running the backbone twice during training.
    """

    def __init__(self):
        self._cache: dict[str, torch.Tensor] = {}

    def set(self, condition_id: str, embedding: torch.Tensor):
        self._cache[condition_id] = embedding.detach().cpu()

    def get(self, condition_id: str) -> Optional[torch.Tensor]:
        return self._cache.get(condition_id)

    def get_batch(
        self,
        condition_ids: list[str],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Retrieve a batch of embeddings by IDs and move to device.

        Args:
            condition_ids: list of length B
            device: target device
        Returns:
            (B, hidden_dim) tensor
        """
        tensors = [self._cache[cid] for cid in condition_ids]
        return torch.stack(tensors, dim=0).to(device)

    def has(self, condition_id: str) -> bool:
        return condition_id in self._cache

    def __len__(self):
        return len(self._cache)

    def save(self, path: str):
        torch.save(self._cache, path)

    @classmethod
    def load(cls, path: str) -> "CachedConditionEmbeddings":
        obj = cls()
        obj._cache = torch.load(path, map_location="cpu", weights_only=True)
        return obj


def get_condition_source(cfg) -> str:
    return cfg.get("condition_encoder", {}).get("source", "backbone_pooled")


def get_condition_input_dim(cfg, default_dim: int) -> int:
    return int(cfg.get("condition_encoder", {}).get("input_dim", default_dim))


def get_condition_head_type(cfg) -> str:
    return str(cfg.get("condition_encoder", {}).get("head_type", "mlp"))


def get_condition_output_normalize(cfg) -> bool:
    return bool(cfg.get("condition_encoder", {}).get("output_normalize", False))


def get_condition_output_radius_init(cfg) -> float:
    return float(cfg.get("condition_encoder", {}).get("output_radius_init", 1.0))


def get_condition_output_radius_learnable(cfg) -> bool:
    return bool(cfg.get("condition_encoder", {}).get("output_radius_learnable", False))


def get_condition_task_embedding_enabled(cfg) -> bool:
    return bool(cfg.get("condition_encoder", {}).get("task_embedding_enabled", False))


def get_condition_task_embedding_init_std(cfg) -> float:
    return float(cfg.get("condition_encoder", {}).get("task_embedding_init_std", 0.0))


def get_condition_task_radius_enabled(cfg) -> bool:
    return bool(cfg.get("condition_encoder", {}).get("task_radius_enabled", False))


def get_condition_task_radius_init(cfg) -> float:
    return float(cfg.get("condition_encoder", {}).get("task_radius_init", 1.0))


def get_condition_task_radius_init_std(cfg) -> float:
    return float(cfg.get("condition_encoder", {}).get("task_radius_init_std", 0.0))


def get_condition_cache_path(cfg) -> str:
    cond_cfg = cfg.get("condition_encoder", {})
    cache_path = cond_cfg.get("cache_path")
    if cache_path:
        return cache_path
    cache_name = cond_cfg.get("cache_name", "condition_embeddings.pt")
    return os.path.join(cfg.data.data_dir, cache_name)


def get_named_condition_cache_path(cfg, section_name: str) -> str:
    cond_cfg = cfg.get(section_name, {})
    cache_path = cond_cfg.get("cache_path")
    if cache_path:
        return cache_path
    cache_name = cond_cfg.get("cache_name", "condition_embeddings.pt")
    return os.path.join(cfg.data.data_dir, cache_name)


def load_condition_encoder_state(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, list[str]]:
    current = module.state_dict()
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in current:
            skipped.append(key)
            continue
        if current[key].shape != value.shape:
            skipped.append(key)
            continue
        filtered[key] = value

    missing, unexpected = module.load_state_dict(filtered, strict=False)
    return {
        "loaded": sorted(filtered.keys()),
        "missing": list(missing),
        "unexpected": list(unexpected),
        "skipped": skipped,
    }
