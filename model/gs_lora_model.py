"""
Geofield-LoRA Model Wrapper.

Wraps a Qwen2.5 (or any causal LM) backbone with Geofield-LoRA:
  1. Freezes all backbone parameters.
  2. Replaces target Linear layers (q_proj, v_proj) with GSLoRALinear.
  3. Adds a shared ConditionEncoder plus per-layer GaussianPrimitiveBanks.

Forward protocol:
  a. condition_hidden (pre-computed) → ConditionEncoder → z
  b. each layer computes alpha_l = PrimitiveBank_l(z)
  c. alpha_l is pushed to its corresponding Geofield-LoRA layer via _set_current_alpha()
  c. backbone.forward() runs normally; Geofield-LoRA layers read alpha from self._current_alpha
"""

import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from omegaconf import DictConfig

from .condition_encoder import (
    ConditionEncoder,
    load_condition_encoder_state,
    pool_hidden_states,
    get_condition_input_dim,
    get_condition_head_type,
    get_condition_output_normalize,
    get_condition_output_radius_init,
    get_condition_output_radius_learnable,
    get_condition_task_embedding_enabled,
    get_condition_task_embedding_init_std,
    get_condition_task_radius_enabled,
    get_condition_task_radius_init,
    get_condition_task_radius_init_std,
    get_condition_source,
)
from .primitives import GaussianPrimitiveBank
from .gs_lora_layer import GSLoRALinear


class GSLoRAModel(nn.Module):
    """
    Geofield-LoRA wrapper around a frozen causal LM backbone.

    Trainable parameters only:
      - ConditionEncoder MLP
      - Per-layer GaussianPrimitiveBanks (μ_l, log_diag_l, log_s_l, optional lower_offdiag_l)
      - GSLoRALinear U, V for each target layer
    """

    def __init__(
        self,
        cfg: DictConfig,
        backbone: nn.Module,
        tokenizer,
    ):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.alpha_lora_fp32 = bool(cfg.training.get("alpha_lora_fp32", False))

        hidden_size = backbone.config.hidden_size
        condition_input_dim = get_condition_input_dim(cfg, hidden_size)
        n_primitives = cfg.gslora.n_primitives
        rank = cfg.gslora.rank
        view_dim = cfg.gslora.view_dim
        lora_alpha = cfg.gslora.lora_alpha
        covariance_type = cfg.gslora.get("covariance_type", "diag")
        readout_mode = str(cfg.gslora.get("readout_mode", "point_softmax"))
        ray_geometry = str(cfg.gslora.get("ray_geometry", "shared_origin"))
        ray_softplus_tau = float(cfg.gslora.get("ray_softplus_tau", 0.1))
        ray_opacity_init = float(cfg.gslora.get("ray_opacity_init", 0.5))
        primitive_init_mode = str(cfg.gslora.get("primitive_init_mode", "random_normal"))
        primitive_init_radius = float(cfg.gslora.get("primitive_init_radius", 0.6))
        primitive_init_radius_jitter = float(
            cfg.gslora.get("primitive_init_radius_jitter", 0.0)
        )
        primitive_init_precision_scale = float(
            cfg.gslora.get("primitive_init_precision_scale", 1.0)
        )
        primitive_init_precision_log_jitter = float(
            cfg.gslora.get("primitive_init_precision_log_jitter", 0.0)
        )
        primitive_init_offdiag_std = float(
            cfg.gslora.get("primitive_init_offdiag_std", 0.0)
        )
        target_modules = list(cfg.model.target_modules)

        # ── Freeze backbone ──────────────────────────────────────────────────
        for param in self.backbone.parameters():
            param.requires_grad = False

        # ── Trainable modules ────────────────────────────────────────────────
        self.condition_encoder = ConditionEncoder(
            input_dim=condition_input_dim,
            hidden_dim=cfg.gslora.mlp_hidden,
            view_dim=view_dim,
            head_type=get_condition_head_type(cfg),
            output_normalize=get_condition_output_normalize(cfg),
            output_radius_init=get_condition_output_radius_init(cfg),
            output_radius_learnable=get_condition_output_radius_learnable(cfg),
            task_embedding_enabled=get_condition_task_embedding_enabled(cfg),
            task_embedding_init_std=get_condition_task_embedding_init_std(cfg),
            task_radius_enabled=get_condition_task_radius_enabled(cfg),
            task_radius_init=get_condition_task_radius_init(cfg),
            task_radius_init_std=get_condition_task_radius_init_std(cfg),
        )
        self._load_pretrained_condition_encoder()
        self.condition_source = get_condition_source(cfg)

        # Keep the base Geofield-LoRA initialization order identical to the standard
        # model so staged template experiments share the same task-side init.
        template_delta_cfg = cfg.get("template_delta_encoder", {})
        self.template_delta_enabled = bool(template_delta_cfg.get("enabled", False))
        self.template_runtime_enabled = self.template_delta_enabled
        self.template_delta_encoder = None
        self.template_delta_mode = str(template_delta_cfg.get("bias_mode", "linear"))
        self.template_bias_mlp = None
        self.template_gamma = None

        # ── Inject Geofield-LoRA layers ────────────────────────────────────────────
        self._gs_lora_layers: Dict[str, GSLoRALinear] = {}
        self._inject_gs_lora(target_modules, n_primitives, rank, lora_alpha)
        self._primitive_bank_keys = {
            name: f"layer_{idx}" for idx, name in enumerate(self._gs_lora_layers)
        }
        self.primitive_banks = nn.ModuleDict({
            key: GaussianPrimitiveBank(
                n_primitives=n_primitives,
                view_dim=view_dim,
                covariance_type=covariance_type,
                center_init_mode=primitive_init_mode,
                center_init_radius=primitive_init_radius,
                center_init_radius_jitter=primitive_init_radius_jitter,
                precision_init_scale=primitive_init_precision_scale,
                precision_init_log_jitter=primitive_init_precision_log_jitter,
                offdiag_init_std=primitive_init_offdiag_std,
                readout_mode=readout_mode,
                ray_geometry=ray_geometry,
                ray_softplus_tau=ray_softplus_tau,
                ray_opacity_init=ray_opacity_init,
            )
            for key in self._primitive_bank_keys.values()
        })

        # nn.ModuleList for proper parameter registration
        self.gs_lora_list = nn.ModuleList(list(self._gs_lora_layers.values()))

        # Initialize the template branch only after the standard GS modules have
        # consumed their RNG so the task-side init matches the baseline run.
        if self.template_delta_enabled:
            template_input_dim = int(template_delta_cfg.get("input_dim", condition_input_dim))
            template_hidden_dim = int(template_delta_cfg.get("mlp_hidden", cfg.gslora.mlp_hidden))
            template_head_type = str(template_delta_cfg.get("head_type", get_condition_head_type(cfg)))
            self.template_delta_encoder = ConditionEncoder(
                input_dim=template_input_dim,
                hidden_dim=template_hidden_dim,
                view_dim=view_dim,
                head_type=template_head_type,
                output_normalize=False,
            )
            gamma_init = float(template_delta_cfg.get("gamma_init", 0.0))
            self.template_gamma = nn.Parameter(torch.tensor(gamma_init))
            if self.template_delta_mode == "mlp_concat":
                fusion_hidden_dim = int(
                    template_delta_cfg.get("fusion_hidden_dim", max(view_dim * 4, 128))
                )
                self.template_bias_mlp = nn.Sequential(
                    nn.Linear(view_dim * 2, fusion_hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(fusion_hidden_dim),
                    nn.Linear(fusion_hidden_dim, view_dim),
                )
            elif self.template_delta_mode != "linear":
                raise ValueError(
                    f"Unsupported template_delta_encoder.bias_mode={self.template_delta_mode!r}. "
                    "Expected one of: 'linear', 'mlp_concat'."
                )

        print(
            f"[Geofield-LoRA] condition_source={self.condition_source}, "
            f"condition_input_dim={condition_input_dim}, "
            f"condition_head={self.condition_encoder.head_type}, "
            f"condition_norm={'on' if self.condition_encoder.output_normalize else 'off'}, "
            f"condition_rho={self.condition_encoder.output_radius_value():.3f}"
            + ("(learnable)" if self.condition_encoder.output_radius_learnable else "(fixed)")
            + (
                ", task_view=on"
                if self.condition_encoder.task_embedding_enabled
                else ", task_view=off"
            )
            + (
                ", task_rho=on"
                if self.condition_encoder.task_radius_enabled
                else ", task_rho=off"
            )
            + ", "
            f"primitive_readout={readout_mode}, ray_geometry={ray_geometry}, "
            f"primitive_init={primitive_init_mode}"
            + (
                f"(r={primitive_init_radius:.3f}, jitter={primitive_init_radius_jitter:.3f}, p={primitive_init_precision_scale:.3f})"
                if primitive_init_mode == "radial_shell"
                else ""
            )
            + ", "
            f"template_delta={'on' if self.template_delta_enabled else 'off'}"
            + (
                f"({self.template_delta_mode}, gamma_init={float(self.template_gamma.detach().cpu()):.3f})"
                if self.template_delta_enabled and self.template_gamma is not None
                else ""
            )
        )
        self.primitive_readout_mode = readout_mode
        self.print_trainable_parameters()

    # ── Injection ────────────────────────────────────────────────────────────

    def _inject_gs_lora(
        self,
        target_modules: List[str],
        n_primitives: int,
        rank: int,
        lora_alpha: float,
    ):
        replaced = 0
        for name, module in list(self.backbone.named_modules()):
            module_leaf = name.split(".")[-1]
            if module_leaf not in target_modules:
                continue
            if not isinstance(module, nn.Linear):
                continue

            parent = self._get_parent(self.backbone, name)
            child_name = name.split(".")[-1]
            gs_layer = GSLoRALinear.from_linear(
                module, n_primitives=n_primitives, rank=rank, lora_alpha=lora_alpha
            )
            gs_layer = gs_layer.to(device=module.weight.device, dtype=module.weight.dtype)
            setattr(parent, child_name, gs_layer)
            self._gs_lora_layers[name] = gs_layer
            replaced += 1

        if replaced == 0:
            raise RuntimeError(
                f"No target modules found: {target_modules}. "
                "Check cfg.model.target_modules matches the backbone's layer names. "
                "If quantized loading is enabled, note this injector currently only "
                "wraps modules that are still exposed as torch.nn.Linear."
            )
        print(f"[Geofield-LoRA] Injected into {replaced} layers {target_modules}")

    @staticmethod
    def _get_parent(root: nn.Module, full_name: str) -> nn.Module:
        parts = full_name.split(".")
        parent = root
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent

    # ── Alpha broadcast ───────────────────────────────────────────────────────

    def _get_primitive_bank(self, layer_name: str) -> GaussianPrimitiveBank:
        return self.primitive_banks[self._primitive_bank_keys[layer_name]]

    def _load_pretrained_condition_encoder(self):
        cond_cfg = self.cfg.get("condition_encoder", {})
        pretrained_path = cond_cfg.get("pretrained_path")
        if not pretrained_path:
            return
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(
                f"condition_encoder.pretrained_path not found: {pretrained_path}"
            )

        state = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        cond_state = state.get("condition_encoder", state)
        load_info = load_condition_encoder_state(self.condition_encoder, cond_state)
        print(
            "[Geofield-LoRA] loaded pretrained condition encoder from "
            f"{pretrained_path} "
            f"(loaded={len(load_info['loaded'])}, skipped={len(load_info['skipped'])}, "
            f"missing={len(load_info['missing'])}, unexpected={len(load_info['unexpected'])})"
        )

    def _set_current_alpha(self, alpha: torch.Tensor | Dict[str, torch.Tensor]):
        """Push alpha to every Geofield-LoRA layer before backbone forward."""
        if isinstance(alpha, dict):
            for name, layer in self._gs_lora_layers.items():
                layer_alpha = alpha[name]
                alpha_dtype = torch.float32 if self.alpha_lora_fp32 else layer.U.dtype
                layer.set_alpha(layer_alpha.to(device=layer.U.device, dtype=alpha_dtype))
            return

        for layer in self._gs_lora_layers.values():
            alpha_dtype = torch.float32 if self.alpha_lora_fp32 else layer.U.dtype
            layer.set_alpha(alpha.to(device=layer.U.device, dtype=alpha_dtype))

    def register_task_ids(self, task_ids: List[str]) -> None:
        self.condition_encoder.register_task_ids(task_ids)
        if self.condition_encoder.has_task_embeddings():
            print(f"[Geofield-LoRA] Registered {len(task_ids)} task-specific view embeddings")
        if self.condition_encoder.has_task_radii():
            print(f"[Geofield-LoRA] Registered {len(task_ids)} task-specific rho values")

    def _clear_alpha(self):
        """Clear alpha after forward to avoid stale state."""
        for layer in self._gs_lora_layers.values():
            layer._current_alpha = None

    # ── Condition encoding ────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_conditions_batch(
        self,
        condition_input_ids: torch.Tensor,
        condition_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode a batch of condition texts with the frozen backbone (no LoRA delta).
        Use this offline to pre-compute condition embeddings.

        Returns:
            pooled: (B, hidden_size)
        """
        # Zero alpha → no Geofield-LoRA delta → pure frozen backbone
        B = condition_input_ids.shape[0]
        zero_alpha = torch.zeros(
            B, self.cfg.gslora.n_primitives,
            device=condition_input_ids.device,
            dtype=next(self.backbone.parameters()).dtype,
        )
        self._set_current_alpha(zero_alpha)
        outputs = self.backbone(
            input_ids=condition_input_ids,
            attention_mask=condition_attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        self._clear_alpha()
        hidden = outputs.hidden_states[-1]   # (B, seq, H)
        return pool_hidden_states(hidden, condition_attention_mask)

    def compute_alpha(
        self,
        condition_hidden: torch.Tensor,
        template_condition_hidden: Optional[torch.Tensor] = None,
        task_ids: Optional[List[str]] = None,
    ) -> tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        condition_hidden (B, H) → z (B, view_dim) → layer-specific alpha_l.
        """
        cond_param = next(self.condition_encoder.parameters())
        condition_hidden = condition_hidden.to(
            device=cond_param.device,
            dtype=cond_param.dtype,
        )
        z_task = self.condition_encoder(condition_hidden, task_ids=task_ids)
        delta_template = None
        z = z_task
        if (
            self.template_delta_enabled
            and self.template_runtime_enabled
            and template_condition_hidden is not None
        ):
            template_param = next(self.template_delta_encoder.parameters())
            template_condition_hidden = template_condition_hidden.to(
                device=template_param.device,
                dtype=template_param.dtype,
            )
            template_features = self.template_delta_encoder(template_condition_hidden)
            if self.template_delta_mode == "linear":
                delta_template = template_features
            else:
                delta_template = self.template_bias_mlp(
                    torch.cat([z_task, template_features], dim=-1)
                )
            gamma = self.template_gamma.to(device=z_task.device, dtype=z_task.dtype)
            z = z_task + gamma * delta_template
        alpha_map: Dict[str, torch.Tensor] = {}
        alpha_stack = []

        for name, layer in self._gs_lora_layers.items():
            primitive_bank = self._get_primitive_bank(name)
            z_layer = z.to(device=primitive_bank.mu.device, dtype=primitive_bank.mu.dtype)
            readout_weight_layer = primitive_bank(z_layer)
            alpha_layer = primitive_bank.normalize_for_regularizer(readout_weight_layer)
            alpha_map[name] = readout_weight_layer
            alpha_stack.append(alpha_layer.to(device=z.device, dtype=z.dtype))
        layer_alphas = torch.stack(alpha_stack, dim=1)  # (B, L, K)
        alpha_mean = layer_alphas.mean(dim=1)           # (B, K)
        return alpha_map, layer_alphas, alpha_mean, z, z_task, delta_template

    # ── Main forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        condition_hidden: torch.Tensor,
        task_ids: Optional[List[str]] = None,
        labels: Optional[torch.Tensor] = None,
        condition_hidden_2: Optional[torch.Tensor] = None,
        return_logits: bool = False,
    ) -> Dict[str, Any]:
        """
        Args:
            input_ids:            (B, seq)
            attention_mask:       (B, seq)
            condition_hidden:     (B, H) pre-computed condition embedding
            labels:               (B, seq) token labels
            condition_hidden_2:   (B, H) second template embedding (for L_cons)

        Returns:
            dict: loss, [logits], alpha, layer_alphas, z, [alpha_2, layer_alphas_2, z_2]
        """
        template_condition_hidden = (
            condition_hidden_2
            if self.template_delta_enabled and self.template_runtime_enabled
            else None
        )
        alpha_map, layer_alphas, alpha, z, z_task, delta_template = self.compute_alpha(
            condition_hidden,
            template_condition_hidden=template_condition_hidden,
            task_ids=task_ids,
        )
        self._set_current_alpha(alpha_map)
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        self._clear_alpha()

        result: Dict[str, Any] = {
            "loss": outputs.loss,
            "alpha": alpha,
            "layer_alphas": layer_alphas,
            "z": z,
            "z_task": z_task,
        }
        if delta_template is not None:
            result["delta_template"] = delta_template
        if return_logits:
            result["logits"] = outputs.logits

        if condition_hidden_2 is not None and not self.template_delta_enabled:
            _, layer_alphas_2, alpha_2, z_2, _, _ = self.compute_alpha(
                condition_hidden_2,
                task_ids=task_ids,
            )
            result["alpha_2"] = alpha_2
            result["layer_alphas_2"] = layer_alphas_2
            result["z_2"] = z_2

        return result

    # ── Generation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        condition_hidden: torch.Tensor,
        task_ids: Optional[List[str]] = None,
        condition_hidden_2: Optional[torch.Tensor] = None,
        max_new_tokens: int = 64,
        **kwargs,
    ) -> torch.Tensor:
        template_condition_hidden = (
            condition_hidden_2
            if self.template_delta_enabled and self.template_runtime_enabled
            else None
        )
        alpha_map, _, _, _, _, _ = self.compute_alpha(
            condition_hidden,
            template_condition_hidden=template_condition_hidden,
            task_ids=task_ids,
        )
        self._set_current_alpha(alpha_map)
        out = self.backbone.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            **kwargs,
        )
        self._clear_alpha()
        return out

    # ── Data-driven initialization ──────────────────────────────────────────

    @torch.no_grad()
    def initialize_from_condition_data(
        self,
        condition_embeddings: dict[str, torch.Tensor],
        gain: float = 1.0,
    ) -> dict[str, Any]:
        if self.condition_encoder.head_type != "linear":
            raise ValueError(
                f"Data-driven init only supports head_type='linear', "
                f"got {self.condition_encoder.head_type!r}"
            )

        keys = sorted(condition_embeddings.keys())
        embeddings = torch.stack(
            [condition_embeddings[k].float() for k in keys], dim=0
        )

        view_dim = self.condition_encoder.view_dim
        mean = embeddings.mean(dim=0)
        centered = embeddings - mean

        _, S, Vt = torch.linalg.svd(centered, full_matrices=False)
        pca_basis = Vt[:view_dim]
        z_pca = centered @ pca_basis.T

        target_std = 1.0 / torch.sqrt(
            torch.tensor(float(view_dim), dtype=z_pca.dtype, device=z_pca.device)
        )
        per_dim_std = z_pca.std(dim=0).clamp(min=1e-8)
        scale = target_std / per_dim_std

        weight = (gain * scale).unsqueeze(1) * pca_basis
        bias = -(weight @ mean)

        linear = self.condition_encoder.mlp
        linear.weight.copy_(
            weight.to(dtype=linear.weight.dtype, device=linear.weight.device)
        )
        linear.bias.copy_(bias.to(dtype=linear.bias.dtype, device=linear.bias.device))

        z_scaled = z_pca * (gain * scale).unsqueeze(0)
        centers = self._kmeans(
            z_scaled,
            self.cfg.gslora.n_primitives,
            n_iter=int(self.cfg.gslora.get("init_kmeans_iters", 100)),
            seed=int(self.cfg.training.seed),
        )

        init_center_shrink = float(self.cfg.gslora.get("init_center_shrink", 1.0))
        init_center_shrink = float(max(0.0, min(1.0, init_center_shrink)))
        field_center = z_scaled.mean(dim=0)
        raw_centers = centers
        if init_center_shrink != 1.0:
            centers = field_center.unsqueeze(0) + init_center_shrink * (
                raw_centers - field_center.unsqueeze(0)
            )

        init_covariance_mode = str(
            self.cfg.gslora.get("init_covariance_mode", "identity")
        ).lower()
        init_covariance_shrinkage = float(
            self.cfg.gslora.get("init_covariance_shrinkage", 0.75)
        )
        init_covariance_min_var = float(
            self.cfg.gslora.get("init_covariance_min_var", 1e-3)
        )
        init_layer_center_jitter = float(
            self.cfg.gslora.get("init_layer_center_jitter", 0.0)
        )
        init_strength_from_cluster_mass = bool(
            self.cfg.gslora.get("init_strength_from_cluster_mass", False)
        )
        init_scale_knn = int(self.cfg.gslora.get("init_scale_knn", 3))
        init_scale_knn_mult = float(self.cfg.gslora.get("init_scale_knn_mult", 0.8))
        init_scale_min = float(self.cfg.gslora.get("init_scale_min", 1e-3))
        init_scale_max = self.cfg.gslora.get("init_scale_max", None)
        init_scale_max = None if init_scale_max is None else float(init_scale_max)
        init_scale_log_jitter = float(
            self.cfg.gslora.get("init_scale_log_jitter", 0.0)
        )
        init_random_rotation = bool(self.cfg.gslora.get("init_random_rotation", False))
        init_covariance_offdiag_std = float(
            self.cfg.gslora.get("init_covariance_offdiag_std", 0.0)
        )

        assignments = torch.cdist(z_scaled, centers).argmin(dim=1)
        counts = torch.bincount(assignments, minlength=self.cfg.gslora.n_primitives)
        precision_chol = None
        diag_precision = None
        if init_covariance_mode in {"cluster", "cluster_cov", "cluster_full"}:
            precision_chol = self._cluster_precision_cholesky(
                z_scaled,
                centers,
                assignments,
                shrinkage=init_covariance_shrinkage,
                min_var=init_covariance_min_var,
            )
            diag_precision = torch.diagonal(
                torch.matmul(precision_chol, precision_chol.transpose(-1, -2)),
                dim1=-2,
                dim2=-1,
            )
        elif init_covariance_mode in {
            "knn",
            "knn_iso",
            "knn_isotropic",
            "knn_aniso",
            "knn_random_aniso",
        }:
            random_rotation = init_random_rotation or init_covariance_mode in {
                "knn_aniso",
                "knn_random_aniso",
            }
            precision_chol = self._knn_precision_cholesky(
                centers,
                n_neighbors=init_scale_knn,
                scale_mult=init_scale_knn_mult,
                min_scale=init_scale_min,
                max_scale=init_scale_max,
                log_scale_jitter=init_scale_log_jitter,
                random_rotation=random_rotation,
                seed=int(self.cfg.training.seed),
            )
            diag_precision = torch.diagonal(
                torch.matmul(precision_chol, precision_chol.transpose(-1, -2)),
                dim1=-2,
                dim2=-1,
            )

        for layer_idx, bank in enumerate(self.primitive_banks.values()):
            bank_centers = centers.to(dtype=bank.mu.dtype, device=bank.mu.device)
            bank.mu.copy_(bank_centers)
            if init_layer_center_jitter > 0.0:
                jitter_gen = torch.Generator(device=bank.mu.device)
                jitter_gen.manual_seed(int(self.cfg.training.seed) + layer_idx)
                noise = torch.empty_like(bank.mu).normal_(
                    mean=0.0,
                    std=init_layer_center_jitter,
                    generator=jitter_gen,
                )
                bank.mu.add_(noise)
            if precision_chol is not None:
                bank_chol = precision_chol.to(
                    dtype=bank.mu.dtype,
                    device=bank.mu.device,
                )
                if bank.lower_offdiag is not None:
                    diag = torch.diagonal(bank_chol, dim1=-2, dim2=-1).clamp_min(1e-8)
                    bank.log_diag.copy_(2.0 * torch.log(diag))
                    bank.lower_offdiag.copy_(bank_chol[:, bank._tril_i, bank._tril_j])
                    if init_covariance_offdiag_std > 0.0:
                        offdiag_gen = torch.Generator(device=bank.mu.device)
                        offdiag_gen.manual_seed(
                            int(self.cfg.training.seed) + 100_000 + layer_idx
                        )
                        bank.lower_offdiag.add_(
                            torch.empty_like(bank.lower_offdiag).normal_(
                                mean=0.0,
                                std=init_covariance_offdiag_std,
                                generator=offdiag_gen,
                            )
                        )
                elif diag_precision is not None:
                    bank_diag_precision = diag_precision.to(
                        dtype=bank.log_diag.dtype,
                        device=bank.log_diag.device,
                    )
                    bank.log_diag.copy_(torch.log(bank_diag_precision.clamp_min(1e-8)))
            if init_strength_from_cluster_mass:
                mass = counts.float().to(device=bank.log_s.device, dtype=bank.log_s.dtype)
                mass = (mass / mass.mean().clamp_min(1e-8)).clamp_min(0.1)
                bank.log_s.copy_(torch.log(torch.expm1(mass) + 1e-8))

        explained_var = (S[:view_dim] ** 2).sum() / (S ** 2).sum()
        z_norms = z_scaled.norm(dim=-1)
        center_pairwise = torch.pdist(centers)
        info = {
            "n_conditions": len(keys),
            "explained_variance_ratio": explained_var.item(),
            "z_mean_norm": z_norms.mean().item(),
            "z_std_norm": z_norms.std().item(),
            "centers_mean_norm": centers.norm(dim=-1).mean().item(),
            "centers_pairwise_mean": center_pairwise.mean().item(),
            "raw_centers_mean_norm": raw_centers.norm(dim=-1).mean().item(),
            "init_center_shrink": init_center_shrink,
            "cluster_count_min": counts.min().item(),
            "cluster_count_max": counts.max().item(),
            "init_covariance_mode": init_covariance_mode,
        }
        print(
            f"[Geofield-LoRA] Data-driven init: N={len(keys)}, PCA({view_dim}), "
            f"var={explained_var.item():.1%}, "
            f"|z|={z_norms.mean().item():.3f}±{z_norms.std().item():.3f}, "
            f"|mu|={centers.norm(dim=-1).mean().item():.3f}, "
            f"shrink={init_center_shrink:.2f}, "
            f"pair(mu)={center_pairwise.mean().item():.3f}, "
            f"cluster={counts.min().item()}..{counts.max().item()}, "
            f"cov_init={init_covariance_mode}"
        )
        return info

    @staticmethod
    def _kmeans(
        data: torch.Tensor,
        k: int,
        n_iter: int = 100,
        seed: int = 0,
    ) -> torch.Tensor:
        n, dim = data.shape
        gen = torch.Generator(device=data.device)
        gen.manual_seed(seed)

        centers = torch.empty(k, dim, dtype=data.dtype, device=data.device)
        first_idx = torch.randint(n, (1,), generator=gen, device=data.device).item()
        centers[0] = data[first_idx]
        for i in range(1, k):
            dists = torch.cdist(data, centers[:i]).min(dim=1).values
            probs = dists / dists.sum().clamp(min=1e-8)
            idx = torch.multinomial(probs, 1, generator=gen).item()
            centers[i] = data[idx]

        for _ in range(n_iter):
            dists = torch.cdist(data, centers)
            assignments = dists.argmin(dim=1)
            new_centers = torch.zeros_like(centers)
            for j in range(k):
                mask = assignments == j
                if mask.any():
                    new_centers[j] = data[mask].mean(dim=0)
                else:
                    idx = torch.randint(n, (1,), generator=gen, device=data.device).item()
                    new_centers[j] = data[idx]
            if torch.allclose(new_centers, centers, atol=1e-6):
                break
            centers = new_centers
        return centers

    @staticmethod
    def _knn_precision_cholesky(
        centers: torch.Tensor,
        n_neighbors: int,
        scale_mult: float,
        min_scale: float,
        max_scale: Optional[float] = None,
        log_scale_jitter: float = 0.0,
        random_rotation: bool = False,
        seed: int = 0,
    ) -> torch.Tensor:
        n_primitives, view_dim = centers.shape
        if n_primitives <= 1:
            sigma = centers.new_full((n_primitives,), max(float(min_scale), 1.0))
        else:
            k = int(max(1, min(int(n_neighbors), n_primitives - 1)))
            distances = torch.cdist(centers, centers)
            inf = torch.full_like(distances[:, :1], float("inf"))
            self_idx = torch.arange(n_primitives, device=centers.device).view(-1, 1)
            distances = distances.scatter(1, self_idx, inf)
            knn = distances.topk(k, largest=False, dim=1).values
            sigma = knn.mean(dim=1) * float(scale_mult)
            sigma = sigma.clamp_min(float(min_scale))
            if max_scale is not None:
                sigma = sigma.clamp_max(float(max_scale))
        axis_sigma = sigma.unsqueeze(-1).expand(n_primitives, view_dim).clone()
        if log_scale_jitter > 0.0:
            gen = torch.Generator(device=centers.device)
            gen.manual_seed(int(seed) + 200_000)
            noise = torch.empty_like(axis_sigma).normal_(
                mean=0.0,
                std=float(log_scale_jitter),
                generator=gen,
            )
            noise = noise - noise.mean(dim=-1, keepdim=True)
            axis_sigma = axis_sigma * torch.exp(noise)
            if max_scale is not None:
                axis_sigma = axis_sigma.clamp_max(float(max_scale))

        inv_sigma = 1.0 / axis_sigma.clamp_min(1e-8)
        if not random_rotation:
            chol = centers.new_zeros(n_primitives, view_dim, view_dim)
            idx = torch.arange(view_dim, device=centers.device)
            chol[:, idx, idx] = inv_sigma
            return chol

        gen = torch.Generator(device=centers.device)
        gen.manual_seed(int(seed) + 300_000)
        chol_list = []
        for primitive_idx in range(n_primitives):
            random_matrix = torch.empty(
                view_dim,
                view_dim,
                dtype=centers.dtype,
                device=centers.device,
            ).normal_(mean=0.0, std=1.0, generator=gen)
            q, r = torch.linalg.qr(random_matrix)
            signs = torch.sign(torch.diagonal(r)).clamp(min=0.0) * 2.0 - 1.0
            q = q * signs.unsqueeze(0)
            precision = q @ torch.diag(inv_sigma[primitive_idx] ** 2) @ q.transpose(0, 1)
            chol_list.append(
                GSLoRAModel._safe_cholesky(precision, min_var=float(min_scale) * 1e-4)
            )
        return torch.stack(chol_list, dim=0)

    @staticmethod
    def _safe_cholesky(matrix: torch.Tensor, min_var: float) -> torch.Tensor:
        eye = torch.eye(matrix.size(-1), dtype=matrix.dtype, device=matrix.device)
        jitter = float(min_var)
        sym = 0.5 * (matrix + matrix.transpose(-1, -2))
        for _ in range(6):
            try:
                return torch.linalg.cholesky(sym + jitter * eye)
            except RuntimeError:
                jitter *= 10.0
        return torch.linalg.cholesky(sym + jitter * eye)

    @classmethod
    def _cluster_precision_cholesky(
        cls,
        z_scaled: torch.Tensor,
        centers: torch.Tensor,
        assignments: torch.Tensor,
        shrinkage: float,
        min_var: float,
    ) -> torch.Tensor:
        view_dim = z_scaled.size(-1)
        n_primitives = centers.size(0)
        eye = torch.eye(view_dim, dtype=z_scaled.dtype, device=z_scaled.device)
        centered_global = z_scaled - z_scaled.mean(dim=0, keepdim=True)
        denom = max(int(z_scaled.size(0)) - 1, 1)
        global_cov = centered_global.transpose(0, 1) @ centered_global / denom
        global_cov = 0.5 * (global_cov + global_cov.transpose(0, 1))
        global_cov = global_cov + float(min_var) * eye

        chol_list = []
        shrink = float(max(0.0, min(1.0, shrinkage)))
        for primitive_idx in range(n_primitives):
            mask = assignments == primitive_idx
            members = z_scaled[mask]
            if members.size(0) >= 2:
                diff = members - centers[primitive_idx].unsqueeze(0)
                cov = diff.transpose(0, 1) @ diff / max(int(members.size(0)) - 1, 1)
            elif members.size(0) == 1:
                diff = (members[0] - centers[primitive_idx]).unsqueeze(1)
                cov = diff @ diff.transpose(0, 1)
            else:
                cov = global_cov
            cov = 0.5 * (cov + cov.transpose(0, 1))
            cov = (1.0 - shrink) * cov + shrink * global_cov
            cov = cov + float(min_var) * eye
            cov_chol = cls._safe_cholesky(cov, min_var=min_var)
            precision = torch.cholesky_inverse(cov_chol)
            chol_list.append(cls._safe_cholesky(precision, min_var=min_var))
        return torch.stack(chol_list, dim=0)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def print_trainable_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        pct = 100 * trainable / max(total, 1)
        print(f"[Geofield-LoRA] Trainable: {trainable:,} / {total:,} ({pct:.3f}%)")

    def get_primitive_stats(self, alpha: torch.Tensor) -> dict:
        first_bank = next(iter(self.primitive_banks.values()), None)
        if first_bank is None:
            return {}
        return first_bank.get_primitive_stats(alpha)

    def save_adapter(self, path: str):
        """Save only trainable (adapter) parameters."""
        os.makedirs(path, exist_ok=True)
        state = {
            "condition_encoder": self.condition_encoder.state_dict(),
            "primitive_banks": self.primitive_banks.state_dict(),
            "gs_lora_layers": {
                k: v.state_dict() for k, v in self._gs_lora_layers.items()
            },
        }
        if self.template_delta_encoder is not None:
            state["template_delta_encoder"] = self.template_delta_encoder.state_dict()
        if self.template_bias_mlp is not None:
            state["template_bias_mlp"] = self.template_bias_mlp.state_dict()
        if self.template_gamma is not None:
            state["template_gamma"] = self.template_gamma.detach().cpu()
        torch.save(state, os.path.join(path, "adapter.pt"))
        # Save config separately as YAML
        from omegaconf import OmegaConf
        OmegaConf.save(self.cfg, os.path.join(path, "config.yaml"))
        print(f"[Geofield-LoRA] Adapter saved to {path}")

    def load_adapter(self, path: str):
        """Load adapter parameters from a saved checkpoint."""
        adapter_path = path
        if os.path.isdir(adapter_path):
            adapter_path = os.path.join(adapter_path, "adapter.pt")
        state = torch.load(adapter_path, map_location="cpu", weights_only=True)
        self.condition_encoder.load_state_dict(state["condition_encoder"])
        if self.template_delta_encoder is not None and "template_delta_encoder" in state:
            self.template_delta_encoder.load_state_dict(state["template_delta_encoder"])
        if self.template_bias_mlp is not None and "template_bias_mlp" in state:
            self.template_bias_mlp.load_state_dict(state["template_bias_mlp"])
        if self.template_gamma is not None and "template_gamma" in state:
            self.template_gamma.data.copy_(
                state["template_gamma"].to(
                    device=self.template_gamma.device,
                    dtype=self.template_gamma.dtype,
                )
            )
        if "primitive_banks" in state:
            try:
                self.primitive_banks.load_state_dict(state["primitive_banks"])
            except RuntimeError:
                load_result = self.primitive_banks.load_state_dict(
                    state["primitive_banks"],
                    strict=False,
                )
                allowed_missing = [
                    key for key in load_result.missing_keys
                    if key.endswith("lower_offdiag")
                ]
                allowed_unexpected = [
                    key for key in load_result.unexpected_keys
                    if key.endswith("lower_offdiag")
                ]
                if (
                    len(allowed_missing) != len(load_result.missing_keys)
                    or len(allowed_unexpected) != len(load_result.unexpected_keys)
                ):
                    raise
                if allowed_missing or allowed_unexpected:
                    print(
                        "[Geofield-LoRA] primitive_banks loaded with covariance compatibility: "
                        f"missing={allowed_missing}, unexpected={allowed_unexpected}"
                    )
        elif "primitive_bank" in state:
            for primitive_bank in self.primitive_banks.values():
                primitive_bank.load_state_dict(state["primitive_bank"])
        for name, layer_state in state["gs_lora_layers"].items():
            if name in self._gs_lora_layers:
                self._gs_lora_layers[name].load_state_dict(layer_state)
        print(f"[Geofield-LoRA] Adapter loaded from {adapter_path}")

    # ── Factory ───────────────────────────────────────────────────────────────

    @staticmethod
    def _as_torch_device(candidate: Any) -> Optional[torch.device]:
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

    @classmethod
    def from_pretrained(
        cls,
        cfg: DictConfig,
        device_map: Any = None,
        device: Any = None,
    ) -> "GSLoRAModel":
        model_name = cfg.model.backbone
        local_path = os.path.join(cfg.data.model_dir, model_name.split("/")[-1])
        model_path = local_path if os.path.isdir(local_path) else model_name

        print(f"[Geofield-LoRA] Loading backbone: {model_path}")

        bnb_config = None
        if cfg.model.get("load_in_4bit", False):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif cfg.model.get("load_in_8bit", False):
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        dtype = getattr(torch, cfg.model.dtype)
        attn_implementation = cfg.model.get("attn_implementation", None)
        requested_device = cls._as_torch_device(device)
        single_device_from_map = cls._as_torch_device(device_map)
        hf_device_map = None
        direct_load_device = None
        if device_map is not None and single_device_from_map is None:
            hf_device_map = device_map
        else:
            direct_load_device = requested_device or (
                single_device_from_map if bnb_config is None else None
            )

        model_kwargs = {
            "quantization_config": bnb_config,
            "dtype": dtype,
            "trust_remote_code": True,
            "use_safetensors": True,
            "weights_only": True,
        }
        if hf_device_map is not None:
            model_kwargs["device_map"] = hf_device_map
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        if direct_load_device is not None:
            print(f"[Geofield-LoRA] Loading via direct single-device path on {direct_load_device}")
        elif hf_device_map is not None:
            print(f"[Geofield-LoRA] Loading via HF device_map={hf_device_map}")

        backbone = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs,
        )
        if direct_load_device is not None:
            backbone = backbone.to(device=direct_load_device, dtype=dtype)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = cls(cfg=cfg, backbone=backbone, tokenizer=tokenizer)

        # Move trainable adapter modules to same device as backbone. By default
        # we preserve the historical behavior and use the backbone dtype. Geometry
        # probes can opt into fp32 task-side modules without changing old configs.
        backbone_param = next(backbone.parameters())
        backbone_device = backbone_param.device
        backbone_dtype = backbone_param.dtype
        task_side_fp32 = bool(
            cfg.training.get(
                "task_side_fp32",
                cfg.training.get("primitive_bank_fp32", False),
            )
        )
        primitive_bank_fp32 = bool(cfg.training.get("primitive_bank_fp32", task_side_fp32))
        task_side_dtype = torch.float32 if task_side_fp32 else backbone_dtype
        model.condition_encoder = model.condition_encoder.to(
            device=backbone_device,
            dtype=task_side_dtype,
        )
        if model.template_delta_encoder is not None:
            model.template_delta_encoder = model.template_delta_encoder.to(
                device=backbone_device,
                dtype=task_side_dtype,
            )
        if model.template_bias_mlp is not None:
            model.template_bias_mlp = model.template_bias_mlp.to(
                device=backbone_device,
                dtype=task_side_dtype,
            )
        if model.template_gamma is not None:
            model.template_gamma = nn.Parameter(
                model.template_gamma.to(device=backbone_device, dtype=task_side_dtype)
            )

        for name, layer in model._gs_lora_layers.items():
            primitive_bank = model._get_primitive_bank(name)
            primitive_dtype = torch.float32 if primitive_bank_fp32 else layer.U.dtype
            primitive_bank.to(device=layer.U.device, dtype=primitive_dtype)

        return model
