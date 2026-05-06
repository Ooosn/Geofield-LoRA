"""
MLP-Gated LoRA Mixture baseline.

This is the critical non-Gaussian comparison for Geofield-LoRA:
  - Same backbone
  - Same condition encoder z
  - Same K low-rank atoms per target layer
  - Same structural losses / training protocol
  - Only difference: per-layer mixture weights come from an MLP+softmax
    router instead of Gaussian visibility kernels
"""

import os
import math
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .condition_encoder import (
    ConditionEncoder,
    load_condition_encoder_state,
    get_condition_head_type,
    get_condition_input_dim,
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
from .gs_lora_layer import GSLoRALinear
from .gs_lora_model import GSLoRAModel


class MLPPrimitiveRouter(nn.Module):
    """
    Lightweight MLP router that maps z -> alpha via softmax.

    The default hidden size is chosen to keep per-layer router capacity
    close to the Gaussian bank parameter count:

      Gaussian params per layer ≈ K * (2 * view_dim + 1)
      2-layer MLP params       ≈ hidden * (view_dim + K + 1) + K
    """

    def __init__(
        self,
        n_primitives: int,
        view_dim: int,
        hidden_dim: int,
        top_k: int | None = None,
    ):
        super().__init__()
        self.n_primitives = n_primitives
        self.view_dim = view_dim
        self.hidden_dim = hidden_dim
        self.top_k = None if top_k is None else int(top_k)
        self.fc1 = nn.Linear(view_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_primitives)
        self._init_weights()

    @staticmethod
    def default_hidden_dim(n_primitives: int, view_dim: int) -> int:
        approx_gaussian_params = n_primitives * (2 * view_dim + 1)
        hidden = round((approx_gaussian_params - n_primitives) / (view_dim + n_primitives + 1))
        return max(4, hidden)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.1)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.fc2(F.gelu(self.fc1(z)))
        probs = torch.softmax(logits, dim=-1)
        if self.top_k is None or self.top_k <= 0 or self.top_k >= self.n_primitives:
            return probs

        top_vals, top_idx = torch.topk(probs, k=self.top_k, dim=-1)
        sparse = torch.zeros_like(probs)
        sparse.scatter_(dim=-1, index=top_idx, src=top_vals)
        return sparse / (sparse.sum(dim=-1, keepdim=True) + 1e-8)

    @staticmethod
    def get_alpha_stats(alpha: torch.Tensor) -> dict:
        if alpha.dim() < 2:
            raise ValueError(f"Expected alpha to have at least 2 dims, got {alpha.shape}")

        alpha = alpha.reshape(-1, alpha.shape[-1])
        entropy = -(alpha * (alpha + 1e-8).log()).sum(dim=-1)
        avg_alpha = alpha.mean(dim=0)

        sorted_alpha, _ = avg_alpha.sort()
        n = alpha.shape[-1]
        gini_num = (2 * torch.arange(1, n + 1, device=alpha.device).float() - n - 1) * sorted_alpha
        gini = gini_num.sum() / (n * sorted_alpha.sum() + 1e-8)

        return {
            "entropy_mean": entropy.mean().item(),
            "entropy_std": entropy.std().item(),
            "avg_utilization": avg_alpha.detach().cpu().tolist(),
            "gini": gini.item(),
        }


class MLPGatedLoRAModel(GSLoRAModel):
    """
    Same conditioned LoRA mixture setup as Geofield-LoRA, but with MLP+softmax routers.
    """

    def __init__(
        self,
        cfg: DictConfig,
        backbone: nn.Module,
        tokenizer,
    ):
        nn.Module.__init__(self)
        self.cfg = cfg
        self.backbone = backbone
        self.tokenizer = tokenizer
        # Keep alpha casting behavior aligned with GSLoRAModel because
        # the shared training path calls the same _set_current_alpha helper.
        self.alpha_lora_fp32 = bool(cfg.training.get("alpha_lora_fp32", False))

        hidden_size = backbone.config.hidden_size
        condition_input_dim = get_condition_input_dim(cfg, hidden_size)
        n_primitives = cfg.gslora.n_primitives
        rank = cfg.gslora.rank
        view_dim = cfg.gslora.view_dim
        lora_alpha = cfg.gslora.lora_alpha
        target_modules = list(cfg.model.target_modules)
        router_hidden = cfg.get("mlp_gated", {}).get(
            "router_hidden",
            MLPPrimitiveRouter.default_hidden_dim(n_primitives, view_dim),
        )
        router_top_k = cfg.get("mlp_gated", {}).get("top_k", None)

        for param in self.backbone.parameters():
            param.requires_grad = False

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
        # MLP/MixLoRA baselines do not use the template-delta branch.
        # Keep these flags explicit so shared training/eval utilities can
        # safely introspect the model without assuming GS-only attributes.
        self.template_delta_enabled = False
        self.template_runtime_enabled = False
        self.template_gamma = None

        self._gs_lora_layers: Dict[str, GSLoRALinear] = {}
        self._inject_gs_lora(target_modules, n_primitives, rank, lora_alpha)
        self._router_keys = {
            name: f"layer_{idx}" for idx, name in enumerate(self._gs_lora_layers)
        }
        self.router_banks = nn.ModuleDict({
            key: MLPPrimitiveRouter(
                n_primitives=n_primitives,
                view_dim=view_dim,
                hidden_dim=router_hidden,
                top_k=router_top_k,
            )
            for key in self._router_keys.values()
        })
        self.gs_lora_list = nn.ModuleList(list(self._gs_lora_layers.values()))

        print(
            f"[MLP-Gated] condition_source={self.condition_source}, "
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
        )
        print(
            f"[MLP-Gated] Router hidden={router_hidden} "
            f"(default matched capacity target={MLPPrimitiveRouter.default_hidden_dim(n_primitives, view_dim)})"
        )
        if router_top_k is not None and int(router_top_k) > 0:
            print(f"[MLP-Gated] Sparse top-k routing enabled (top_k={int(router_top_k)})")
        self.print_trainable_parameters()

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
            "[MLP-Gated] loaded pretrained condition encoder from "
            f"{pretrained_path} "
            f"(loaded={len(load_info['loaded'])}, skipped={len(load_info['skipped'])}, "
            f"missing={len(load_info['missing'])}, unexpected={len(load_info['unexpected'])})"
        )

    def _get_router_bank(self, layer_name: str) -> MLPPrimitiveRouter:
        return self.router_banks[self._router_keys[layer_name]]

    # Keep the inherited helper name used by shared utility code.
    def _get_primitive_bank(self, layer_name: str) -> MLPPrimitiveRouter:
        return self._get_router_bank(layer_name)

    def compute_alpha(
        self,
        condition_hidden: torch.Tensor,
        template_condition_hidden: Optional[torch.Tensor] = None,
        task_ids: Optional[List[str]] = None,
    ) -> tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        del template_condition_hidden
        z = self.condition_encoder(condition_hidden, task_ids=task_ids)
        alpha_map: Dict[str, torch.Tensor] = {}
        alpha_stack = []

        for name, layer in self._gs_lora_layers.items():
            router = self._get_router_bank(name)
            z_layer = z.to(device=router.fc1.weight.device, dtype=router.fc1.weight.dtype)
            alpha_layer = router(z_layer)
            alpha_map[name] = alpha_layer
            alpha_stack.append(alpha_layer.to(device=z.device, dtype=z.dtype))
        layer_alphas = torch.stack(alpha_stack, dim=1)
        alpha_mean = layer_alphas.mean(dim=1)
        return alpha_map, layer_alphas, alpha_mean, z

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
        Template-free forward used by both the dense MLP-gated and sparse
        MixLoRA-style baselines.
        """
        alpha_map, layer_alphas, alpha, z = self.compute_alpha(
            condition_hidden,
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
            "z_task": z,
        }
        if return_logits:
            result["logits"] = outputs.logits

        if condition_hidden_2 is not None:
            _, layer_alphas_2, alpha_2, z_2 = self.compute_alpha(
                condition_hidden_2,
                task_ids=task_ids,
            )
            result["alpha_2"] = alpha_2
            result["layer_alphas_2"] = layer_alphas_2
            result["z_2"] = z_2

        return result

    def get_primitive_stats(self, alpha: torch.Tensor) -> dict:
        return MLPPrimitiveRouter.get_alpha_stats(alpha)

    def save_adapter(self, path: str):
        os.makedirs(path, exist_ok=True)
        state = {
            "condition_encoder": self.condition_encoder.state_dict(),
            "router_banks": self.router_banks.state_dict(),
            "gs_lora_layers": {
                k: v.state_dict() for k, v in self._gs_lora_layers.items()
            },
        }
        torch.save(state, os.path.join(path, "adapter.pt"))
        from omegaconf import OmegaConf
        OmegaConf.save(self.cfg, os.path.join(path, "config.yaml"))
        print(f"[MLP-Gated] Adapter saved to {path}")

    def load_adapter(self, path: str):
        state = torch.load(os.path.join(path, "adapter.pt"), map_location="cpu", weights_only=True)
        self.condition_encoder.load_state_dict(state["condition_encoder"])
        if "router_banks" in state:
            self.router_banks.load_state_dict(state["router_banks"])
        for name, layer_state in state["gs_lora_layers"].items():
            if name in self._gs_lora_layers:
                self._gs_lora_layers[name].load_state_dict(layer_state)
        print(f"[MLP-Gated] Adapter loaded from {path}")

    @classmethod
    def from_pretrained(
        cls,
        cfg: DictConfig,
        device_map: Any = None,
        device: Any = None,
    ) -> "MLPGatedLoRAModel":
        model_name = cfg.model.backbone
        local_path = os.path.join(cfg.data.model_dir, model_name.split("/")[-1])
        model_path = local_path if os.path.isdir(local_path) else model_name

        print(f"[MLP-Gated] Loading backbone: {model_path}")

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
            print(f"[MLP-Gated] Loading via direct single-device path on {direct_load_device}")
        elif hf_device_map is not None:
            print(f"[MLP-Gated] Loading via HF device_map={hf_device_map}")

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

        backbone_param = next(backbone.parameters())
        backbone_device = backbone_param.device
        backbone_dtype = backbone_param.dtype
        model.condition_encoder = model.condition_encoder.to(device=backbone_device, dtype=backbone_dtype)

        for name, layer in model._gs_lora_layers.items():
            router = model._get_router_bank(name)
            router.to(device=layer.U.device, dtype=layer.U.dtype)

        return model
