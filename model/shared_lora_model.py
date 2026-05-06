import os
from typing import Any, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class SharedLoRAModel(nn.Module):
    def __init__(
        self,
        cfg: DictConfig,
        backbone: nn.Module,
        tokenizer,
        attach_trainable_lora: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.tokenizer = tokenizer

        k = cfg.gslora.n_primitives
        r = cfg.gslora.rank
        lora_rank = int(k * r)
        lora_alpha = int(cfg.gslora.lora_alpha * (lora_rank / r))
        self.effective_lora_rank = lora_rank

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            target_modules=list(cfg.model.target_modules),
            bias="none",
        )
        self.model = get_peft_model(self.backbone, lora_config) if attach_trainable_lora else self.backbone
        self.model_type = "shared"
        print(
            "[Shared-LoRA] "
            f"n_primitives={k}, rank_per_primitive={r}, "
            f"effective_lora_rank={lora_rank}, lora_alpha={lora_alpha}, "
            f"attach_trainable_lora={'on' if attach_trainable_lora else 'off'}"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        condition_hidden: Optional[torch.Tensor] = None,
        condition_hidden_2: Optional[torch.Tensor] = None,
        return_logits: bool = False,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        result = {"loss": outputs.loss}
        if return_logits:
            result["logits"] = outputs.logits
        return result

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def print_trainable_parameters(self):
        if hasattr(self.model, "print_trainable_parameters"):
            self.model.print_trainable_parameters()
            return
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        pct = 100 * trainable / max(total, 1)
        print(f"[Shared-LoRA] Trainable: {trainable:,} / {total:,} ({pct:.3f}%)")

    def get_primitive_stats(self, alpha: Optional[torch.Tensor] = None) -> dict:
        return {}

    def save_adapter(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"[Shared-LoRA] Adapter saved to {path}")

    def load_adapter(self, path: str):
        self.model = PeftModel.from_pretrained(self.backbone, path, is_trainable=True)
        print(f"[Shared-LoRA] Adapter loaded from {path}")

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
        attach_trainable_lora: bool = True,
    ) -> "SharedLoRAModel":
        model_name = cfg.model.backbone
        local_path = os.path.join(cfg.data.model_dir, model_name.split("/")[-1])
        model_path = local_path if os.path.isdir(local_path) else model_name

        print(f"[Shared-LoRA] Loading backbone: {model_path}")

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
            print(f"[Shared-LoRA] Loading via direct single-device path on {direct_load_device}")
        elif hf_device_map is not None:
            print(f"[Shared-LoRA] Loading via HF device_map={hf_device_map}")

        backbone = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        if direct_load_device is not None:
            backbone = backbone.to(device=direct_load_device, dtype=dtype)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = cls(
            cfg=cfg,
            backbone=backbone,
            tokenizer=tokenizer,
            attach_trainable_lora=attach_trainable_lora,
        )
        return model
