"""
Geofield-LoRA Linear Layer.

Replaces a frozen Linear(in_features, out_features) with a Geofield-LoRA version.

The key design: GSLoRALinear.forward(x) has the SAME signature as nn.Linear.forward(x),
so the backbone's attention modules call it transparently. Alpha is stored as
self._current_alpha and set externally before each backbone forward pass.

ΔW(α) = Σ_k α_k · U_k @ V_kᵀ   (weighted sum of K low-rank atoms)

Efficient batched einsum (no explicit K-loop):
  1. xU   = einsum('bsi,kir -> bskr', x, U)      # (B, S, K, rank)
  2. xU_w = xU * α[:, None, :, None]              # broadcast α
  3. Δout  = einsum('bskr,kor -> bso', xU_w, V)  # (B, S, out)

Note: 'o' is out_features in the einsum.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GSLoRALinear(nn.Module):
    """
    A frozen Linear layer augmented with Geofield-LoRA adaptation.

    The base weight W is frozen. Only U and V (per-primitive) are trainable.
    Alpha α must be set via set_alpha() before each forward pass.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_primitives: int,
        rank: int,
        lora_alpha: float = 16.0,
        bias: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.n_primitives = n_primitives
        self.rank = rank
        self.scaling = lora_alpha / rank

        # Frozen base weight (requires_grad=False)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features), requires_grad=False
            )
        else:
            self.register_parameter("bias", None)

        # Trainable: U (K, in, rank), V (K, out, rank)
        self.U = nn.Parameter(torch.empty(n_primitives, in_features, rank))
        self.V = nn.Parameter(torch.zeros(n_primitives, out_features, rank))

        # Current alpha – set before each forward, NOT a parameter
        self._current_alpha: Optional[torch.Tensor] = None

        self._init_lora_weights()

    def _init_lora_weights(self):
        # U ~ Kaiming uniform, V = 0 so initial ΔW = 0
        nn.init.kaiming_uniform_(
            self.U.view(self.n_primitives * self.in_features, self.rank),
            a=math.sqrt(5),
        )
        nn.init.zeros_(self.V)

    def set_alpha(self, alpha: torch.Tensor):
        """Set the current batch's mixture weights. Call before forward()."""
        self._current_alpha = alpha

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        n_primitives: int,
        rank: int,
        lora_alpha: float = 16.0,
    ) -> "GSLoRALinear":
        has_bias = linear.bias is not None
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            n_primitives=n_primitives,
            rank=rank,
            lora_alpha=lora_alpha,
            bias=has_bias,
        )
        with torch.no_grad():
            layer.weight.copy_(linear.weight)
            if has_bias:
                layer.bias.copy_(linear.bias)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard Linear-compatible forward. Reads alpha from self._current_alpha.

        Args:
            x: (..., in_features) – works for any leading dims, including (B, S, in)

        Returns:
            output: (..., out_features)
        """
        base_out = F.linear(x, self.weight, self.bias)

        if self._current_alpha is None:
            return base_out

        alpha = self._current_alpha   # (B, K)

        # Handle input shape: could be (B, S, in) or (B, in) or even (B, heads, S, in)
        orig_shape = x.shape
        if x.dim() == 2:
            # (B, in) → treat as (B, 1, in)
            x3 = x.unsqueeze(1)
        elif x.dim() == 3:
            x3 = x                    # (B, S, in)
        elif x.dim() == 4:
            # (B, heads, S, in) – unusual but handle it by merging B*heads
            B, H, S, D = x.shape
            x3 = x.reshape(B * H, S, D)
            # Repeat alpha for each head
            alpha = alpha.repeat_interleave(H, dim=0)
        else:
            # Fallback: no LoRA delta
            return base_out

        B, S, _ = x3.shape

        # Batched Geofield-LoRA delta
        # xU: (B, S, K, rank)
        xU = torch.einsum("bsi,kir->bskr", x3, self.U)
        # Weight by alpha
        if alpha.dtype != xU.dtype:
            xU = xU.to(dtype=alpha.dtype)
        xU_w = xU * alpha[:, None, :, None]
        # Project back through V
        V = self.V if self.V.dtype == xU_w.dtype else self.V.to(dtype=xU_w.dtype)
        delta = torch.einsum("bskr,kor->bso", xU_w, V)   # (B, S, out)

        if x.dim() == 2:
            delta = delta.squeeze(1)   # (B, out)
        elif x.dim() == 4:
            delta = delta.reshape(B, H, S, self.out_features)

        # Keep the surrounding backbone dtype stable even when a routing probe
        # keeps alpha in fp32 for more accurate geometry gradients.
        if delta.dtype != base_out.dtype:
            delta = delta.to(dtype=base_out.dtype)

        return base_out + self.scaling * delta

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"K={self.n_primitives}, rank={self.rank}, scaling={self.scaling:.3f}"
        )
