"""
Geofield-LoRA structural regularizers.

L_total = L_task + λ_cons * L_cons + λ_sparse * L_sparse + λ_balance * L_balance

L_cons:    consistency of α distributions across different prompt templates
           of the same underlying example → discourages template memorization
L_sparse:  low-entropy per-example mixtures → primitives remain locally active
L_balance: batch-level α distribution stays broad → prevents primitive collapse
"""

import torch
import torch.nn.functional as F
from typing import Optional


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Jensen-Shannon divergence (symmetric, bounded [0, log 2]).

    Args:
        p, q: (B, K) or (B, L, K) — distributions along the last dim (sum to 1 over K)
    Returns:
        jsd: (B,) or (B, L) — one JSD per leading "row"
    """
    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p + eps) - torch.log(m + eps))).sum(dim=-1)
    kl_qm = (q * (torch.log(q + eps) - torch.log(m + eps))).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Symmetric KL: KL(p||q) + KL(q||p).

    Args:
        p, q: (B, K) probability distributions
    Returns:
        sym_kl: (B,)
    """
    kl_pq = (p * (torch.log(p + eps) - torch.log(q + eps))).sum(dim=-1)
    kl_qp = (q * (torch.log(q + eps) - torch.log(p + eps))).sum(dim=-1)
    return 0.5 * (kl_pq + kl_qp)


def consistency_loss(
    alpha: torch.Tensor,
    alpha_2: torch.Tensor,
    metric: str = "js",
    pair_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    L_cons: encourages the same example under two different prompt templates
    to produce similar α distributions.

    Args:
        alpha:   (B, K) or (B, L, K) mixture weights from template 1
        alpha_2: (B, K) or (B, L, K) mixture weights from template 2
        metric:  "js" | "kl" | "cosine"
    Returns:
        scalar loss
    """
    alpha = alpha.float()
    alpha_2 = alpha_2.float()

    if metric == "js":
        row_loss = js_divergence(alpha, alpha_2)
    elif metric == "kl":
        row_loss = kl_divergence(alpha, alpha_2)
    elif metric == "cosine":
        row_loss = 1 - F.cosine_similarity(alpha, alpha_2, dim=-1)
    else:
        raise ValueError(f"Unknown consistency metric: {metric}")

    if pair_weight is None:
        return row_loss.mean()

    weight = pair_weight.to(device=row_loss.device, dtype=row_loss.dtype)
    while weight.dim() < row_loss.dim():
        weight = weight.unsqueeze(-1)

    weighted = row_loss * weight
    denom = weight.sum().clamp(min=1e-8)
    return weighted.sum() / denom


def sparsity_loss(alpha: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    L_sparse: minimize per-example mixture entropy to encourage
    primitive local support (fewer primitives active per example).

    H(α) = -Σ_k α_k log(α_k)

    Lower entropy → more concentrated → more interpretable primitive usage.

    If alpha is (B, L, K), entropy is (B, L); this loss averages over batch and layers.

    Returns:
        scalar loss (mean entropy)
    """
    alpha = alpha.float()
    entropy = -(alpha * torch.log(alpha + eps)).sum(dim=-1)   # (B,) or (B, L)
    return entropy.mean()


def balance_loss(alpha: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    L_balance: prevent all examples from collapsing onto the same primitives.

    Encourage batch-average ᾱ to be close to uniform distribution.
    KL(uniform || ᾱ) = Σ_k (1/K) * log((1/K) / ᾱ_k)
                     = -log(K) - (1/K) Σ_k log(ᾱ_k)

    Minimizing this pushes ᾱ toward uniform.

    Returns:
        scalar loss
    """
    alpha = alpha.float()
    avg_alpha = alpha.mean(dim=0)
    K = alpha.shape[-1]

    if avg_alpha.dim() == 1:
        return -(1.0 / K) * torch.log(avg_alpha + eps).sum()

    # Per-layer balance when alpha is (B, L, K), then average over layers.
    return (-(1.0 / K) * torch.log(avg_alpha + eps).sum(dim=-1)).mean()


def compute_structural_losses(
    alpha: torch.Tensor,
    alpha_2: Optional[torch.Tensor] = None,
    lambda_cons: float = 0.1,
    lambda_sparse: float = 0.01,
    lambda_balance: float = 0.01,
    consistency_metric: str = "js",
    consistency_pair_weight: Optional[torch.Tensor] = None,
) -> dict:
    """
    Compute all three structural regularizers and their weighted sum.

    Args:
        alpha:    (B, K) or (B, L, K) mixture weights for template 1
        alpha_2:  (B, K) or (B, L, K) mixture weights for template 2
                  (optional, for L_cons)
        lambda_*: loss weights from config

    Returns:
        dict with individual losses and total structural loss
    """
    losses = {}

    # L_cons: only when two templates are available
    if alpha_2 is not None and lambda_cons > 0:
        l_cons = consistency_loss(
            alpha,
            alpha_2,
            metric=consistency_metric,
            pair_weight=consistency_pair_weight,
        )
        losses["loss_cons"] = l_cons
        losses["loss_struct"] = lambda_cons * l_cons
    else:
        losses["loss_cons"] = torch.tensor(0.0, device=alpha.device, dtype=torch.float32)
        losses["loss_struct"] = torch.tensor(0.0, device=alpha.device, dtype=torch.float32)

    # L_sparse
    if lambda_sparse > 0:
        l_sparse = sparsity_loss(alpha)
        losses["loss_sparse"] = l_sparse
        losses["loss_struct"] = losses["loss_struct"] + lambda_sparse * l_sparse
    else:
        losses["loss_sparse"] = torch.tensor(0.0, device=alpha.device, dtype=torch.float32)

    # L_balance
    if lambda_balance > 0:
        l_balance = balance_loss(alpha)
        losses["loss_balance"] = l_balance
        losses["loss_struct"] = losses["loss_struct"] + lambda_balance * l_balance
    else:
        losses["loss_balance"] = torch.tensor(0.0, device=alpha.device, dtype=torch.float32)

    return losses
