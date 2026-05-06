"""
Gaussian Primitive Bank for Geofield-LoRA.

Each primitive k is defined by:
  - μ_k  ∈ R^view_dim   : center in view space
  - d_k  ∈ R^view_dim   : log-diagonal of a lower-triangular factor L_k
  - s_k  ∈ R            : log-strength (learnable scalar)

We parameterize the precision matrix as:
  P_k = L_k L_k^T

where the diagonal of L_k is constrained positive via exp(0.5 * d_k).
The "diag" setting is treated as a special case of the same framework:
all strict-lower entries of L_k are fixed to zero.

Default visibility of primitive k given view embedding z:
  a_k(z) = softplus(s_k) · exp(-½ (z - μ_k)ᵀ P_k (z - μ_k))

Alpha (order-free soft mixture weights):
  α_k(z) = a_k(z) / (Σ_j a_j(z) + ε)

This is NOT physically-ordered depth compositing by default; it is a smooth,
order-free Gaussian mixture over the view space. An opt-in ray-composited
readout is also available. The legacy ray geometry treats z as a direction
from a shared layer origin; the inward ray geometry treats z as a task camera
center and looks back toward the coordinate origin.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GaussianPrimitiveBank(nn.Module):
    """
    Bank of K anisotropic Gaussian primitives in view space.

    Args:
        n_primitives: K – number of primitives
        view_dim:     dimensionality of the view space
    """

    def __init__(
        self,
        n_primitives: int,
        view_dim: int,
        covariance_type: str = "diag",
        center_init_mode: str = "random_normal",
        center_init_radius: float = 0.6,
        center_init_radius_jitter: float = 0.0,
        precision_init_scale: float = 1.0,
        precision_init_log_jitter: float = 0.0,
        offdiag_init_std: float = 0.0,
        readout_mode: str = "point_softmax",
        ray_geometry: str = "shared_origin",
        ray_softplus_tau: float = 0.1,
        ray_opacity_init: float = 0.5,
    ):
        super().__init__()
        self.n_primitives = n_primitives
        self.view_dim = view_dim
        self.readout_mode = str(readout_mode).lower()
        if self.readout_mode in {"point", "softmax", "point_softmax"}:
            self.readout_mode = "point_softmax"
        elif self.readout_mode in {"ray", "ray_composite", "ray_project"}:
            self.readout_mode = "ray_composite"
        elif self.readout_mode in {
            "ray_normalized",
            "ray_normalize",
            "ray_norm",
            "ray_weighted_sum",
        }:
            self.readout_mode = "ray_normalized"
        else:
            raise ValueError(
                "Unsupported readout_mode="
                f"{readout_mode!r}; expected 'point_softmax', 'ray_composite', "
                "or 'ray_normalized'"
            )
        self.ray_geometry = str(ray_geometry).lower()
        if self.ray_geometry in {"shared", "shared_origin", "legacy", "outward"}:
            self.ray_geometry = "shared_origin"
        elif self.ray_geometry in {"inward", "inward_origin", "task_origin", "task_camera"}:
            self.ray_geometry = "inward_origin"
        else:
            raise ValueError(
                "Unsupported ray_geometry="
                f"{ray_geometry!r}; expected 'shared_origin' or 'inward_origin'"
            )
        self.covariance_type = str(covariance_type).lower()
        if self.covariance_type not in {"diag", "full"}:
            raise ValueError(
                f"Unsupported covariance_type={covariance_type!r}; expected 'diag' or 'full'"
            )
        self.use_offdiag = self.covariance_type == "full"
        self.center_init_mode = str(center_init_mode).lower()
        if self.center_init_mode not in {"random_normal", "radial_shell"}:
            raise ValueError(
                "Unsupported center_init_mode="
                f"{center_init_mode!r}; expected 'random_normal' or 'radial_shell'"
            )
        self.center_init_radius = float(center_init_radius)
        self.center_init_radius_jitter = float(center_init_radius_jitter)
        self.precision_init_scale = float(precision_init_scale)
        self.precision_init_log_jitter = float(precision_init_log_jitter)
        self.offdiag_init_std = float(offdiag_init_std)
        self.ray_softplus_tau = float(ray_softplus_tau)
        self.ray_opacity_init = float(ray_opacity_init)
        if self.center_init_radius < 0.0:
            raise ValueError("center_init_radius must be non-negative")
        if self.center_init_radius_jitter < 0.0:
            raise ValueError("center_init_radius_jitter must be non-negative")
        if self.precision_init_scale <= 0.0:
            raise ValueError("precision_init_scale must be positive")
        if self.precision_init_log_jitter < 0.0:
            raise ValueError("precision_init_log_jitter must be non-negative")
        if self.offdiag_init_std < 0.0:
            raise ValueError("offdiag_init_std must be non-negative")
        if self.ray_softplus_tau <= 0.0:
            raise ValueError("ray_softplus_tau must be positive")
        if self.ray_opacity_init <= 0.0:
            raise ValueError("ray_opacity_init must be positive")

        # Primitive centers μ_k, initialized to cover the space
        self.mu = nn.Parameter(torch.empty(n_primitives, view_dim))

        # Log-diagonal of the lower-triangular precision factor L_k.
        # diag(L_k) = exp(0.5 * log_diag_k), so P_k = L_k L_k^T is SPD.
        # Initialize to 0 → diag(L_k) = 1 (unit precision on each axis)
        self.log_diag = nn.Parameter(torch.zeros(n_primitives, view_dim))

        # Log-strength per primitive. softplus(log_s) ensures positivity.
        self.log_s = nn.Parameter(torch.zeros(n_primitives))
        tril_size = view_dim * (view_dim - 1) // 2
        tril_i, tril_j = torch.tril_indices(view_dim, view_dim, offset=-1)
        self.register_buffer("_tril_i", tril_i, persistent=False)
        self.register_buffer("_tril_j", tril_j, persistent=False)
        if self.use_offdiag:
            self.lower_offdiag = nn.Parameter(torch.zeros(n_primitives, tril_size))
        else:
            self.register_parameter("lower_offdiag", None)
        if self.readout_mode == "ray_composite" and self.ray_geometry == "shared_origin":
            self.ray_origin = nn.Parameter(torch.zeros(view_dim))
        else:
            self.register_parameter("ray_origin", None)

        self._init_weights()

    def _init_weights(self):
        if self.center_init_mode == "random_normal":
            # Current baseline: spread centers roughly uniformly by iid Gaussian init.
            nn.init.normal_(self.mu, mean=0.0, std=1.0 / math.sqrt(self.view_dim))
        else:
            # Structured alternative: place centers on distinct directions with a
            # shared mid-radius so primitives begin inside the sphere rather than
            # collapsed at the origin.
            with torch.no_grad():
                directions = torch.randn_like(self.mu)
                directions = F.normalize(directions, p=2, dim=-1, eps=1e-8)
                radii = torch.full(
                    (self.n_primitives, 1),
                    self.center_init_radius,
                    dtype=self.mu.dtype,
                    device=self.mu.device,
                )
                if self.center_init_radius_jitter > 0.0:
                    jitter = torch.empty_like(radii).uniform_(
                        -self.center_init_radius_jitter,
                        self.center_init_radius_jitter,
                    )
                    radii = radii + jitter
                radii.clamp_(min=1e-4, max=0.999)
                self.mu.copy_(directions * radii)
        init_log_diag = 2.0 * math.log(self.precision_init_scale)
        nn.init.constant_(self.log_diag, init_log_diag)
        if self.precision_init_log_jitter > 0.0:
            with torch.no_grad():
                self.log_diag.add_(
                    torch.empty_like(self.log_diag).normal_(
                        mean=0.0,
                        std=self.precision_init_log_jitter,
                    )
                )
        if self.lower_offdiag is not None and self.offdiag_init_std > 0.0:
            nn.init.normal_(self.lower_offdiag, mean=0.0, std=self.offdiag_init_std)
        if self.readout_mode == "ray_composite":
            # log_s is passed through softplus; initialize opacity strength to
            # a small positive value instead of the point-softmax default.
            init_strength = torch.tensor(float(self.ray_opacity_init)).expm1().log().item()
            nn.init.constant_(self.log_s, init_strength)
        else:
            nn.init.zeros_(self.log_s)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute contribution weights from z.

        Args:
            z: (batch, view_dim) view embeddings

        Returns:
            weights: (batch, n_primitives). In point_softmax and
                ray_normalized mode these sum to 1. In ray_composite mode
                they are alpha-composited opacity weights and sum to <= 1.
        """
        if self.readout_mode == "ray_composite":
            return self._forward_ray_composite(z)
        if self.readout_mode == "ray_normalized":
            return self._forward_ray_normalized(z)
        return self._forward_point_softmax(z)

    def _precision_cholesky(self, ref: torch.Tensor) -> torch.Tensor:
        chol = ref.new_zeros(self.n_primitives, self.view_dim, self.view_dim)
        diag_values = torch.exp(0.5 * self.log_diag)
        diag_idx = torch.arange(self.view_dim, device=ref.device)
        chol[:, diag_idx, diag_idx] = diag_values
        if self.lower_offdiag is not None:
            chol[:, self._tril_i, self._tril_j] = self.lower_offdiag
        return chol

    def _forward_point_softmax(self, z: torch.Tensor) -> torch.Tensor:
        # Difference from centers: (batch, K, view_dim)
        diff = z.unsqueeze(1) - self.mu.unsqueeze(0)

        # Unified precision path: "diag" is simply the zero-offdiag case.
        chol = self._precision_cholesky(diff)

        transformed = torch.einsum("bkd,kde->bke", diff, chol)
        mahal = (transformed ** 2).sum(dim=-1)

        # Unnormalized visibility: softplus(s_k) · exp(-½ mahal)
        strength = F.softplus(self.log_s)              # (K,)
        log_a = -0.5 * mahal + torch.log(strength + 1e-8).unsqueeze(0)
        # Use log-sum-exp trick for numerical stability
        log_a_max = log_a.detach().max(dim=-1, keepdim=True).values
        a = torch.exp(log_a - log_a_max)               # (batch, K)

        # Normalize to get mixture weights
        alpha = a / (a.sum(dim=-1, keepdim=True) + 1e-8)
        return alpha

    def _forward_ray_composite(self, z: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        u, gaussian_response, strength = self._ray_visibility(z, eps=eps)
        opacity = 1.0 - torch.exp(-strength * gaussian_response)
        opacity = opacity.clamp(min=0.0, max=1.0 - 1e-6)

        order = torch.argsort(u, dim=-1)
        opacity_sorted = torch.gather(opacity, dim=-1, index=order)
        one_minus = (1.0 - opacity_sorted).clamp_min(eps)
        transmittance = torch.cumprod(
            torch.cat(
                [
                    torch.ones_like(one_minus[..., :1]),
                    one_minus[..., :-1],
                ],
                dim=-1,
            ),
            dim=-1,
        )
        weights_sorted = transmittance * opacity_sorted

        weights = torch.zeros_like(weights_sorted)
        weights.scatter_(dim=-1, index=order, src=weights_sorted)
        return weights

    def _forward_ray_normalized(self, z: torch.Tensor) -> torch.Tensor:
        # Keep the inward/shared ray geometry and closest-point Gaussian
        # visibility, but replace depth compositing with a simple normalized
        # weighted sum over all primitives.
        _, gaussian_response, strength = self._ray_visibility(z, eps=1e-8)
        visibility = strength * gaussian_response
        return visibility / visibility.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def _ray_visibility(
        self,
        z: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        origin, direction = self._ray_origin_and_direction(z, eps=eps)
        chol = self._precision_cholesky(z)

        # Work in the Gaussian precision metric P_k=L_k L_k^T. Since
        # ||x L_k||^2 = x^T P_k x, the closest depth can be computed using
        # transformed ray directions and transformed center offsets.
        center_offset = self.mu.unsqueeze(0) - origin.unsqueeze(1)         # (B, K, D)
        d_chol = torch.einsum("bd,kde->bke", direction, chol)              # (B, K, D)
        offset_chol = torch.einsum("bkd,kde->bke", center_offset, chol)    # (B, K, D)

        numerator = (d_chol * offset_chol).sum(dim=-1)                    # (B, K)
        denominator = (d_chol * d_chol).sum(dim=-1).clamp_min(eps)
        u_raw = numerator / denominator
        u = self.ray_softplus_tau * F.softplus(u_raw / self.ray_softplus_tau)

        closest = origin.unsqueeze(1) + u.unsqueeze(-1) * direction.unsqueeze(1)
        diff = closest - self.mu.unsqueeze(0)
        transformed = torch.einsum("bkd,kde->bke", diff, chol)
        dist2 = (transformed * transformed).sum(dim=-1)
        gaussian_response = torch.exp(-0.5 * dist2)
        strength = F.softplus(self.log_s).unsqueeze(0)
        return u, gaussian_response, strength

    def _ray_origin_and_direction(
        self,
        z: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.ray_geometry == "shared_origin":
            if self.ray_origin is None:
                raise RuntimeError("ray_origin is required for shared_origin ray geometry")
            origin = self.ray_origin.unsqueeze(0).expand_as(z)
            direction = F.normalize(z, p=2, dim=-1, eps=eps)
            return origin, direction

        # Inward geometry: z is the task camera center in view space, and the
        # ray looks back toward the coordinate origin.
        origin = z
        direction = F.normalize(-z, p=2, dim=-1, eps=eps)
        return origin, direction

    def normalize_for_regularizer(self, weights: torch.Tensor) -> torch.Tensor:
        """Return a distribution over primitives for entropy/balance metrics."""
        if self.readout_mode in {"point_softmax", "ray_normalized"}:
            return weights
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def get_primitive_stats(self, alpha: torch.Tensor) -> dict:
        """
        Compute diagnostic statistics over a batch of alpha distributions.

        Args:
            alpha: (batch, K) or (batch, n_layers, K)
        Returns:
            dict with entropy, utilization, gini coefficient stats
        """
        if alpha.dim() < 2:
            raise ValueError(f"Expected alpha to have at least 2 dims, got {alpha.shape}")

        alpha = alpha.reshape(-1, alpha.shape[-1])

        # Per-sample entropy
        entropy = -(alpha * (alpha + 1e-8).log()).sum(dim=-1)   # (batch,)

        # Batch-average utilization per primitive
        avg_alpha = alpha.mean(dim=0)                            # (K,)

        # Gini coefficient of average utilization
        sorted_alpha, _ = avg_alpha.sort()
        n = self.n_primitives
        gini_num = (2 * torch.arange(1, n + 1, device=alpha.device).float() - n - 1) * sorted_alpha
        gini = gini_num.sum() / (n * sorted_alpha.sum() + 1e-8)

        return {
            "entropy_mean": entropy.mean().item(),
            "entropy_std": entropy.std().item(),
            "avg_utilization": avg_alpha.detach().cpu().tolist(),
            "gini": gini.item(),
        }
