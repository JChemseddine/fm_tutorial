# Copied from https://github.com/JChemseddine/spherical (paper-release-anon)
# Paper: Spherical Flows for Sampling Discrete Distributions (arXiv:2605.05629)

"""Spherical flow path: vMF noise sampling, velocity target, source/target lookup.

vMF noise process: h_t ~ vMF(w_k, kappa). kappa sampled via CDCD warp or linear schedule.

Velocity target uses psi_tilde weights from a precomputed lookup table
(replaces alpha/sin(alpha) from SLERP).

Legacy SLERP and Cholesky source methods retained for comparison.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .vmf import PsiTable


class SphericalPath:
    """Spherical flow path operations."""

    @staticmethod
    def sample_source(W_E: Tensor, sigma_max: float, batch_size: int,
                      seq_len: int, device: torch.device) -> Tensor:
        """Sample from learned source distribution via Cholesky.

        Args:
            W_E: (embed_dim, V) unit-norm embedding matrix
            sigma_max: sigma of initial Gaussian in R^V
            batch_size: B
            seq_len: L
            device: torch device

        Returns:
            z: (B, L, embed_dim) source samples on the sphere
        """
        embed_dim, V = W_E.shape

        # Gram matrix: Sigma = W_E @ W_E^T + eps*I for numerical stability
        Sigma = W_E @ W_E.T  # (embed_dim, embed_dim)
        Sigma = Sigma + 1e-6 * torch.eye(embed_dim, device=device)
        L_chol = torch.linalg.cholesky(Sigma)  # (embed_dim, embed_dim)

        # Sample: z_std ~ N(0, I), z = sigma_max * z_std @ L_chol^T
        z_std = torch.randn(batch_size, seq_len, embed_dim, device=device)
        z = sigma_max * (z_std @ L_chol.T)  # (B, L, embed_dim)

        # Project onto sphere
        z = F.normalize(z, dim=-1)
        return z

    @staticmethod
    def interpolate(z: Tensor, w_k: Tensor, t: Tensor) -> Tensor:
        """SLERP interpolation on the sphere (geodesic path).

        h_t = sin((1-t)*alpha)/sin(alpha) * z + sin(t*alpha)/sin(alpha) * w_k

        SLERP coefficients are detached from the gradient graph to avoid
        arccos/sqrt gradient singularities at cos(alpha) ≈ ±1 (common for low V).
        Gradients flow through z and w_k via the linear combination + normalize.

        Falls back to NLERP when sin(alpha) < eps (nearly parallel vectors).

        Args:
            z: (B, L, embed_dim) source points on sphere
            w_k: (B, L, embed_dim) target embeddings on sphere
            t: (B, 1, 1) or broadcastable time values in [0, 1]

        Returns:
            h_t: (B, L, embed_dim) interpolated points on sphere
        """
        # Angle via atan2 + clamped sqrt (gradient-stable, unlike arccos)
        cos_alpha = (z * w_k).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        sin_alpha = torch.sqrt((1 - cos_alpha.square()).clamp(min=1e-8))
        alpha = torch.atan2(sin_alpha, cos_alpha)

        # SLERP coefficients (both branches finite for torch.where gradient safety)
        sin_alpha_safe = sin_alpha.clamp(min=1e-6)
        c0 = torch.where(sin_alpha > 1e-4, torch.sin((1 - t) * alpha) / sin_alpha_safe, 1 - t)
        c1 = torch.where(sin_alpha > 1e-4, torch.sin(t * alpha) / sin_alpha_safe, t)

        # SLERP guarantees ||h_t|| = 1 — no normalize needed
        h_t = c0 * z + c1 * w_k
        return h_t

    @staticmethod
    def slerp_velocity_target(h_t: Tensor, W_E: Tensor, probs: Tensor) -> Tensor:
        """Compute the expected SLERP velocity direction (already on tangent plane).

        D_tilde = sum_k q_k * w_k - (sum_k cos_k * q_k) * h_t
        where q_k = p_k * alpha_k / sin(alpha_k)

        This is E_p[ alpha_k/sin(alpha_k) * proj_tan(w_k, h_t) ].

        Args:
            h_t: (B, L, embed_dim) current point on sphere
            W_E: (embed_dim, V) unit-norm embedding matrix
            probs: (B, L, V) softmax probabilities

        Returns:
            D_tilde: (B, L, embed_dim) velocity target, already tangent to sphere at h_t
        """
        # Cosine similarities h_t @ W_E -> (B, L, V)
        cos_k = torch.matmul(h_t, W_E)

        # Angles via atan2 (gradient-stable)
        sin_alpha_k = torch.sqrt((1 - cos_k.square()).clamp(min=0.0))  # (B, L, V)
        alpha_k = torch.atan2(sin_alpha_k, cos_k)  # (B, L, V)
        # alpha/sin(alpha) -> 1 as alpha -> 0
        weight_k = torch.where(
            sin_alpha_k.abs() > 1e-6,
            alpha_k / sin_alpha_k.clamp(min=1e-8),
            torch.ones_like(alpha_k),
        )

        q_k = probs * weight_k  # (B, L, V)

        # D_tilde = sum_k(q_k * w_k) - sum_k(cos_k * q_k) * h_t
        D_tilde = torch.matmul(q_k, W_E.T)  # (B, L, embed_dim)
        D_tilde = D_tilde - (cos_k * q_k).sum(dim=-1, keepdim=True) * h_t

        return D_tilde

    @staticmethod
    def get_target_embeddings(W_E: Tensor, token_ids: Tensor) -> Tensor:
        """Look up target embeddings from W_E.

        Args:
            W_E: (embed_dim, V) unit-norm embedding matrix
            token_ids: (B, L) target token indices

        Returns:
            w_k: (B, L, embed_dim) target embeddings (already unit-norm)
        """
        # W_E[:, token_ids] -> (embed_dim, B, L) -> permute to (B, L, embed_dim)
        return W_E[:, token_ids].permute(1, 2, 0)

    @staticmethod
    def sample_vmf_noise(
        W_E: Tensor,
        kappa: Union[Tensor, float],
        token_ids: Tensor,
        device: torch.device,
        cdf_table: 'VMFRadialCDF' = None,
    ) -> Tensor:
        """Sample h_t ~ vMF(w_k, kappa) for each position.

        Args:
            W_E: (embed_dim, V) unit-norm embedding matrix
            kappa: (B,) or scalar concentration parameter
            token_ids: (B, L) target token indices
            device: torch device
            cdf_table: VMFRadialCDF instance for CDF-based sampling (None = rejection)

        Returns:
            h_t: (B, L, embed_dim) samples on S^{d-1}
        """
        w_k = SphericalPath.get_target_embeddings(W_E, token_ids)  # (B, L, embed_dim)

        if cdf_table is not None:
            from .vmf import sample_vmf_cdf
            h_t = sample_vmf_cdf(w_k, kappa, cdf_table)
        else:
            from .vmf import sample_vmf
            h_t = sample_vmf(w_k, kappa)
        return h_t

    @staticmethod
    def vmf_velocity_target(
        h_t: Tensor,
        W_E: Tensor,
        probs: Tensor,
        psi_table: 'PsiTable',
        kappa: Union[Tensor, float],
    ) -> Tensor:
        """Compute the vMF velocity target (bounded, no kappa_max).

        v_target = sum_k q_k * w_k - (sum_k cos_k * q_k) * h_t
        where q_k = p_k * psi_tilde(cos_k, kappa)

        This is on the tangent plane at h_t. The actual velocity is
        kappa_max * v_target (applied at inference, not here).

        Args:
            h_t: (B, L, embed_dim) current point on sphere
            W_E: (embed_dim, V) unit-norm embedding matrix
            probs: (B, L, V) softmax probabilities
            psi_table: PsiTable instance for psi_tilde lookup
            kappa: scalar or (B,) concentration at current time

        Returns:
            v_target: (B, L, embed_dim) velocity target, tangent to sphere at h_t
        """
        cos_k = torch.matmul(h_t, W_E)  # (B, L, V)

        # 1D psi lookup when kappa is scalar (sampling); 2D fallback otherwise
        if isinstance(kappa, (int, float)) or (isinstance(kappa, Tensor) and kappa.dim() <= 1):
            psi_k = psi_table.lookup_1d(cos_k, kappa)
        else:
            psi_k = psi_table.lookup(cos_k, kappa)
        del cos_k  # free — not needed for tangent projection (use h·v_sum instead)

        q_k = probs * psi_k
        del psi_k

        v_sum = torch.matmul(q_k, W_E.T)  # (B, L, embed_dim)
        del q_k

        # Tangent projection: radial component = h_t · v_sum (avoids keeping cos_k alive)
        radial = (h_t * v_sum).sum(dim=-1, keepdim=True)
        v_target = v_sum - radial * h_t

        return v_target

    @staticmethod
    def score_on_sphere(
        h_t: Tensor,
        W_E: Tensor,
        probs: Tensor,
        kappa: Union[Tensor, float],
    ) -> Tensor:
        """Score ∇_h log p_t(h) on the tangent plane at h_t.

        Same structure as vmf_velocity_target but with κ replacing ψ_tilde:
            score = κ * Σ_k p(k|h) * proj_tan(w_k, h_t)

        Args:
            h_t: (B, L, embed_dim) current point on sphere
            W_E: (embed_dim, V) unit-norm embedding matrix
            probs: (B, L, V) softmax probabilities
            kappa: scalar or (B,) or (B, L) concentration

        Returns:
            score: (B, L, embed_dim) score vector, tangent to sphere at h_t
        """
        # Broadcast kappa
        if isinstance(kappa, Tensor):
            k = kappa
            while k.dim() < probs.dim():
                k = k.unsqueeze(-1)
        else:
            k = kappa

        q_k = probs * k  # (B, L, V)
        score_sum = torch.matmul(q_k, W_E.T)  # (B, L, embed_dim)
        del q_k

        # Tangent projection via h · score_sum (avoids materializing cos_k)
        radial = (h_t * score_sum).sum(dim=-1, keepdim=True)
        score = score_sum - radial * h_t

        return score

    @staticmethod
    def vmf_sde_drift(
        h_t: Tensor,
        W_E: Tensor,
        probs: Tensor,
        psi_table: 'PsiTable',
        kappa: Union[Tensor, float],
        sigma_sq: float,
    ) -> Tensor:
        """Fused SDE drift = ODE velocity + σ² · score in one matmul.

        drift = Σ_k p_k (ψ̃_k + σ²κ) proj_tan(w_k, h_t)

        Saves one (B,L,V) @ (V,d) matmul vs computing velocity and score separately.
        """
        cos_k = torch.matmul(h_t, W_E)  # (B, L, V)

        # 1D psi lookup when kappa is scalar
        if isinstance(kappa, (int, float)) or (isinstance(kappa, Tensor) and kappa.dim() <= 1):
            psi_k = psi_table.lookup_1d(cos_k, kappa)
        else:
            psi_k = psi_table.lookup(cos_k, kappa)
        del cos_k

        # Combined weight: ψ̃ + σ²κ
        k = kappa
        if isinstance(k, Tensor):
            while k.dim() < psi_k.dim():
                k = k.unsqueeze(-1)
        combined_w = probs * (psi_k + sigma_sq * k)
        del psi_k

        v_sum = torch.matmul(combined_w, W_E.T)  # (B, L, embed_dim)
        del combined_w

        radial = (h_t * v_sum).sum(dim=-1, keepdim=True)
        return v_sum - radial * h_t

    @staticmethod
    def proj_tan(u: Tensor, h: Tensor) -> Tensor:
        """Project u onto tangent plane of sphere at h.

        proj_tan(u, h) = u - (u * h).sum(dim=-1, keepdim=True) * h

        Args:
            u: (..., embed_dim) vector to project
            h: (..., embed_dim) point on sphere (unit-norm)

        Returns:
            u_tan: (..., embed_dim) tangent component
        """
        return u - (u * h).sum(dim=-1, keepdim=True) * h

    @staticmethod
    def exp_map(x: Tensor, v: Tensor) -> Tensor:
        """Riemannian exp map on S^{d-1}: geodesic from x with tangent velocity v.

        exp_x(v) = cos(||v||) * x + sin(||v||) * (v / ||v||)

        Args:
            x: (..., d) base point on sphere (unit-norm)
            v: (..., d) tangent vector at x

        Returns:
            y: (..., d) point on sphere
        """
        norm_v = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.cos(norm_v) * x + torch.sin(norm_v) * (v / norm_v)

    @staticmethod
    def few_step_velocity_target(
        h_s: Tensor,
        W_E: Tensor,
        probs: Tensor,
        kappa_s: Union[Tensor, float],
        kappa_t: Union[Tensor, float],
        cdf_table,
    ) -> Tensor:
        """Exact finite-step velocity via CDF transport (flow map inference).

        For each token k, exactly transports h_s from kappa_s to kappa_t:
            mu_s = w_k . h_s
            mu_t = F_{kappa_t}^{-1}(F_{kappa_s}(mu_s))
            Phi_k = mu_t * w_k + sqrt(1 - mu_t^2) * xi_k

        where xi_k = normalize(h_s - mu_s * w_k) is the perpendicular direction.
        The marginal velocity is the posterior-weighted log map average.

        Converges to vmf_velocity_target as dkappa -> 0.

        Args:
            h_s:      (B, L, d) state at kappa_s on S^{d-1}
            W_E:      (d, V) unit-norm embedding matrix
            probs:    (B, L, V) posterior probabilities
            kappa_s:  scalar or (B,) source concentration
            kappa_t:  scalar or (B,) target concentration
            cdf_table: VMFRadialCDF instance with cdf() and quantile()

        Returns:
            v_target: (B, L, d) tangent vector at h_s (velocity per unit kappa)
        """
        # Algebraic trick: decompose Phi_k = c_k·w_k + b_k·h_s and
        # log_{h_s}(Phi_k) = A_k·w_k + B_k·h_s with A_k, B_k scalars per (B,L,V).
        # Peak memory is then (B,L,V), not (B,L,V,d) — critical for large vocab.

        mu_s = torch.matmul(h_s, W_E)  # (B, L, V)
        u = cdf_table.cdf(mu_s, kappa_s)
        mu_t = cdf_table.quantile(u, kappa_t)  # (B, L, V)

        one_minus_ms2 = (1.0 - mu_s ** 2).clamp(min=1e-12)
        one_minus_mt2 = (1.0 - mu_t ** 2).clamp(min=0)
        b_k = (one_minus_mt2 / one_minus_ms2).sqrt()
        c_k = mu_t - b_k * mu_s

        cos_a = (mu_s * mu_t + (one_minus_ms2 * one_minus_mt2).clamp(min=0).sqrt()
                 ).clamp(-1 + 1e-7, 1 - 1e-7)
        alpha = cos_a.acos()
        sin_a = alpha.sin().clamp(min=1e-8)
        scale = alpha / sin_a

        A_k = scale * c_k              # coefficient of w_k in log_{h_s}(Phi_k)
        B_k = scale * (b_k - cos_a)    # coefficient of h_s in log_{h_s}(Phi_k)

        if isinstance(kappa_s, Tensor) and kappa_s.dim() >= 1:
            dkappa = (kappa_t - kappa_s).clamp(min=1e-8)
            while dkappa.dim() < 3:
                dkappa = dkappa.unsqueeze(-1)
        else:
            dkappa = max(float(kappa_t) - float(kappa_s), 1e-8)

        pa = probs * A_k
        v_from_w = torch.matmul(pa, W_E.T)                # (B, L, d)
        scalar_h = (probs * B_k).sum(dim=-1, keepdim=True)  # (B, L, 1)
        return (v_from_w + scalar_h * h_s) / dkappa