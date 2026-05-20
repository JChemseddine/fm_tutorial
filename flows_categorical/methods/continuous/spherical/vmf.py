# Copied from https://github.com/JChemseddine/spherical (paper-release-anon)
# Paper: Spherical Flows for Sampling Discrete Distributions (arXiv:2605.05629)

"""Von Mises-Fisher utilities: psi_tilde lookup table and batched vMF sampling.

The psi_tilde table solves the continuity equation ODE for the vMF conditional
probability path on S^{d-1}. It replaces alpha/sin(alpha) from SLERP.

psi_tilde(mu, kappa) is independent of kappa_max — changing kappa_max only
requires extending the kappa range, not recomputation.

vMF sampling supports two methods:
  - "rejection": Wood's (1994) rejection algorithm (default, exact)
  - "cdf": Precomputed CDF table with inverse-CDF sampling (faster, approximate)
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Union


class PsiTable:
    """Precomputed 2D lookup table for psi_tilde(mu, kappa).

    psi_tilde is the scalar weight from the vMF continuity equation on S^{d-1}.
    It solves the ODE derived from the transport equation for the conditional
    probability path p_t(x|x_1) = vMF(x | x_1, kappa(t)):

        H'(mu) = -(mu - A_d(kappa)) * (1-mu^2)^{(d-3)/2} - kappa * H
        psi_tilde(mu, kappa) = H(mu) / (1 - mu^2)^{(d-1)/2}

    where A_d(kappa) = I_{d/2}(kappa) / I_{d/2-1}(kappa) is the mean resultant
    length, and H(-1) = 0 is the boundary condition.

    The table is independent of kappa_max — it depends only on dimension d
    and the kappa range covered by the grid.
    """

    def __init__(self, dim: int, kappa_range: float = 50.0, grid_size: int = 1000):
        """
        Args:
            dim: Embedding dimension d (sphere is S^{d-1})
            kappa_range: Maximum kappa value in the table
            grid_size: Number of grid points per axis
        """
        self.dim = dim
        self.kappa_range = kappa_range
        self.grid_size = grid_size

        eps = 1e-6
        self.mu_min = -1.0 + eps
        self.mu_max = 1.0 - eps

        table_np = self._compute_table()
        self.table = torch.from_numpy(table_np).float()  # (grid_size, grid_size) [mu_idx, kappa_idx]

    @staticmethod
    def _compute_A_d_cf(kappa: np.ndarray, d: int, n_iter: int = 200) -> np.ndarray:
        """A_d(kappa) = I_{d/2}(kappa) / I_{d/2-1}(kappa) via backward continued fraction.

        Uses the recurrence r_n = 1 / (2n/kappa + r_{n+1}) starting from
        r_N = 0 at large N. Stable for any d and kappa, unlike ive which
        underflows when kappa << d.
        """
        nu = d / 2.0
        result = np.zeros_like(kappa, dtype=np.float64)
        mask = kappa > 1e-10
        if not mask.any():
            return result
        k = kappa[mask]
        # Backward CF: r = 0, then for n = nu+n_iter, nu+n_iter-1, ..., nu:
        #   r ← 1/(2n/k + r)
        # After the loop, r = I_nu(k)/I_{nu-1}(k) = A_d(k).
        # Using float n throughout handles both integer and half-integer nu.
        r = np.zeros_like(k)
        for i in range(n_iter + 1):
            n = nu + n_iter - i  # nu+n_iter, nu+n_iter-1, ..., nu
            r = 1.0 / (2.0 * n / k + r)
        result[mask] = r
        return result

    def _compute_table(self) -> np.ndarray:
        """Compute psi_tilde on a 2D grid via the flux method.

        Uses the continuity equation in flux form:
            psi_tilde(mu, kappa) = -J(mu) / [f(mu) * (1 - mu^2)]
        where f is the unnormalized vMF radial density and J is the
        cumulative flux integral.

        Works in max-shifted log-space: g = exp(log_f - max(log_f)).
        The shift cancels in the ratio, so g never underflows near the
        density peak. Boundary regions where g underflows are filled with
        analytical values and a smooth linear blend.

        Vectorized over kappa: one forward cumsum along mu, all kappa in parallel.
        """
        d = self.dim
        N = self.grid_size
        alpha = (d - 3) / 2.0

        eps = 1e-7
        mu = np.linspace(-1 + eps, 1 - eps, N)
        kappa = np.linspace(0, self.kappa_range, N)
        dmu = mu[1] - mu[0]
        one_minus_mu2 = 1.0 - mu ** 2  # (N,)

        A_d_all = self._compute_A_d_cf(kappa, d)  # (N,)

        # log unnormalized vMF radial density: (N_mu, N_kappa)
        log_1mm2 = np.log(np.maximum(one_minus_mu2, 1e-300))
        log_f = kappa[None, :] * mu[:, None] + alpha * log_1mm2[:, None]

        # Max-shift per kappa column. exp(M) cancels in the ratio.
        log_f_max = np.max(log_f, axis=0, keepdims=True)  # (1, N_kappa)
        g = np.exp(log_f - log_f_max)  # (N_mu, N_kappa), peak=1

        # Flux integrand: g * (mu - A_d)
        integrand = g * (mu[:, None] - A_d_all[None, :])  # (N_mu, N_kappa)

        # Cumulative trapezoid from the RIGHT: J(mu) = -int_mu^1 integrand dmu'.
        # Uses J(1) = 0 (exact, since A_d = E[mu] under vMF). Integrating from
        # the right keeps precision good even where g decays into the float
        # floor far from the peak, because the partial sum has already traversed
        # the density peak (where magnitudes are O(1)) before entering the
        # boundary tail.
        trap = 0.5 * (integrand[:-1] + integrand[1:]) * dmu
        cum_from_right = np.concatenate(
            [np.cumsum(trap[::-1], axis=0)[::-1], np.zeros((1, N))], axis=0
        )
        J = -cum_from_right

        # psi_tilde = -J / [g * (1 - mu^2)]
        denom = g * one_minus_mu2[:, None]
        # At d~800, (1-mu^2)^alpha collapses g below ~1e-16 for |mu-mu_peak|>~0.3;
        # 1e-14 catches those cells safely before their J/g ratio loses precision.
        SAFE_THRESH = 1e-14
        safe = denom > SAFE_THRESH
        table = np.zeros_like(g)
        table[safe] = -J[safe] / denom[safe]

        # Analytical boundary values
        psi_left = (1 + A_d_all) / (d - 1)   # (N_kappa,)
        psi_right = (1 - A_d_all) / (d - 1)  # (N_kappa,)

        # Fill unsafe cells via a linear blend from the last-reliable computed
        # cell to the exact analytical endpoint (psi_left at mu=-1, psi_right
        # at mu=+1). psi_tilde is smooth/monotone in the boundary regime, so
        # this is within a few percent of the true value. The old boundary
        # logic used a fixed-width N_BLEND with a constant outside; the linear
        # blend across the entire unsafe span is smoother and fully fills
        # unsafe islands without leaving any cells at zero.
        for j in range(N):
            col_safe = safe[:, j]
            if not col_safe.any():
                table[:, j] = np.linspace(psi_left[j], psi_right[j], N)
                continue
            first_safe = int(np.argmax(col_safe))
            last_safe = N - 1 - int(np.argmax(col_safe[::-1]))

            # Left blend: psi_left[j] -> table[first_safe, j]
            if first_safe > 0:
                anchor_L = table[first_safe, j]
                table[:first_safe, j] = np.linspace(
                    psi_left[j], anchor_L, first_safe + 1
                )[:-1]

            # Right blend: table[last_safe, j] -> psi_right[j]
            if last_safe < N - 1:
                anchor_R = table[last_safe, j]
                table[last_safe + 1:, j] = np.linspace(
                    anchor_R, psi_right[j], N - last_safe
                )[1:]

        return table

    def to(self, device: torch.device) -> 'PsiTable':
        """Move table to device."""
        self.table = self.table.to(device)
        return self

    def lookup(self, cos_vals: Tensor, kappa: Union[Tensor, float]) -> Tensor:
        """Bilinear interpolation lookup into the psi_tilde table.

        Args:
            cos_vals: (...) cosine similarities (mu values in [-1, 1])
            kappa: scalar or (...) broadcastable concentration values

        Returns:
            psi_tilde: same shape as cos_vals
        """
        # Map mu to continuous grid index
        mu_idx = (cos_vals - self.mu_min) / (self.mu_max - self.mu_min) * (self.grid_size - 1)
        mu_idx = mu_idx.clamp(0, self.grid_size - 1)

        # Map kappa to continuous grid index
        if isinstance(kappa, (int, float)):
            kappa_idx = torch.full_like(cos_vals, kappa / self.kappa_range * (self.grid_size - 1))
        else:
            kappa_t = kappa
            while kappa_t.dim() < cos_vals.dim():
                kappa_t = kappa_t.unsqueeze(-1)
            kappa_idx = kappa_t.expand_as(cos_vals) / self.kappa_range * (self.grid_size - 1)
        kappa_idx = kappa_idx.clamp(0, self.grid_size - 1)

        # Integer indices for four corners
        mu_lo = mu_idx.long().clamp(0, self.grid_size - 2)
        mu_hi = mu_lo + 1
        kappa_lo = kappa_idx.long().clamp(0, self.grid_size - 2)
        kappa_hi = kappa_lo + 1

        # Fractional parts
        mu_frac = mu_idx - mu_lo.float()
        kappa_frac = kappa_idx - kappa_lo.float()

        # Four corner values
        v00 = self.table[mu_lo, kappa_lo]
        v01 = self.table[mu_lo, kappa_hi]
        v10 = self.table[mu_hi, kappa_lo]
        v11 = self.table[mu_hi, kappa_hi]

        # Bilinear interpolation
        result = (v00 * (1 - mu_frac) * (1 - kappa_frac)
                  + v01 * (1 - mu_frac) * kappa_frac
                  + v10 * mu_frac * (1 - kappa_frac)
                  + v11 * mu_frac * kappa_frac)

        return result

    def lookup_1d(self, cos_vals: Tensor, kappa: Union[Tensor, float]) -> Tensor:
        """1D linear interpolation when kappa is scalar per step.

        Pre-slices the 2D table along the kappa axis (one interpolation on a
        1000-element vector), then interpolates only along the mu axis.
        Numerically identical to lookup() when kappa is constant across cos_vals.
        Creates 3 intermediate (B,L,V) tensors instead of 11.
        """
        G = self.grid_size

        # Interpolate table along kappa axis → 1D function of mu: (G,)
        if isinstance(kappa, (int, float)):
            k_idx = kappa / self.kappa_range * (G - 1)
        else:
            k_idx = kappa.item() / self.kappa_range * (G - 1)
        k_idx = max(0.0, min(float(k_idx), G - 2))
        k_lo = int(k_idx)
        k_frac = k_idx - k_lo
        psi_1d = self.table[:, k_lo] * (1 - k_frac) + self.table[:, k_lo + 1] * k_frac
        psi_1d = psi_1d.to(cos_vals.device)

        # 1D linear interpolation along mu axis
        mu_idx = (cos_vals - self.mu_min) / (self.mu_max - self.mu_min) * (G - 1)
        mu_idx = mu_idx.clamp(0, G - 2)
        lo = mu_idx.long()
        frac = mu_idx - lo.float()
        return psi_1d[lo] * (1 - frac) + psi_1d[lo + 1] * frac


def sample_vmf(mu: Tensor, kappa: Tensor) -> Tensor:
    """Sample from von Mises-Fisher distribution on S^{d-1}.

    Uses Wood's (1994) rejection sampling algorithm. Handles per-element
    concentration parameters and the kappa=0 (uniform) edge case.

    Args:
        mu: (..., d) mean directions (unit norm)
        kappa: (...) or scalar, concentration (broadcastable with mu[..., 0])

    Returns:
        samples: same shape as mu, on S^{d-1}
    """
    d = mu.shape[-1]
    shape = mu.shape[:-1]
    device = mu.device

    if not isinstance(kappa, Tensor):
        kappa = torch.tensor(kappa, device=device, dtype=mu.dtype)
    # Broadcast kappa to match mu's leading dims (e.g. (B,) -> (B, L))
    while kappa.dim() < len(shape):
        kappa = kappa.unsqueeze(-1)
    kappa = kappa.expand(shape)
    kappa_flat = kappa.reshape(-1)
    n = kappa_flat.shape[0]

    # Output cosine with mean direction
    w = torch.zeros(n, device=device, dtype=mu.dtype)

    # Split into kappa > 0 (rejection sampling) and kappa ≈ 0 (uniform)
    active = kappa_flat > 1e-6

    if active.any():
        k = kappa_flat[active]
        n_active = k.shape[0]
        dm1 = d - 1

        # Wood's algorithm parameters
        b = (-2 * k + torch.sqrt(4 * k**2 + dm1**2)) / dm1
        x0 = (1 - b) / (1 + b)
        c = k * x0 + dm1 * torch.log(1 - x0**2)

        accepted = torch.zeros(n_active, dtype=torch.bool, device=device)
        w_active = torch.zeros(n_active, device=device, dtype=mu.dtype)

        for _ in range(200):
            if accepted.all():
                break

            remaining = ~accepted
            n_rem = remaining.sum().item()

            # z ~ Beta((d-1)/2, (d-1)/2)
            z = torch.distributions.Beta(
                torch.full((n_rem,), dm1 / 2.0, device=device),
                torch.full((n_rem,), dm1 / 2.0, device=device),
            ).sample()

            b_rem = b[remaining]
            w_prop = (1 - (1 + b_rem) * z) / (1 - (1 - b_rem) * z)

            # Acceptance criterion
            k_rem = k[remaining]
            c_rem = c[remaining]
            x0_rem = x0[remaining]

            log_ratio = k_rem * w_prop + dm1 * torch.log((1 - x0_rem * w_prop).clamp(min=1e-30)) - c_rem
            u = torch.rand(n_rem, device=device)
            accept = log_ratio >= torch.log(u.clamp(min=1e-30))

            idx = torch.where(remaining)[0]
            newly_accepted = idx[accept]
            w_active[newly_accepted] = w_prop[accept]
            accepted[newly_accepted] = True

        # Rare fallback: unaccepted samples get w=1 (at mean direction)
        if not accepted.all():
            w_active[~accepted] = 1.0

        w[active] = w_active

    # kappa ≈ 0: uniform on sphere -> cosine ~ Beta-transformed
    if (~active).any():
        n_uniform = (~active).sum().item()
        if d >= 3:
            u_beta = torch.distributions.Beta(
                torch.full((n_uniform,), (d - 1) / 2.0, device=device),
                torch.full((n_uniform,), (d - 1) / 2.0, device=device),
            ).sample()
            w[~active] = 2 * u_beta - 1
        else:
            # d=2: uniform angle
            w[~active] = 2 * torch.rand(n_uniform, device=device, dtype=mu.dtype) - 1

    w = w.reshape(shape)

    # Construct sample: x = w * mu + sqrt(1 - w^2) * v
    # where v is uniform on S^{d-2} in the tangent plane at mu
    mu_flat = mu.reshape(-1, d)
    z_rand = torch.randn(*shape, d, device=device, dtype=mu.dtype).reshape(-1, d)

    # Project out mu component to get tangent direction
    z_perp = z_rand - (z_rand * mu_flat).sum(dim=-1, keepdim=True) * mu_flat
    v = F.normalize(z_perp, dim=-1)
    v = v.reshape(*shape, d)

    w_expanded = w.unsqueeze(-1)
    sin_w = torch.sqrt((1 - w_expanded**2).clamp(min=0))
    samples = w_expanded * mu + sin_w * v

    return samples


class VMFRadialCDF:
    """Precomputed CDF/quantile table for vMF radial (cosine) distribution.

    For vMF on S^{d-1} with concentration kappa, the cosine mu = x . mean
    has unnormalized pdf:  log f(mu; kappa, d) = kappa*mu + ((d-3)/2)*log(1-mu^2)

    CDF is computed via cumulative trapezoid on a uniform mu-grid.
    Quantile (inverse CDF) uses searchsorted + linear interpolation.

    No scipy needed — pure NumPy at init, pure PyTorch at runtime.
    """

    def __init__(self, dim: int, kappa_range: float = 50.0,
                 kappa_grid_size: int = 2000, mu_grid_size: int = 4096):
        self.dim = dim
        self.kappa_range = kappa_range
        self.kappa_grid_size = kappa_grid_size
        self.mu_grid_size = mu_grid_size

        mu_np, kappa_np, cdf_np = self._compute_table()
        self.mu_grid = torch.from_numpy(mu_np)
        self.kappa_grid_t = torch.from_numpy(kappa_np)
        self.cdf_table = torch.from_numpy(cdf_np)  # (K, M)

        self.mu_min = float(mu_np[0])
        self.mu_max = float(mu_np[-1])

    def _compute_table(self):
        d = self.dim
        nu = (d - 3) / 2.0

        eps = 1e-6
        mu = np.linspace(-1.0 + eps, 1.0 - eps, self.mu_grid_size)
        kappa = np.linspace(0.0, self.kappa_range, self.kappa_grid_size)
        dmu = mu[1] - mu[0]

        # Log unnormalized pdf: (K, M)
        log_f = kappa[:, None] * mu[None, :]
        if abs(nu) > 1e-10:
            log_f = log_f + nu * np.log(np.maximum(1.0 - mu ** 2, 1e-30))[None, :]

        # Shift each row by its max for stability
        log_f -= log_f.max(axis=1, keepdims=True)
        f = np.exp(log_f)

        # Cumulative trapezoid
        cdf = np.zeros_like(f)
        cdf[:, 1:] = np.cumsum(0.5 * (f[:, :-1] + f[:, 1:]) * dmu, axis=1)

        # Normalize
        total = cdf[:, -1:].clip(min=1e-30)
        cdf /= total
        cdf = np.maximum.accumulate(cdf, axis=1)

        return mu.astype(np.float32), kappa.astype(np.float32), cdf.astype(np.float32)

    def to(self, device) -> 'VMFRadialCDF':
        self.mu_grid = self.mu_grid.to(device)
        self.kappa_grid_t = self.kappa_grid_t.to(device)
        self.cdf_table = self.cdf_table.to(device)
        return self

    @torch.no_grad()
    def cdf(self, mu: Tensor, kappa: Union[Tensor, float]) -> Tensor:
        """Forward CDF: (mu, kappa) -> F(mu; kappa, d) in [0, 1].

        Args:
            mu:    (...) cosine values in [-1, 1]
            kappa: scalar or (...) broadcastable concentration

        Returns:
            cdf_vals: same shape as mu
        """
        shape = mu.shape
        mu_flat = mu.reshape(-1)
        N = mu_flat.shape[0]

        # Map mu to continuous grid index
        mu_idx = ((mu_flat - self.mu_min) / (self.mu_max - self.mu_min)
                  * (self.mu_grid_size - 1)).clamp(0, self.mu_grid_size - 1)

        # Map kappa to continuous grid index
        if isinstance(kappa, (int, float)):
            kappa_t = torch.full((N,), kappa, dtype=torch.float32, device=mu.device)
        else:
            kappa_float = kappa.float()
            while kappa_float.dim() < len(shape):
                kappa_float = kappa_float.unsqueeze(-1)
            kappa_t = kappa_float.expand(shape).reshape(-1)

        kappa_idx = (kappa_t / self.kappa_range * (self.kappa_grid_size - 1)
                     ).clamp(0, self.kappa_grid_size - 1)

        # Integer corners for bilinear interpolation
        mu_lo = mu_idx.long().clamp(0, self.mu_grid_size - 2)
        mu_hi = mu_lo + 1
        k_lo = kappa_idx.long().clamp(0, self.kappa_grid_size - 2)
        k_hi = k_lo + 1

        mu_frac = mu_idx - mu_lo.float()
        k_frac = kappa_idx - k_lo.float()

        # Bilinear interpolation
        v00 = self.cdf_table[k_lo, mu_lo]
        v01 = self.cdf_table[k_lo, mu_hi]
        v10 = self.cdf_table[k_hi, mu_lo]
        v11 = self.cdf_table[k_hi, mu_hi]

        cdf_vals = (v00 * (1 - k_frac) * (1 - mu_frac)
                    + v01 * (1 - k_frac) * mu_frac
                    + v10 * k_frac * (1 - mu_frac)
                    + v11 * k_frac * mu_frac)

        return cdf_vals.reshape(shape)

    @torch.no_grad()
    def quantile(self, u: Tensor, kappa: Union[Tensor, float]) -> Tensor:
        """Inverse CDF: (u, kappa) -> mu in [-1, 1].

        Args:
            u:     (...) uniform values in [0, 1]
            kappa: scalar or (...) broadcastable concentration

        Returns:
            mu: same shape as u
        """
        u = u.clamp(0.0, 1.0)
        shape = u.shape
        u_flat = u.reshape(-1)
        N = u_flat.shape[0]

        # Scalar kappa fast path: build one (M,) CDF row, broadcast searchsorted.
        # Avoids materializing (N, M) — critical when N ~ B·L·V at LM1B scale.
        if isinstance(kappa, (int, float)):
            kappa_idx_f = max(0.0, min(self.kappa_grid_size - 1.0,
                                        float(kappa) / self.kappa_range * (self.kappa_grid_size - 1)))
            k_lo_s = min(self.kappa_grid_size - 2, int(kappa_idx_f))
            k_hi_s = k_lo_s + 1
            k_frac_s = kappa_idx_f - k_lo_s
            row = ((1.0 - k_frac_s) * self.cdf_table[k_lo_s]
                   + k_frac_s * self.cdf_table[k_hi_s]).contiguous()  # (M,)
            idx = torch.searchsorted(row, u_flat).clamp(1, self.mu_grid_size - 1)
            cdf_lo = row[idx - 1]
            cdf_hi = row[idx]
            mu_lo = self.mu_grid[idx - 1]
            mu_hi = self.mu_grid[idx]
            frac = ((u_flat - cdf_lo) / (cdf_hi - cdf_lo).clamp(min=1e-12)).clamp(0.0, 1.0)
            return (mu_lo + frac * (mu_hi - mu_lo)).reshape(shape)

        kappa_float = kappa.float()
        while kappa_float.dim() < len(shape):
            kappa_float = kappa_float.unsqueeze(-1)
        kappa_t = kappa_float.expand(shape).reshape(-1)

        kappa_idx_f = (kappa_t / self.kappa_range * (self.kappa_grid_size - 1)
                       ).clamp(0, self.kappa_grid_size - 1)
        k_lo = kappa_idx_f.long().clamp(0, self.kappa_grid_size - 2)
        k_hi = k_lo + 1
        k_frac = (kappa_idx_f - k_lo.float()).unsqueeze(1)  # (N, 1)

        # Bilinear interpolation in kappa: blend CDF rows
        cdf_rows = (1.0 - k_frac) * self.cdf_table[k_lo] + k_frac * self.cdf_table[k_hi]  # (N, M)

        # searchsorted for inverse CDF
        idx = torch.searchsorted(cdf_rows.contiguous(), u_flat.unsqueeze(1)).squeeze(1)
        idx = idx.clamp(1, self.mu_grid_size - 1)

        # Linear interpolation in mu
        cdf_lo = cdf_rows.gather(1, (idx - 1).unsqueeze(1)).squeeze(1)
        cdf_hi = cdf_rows.gather(1, idx.unsqueeze(1)).squeeze(1)
        mu_lo = self.mu_grid[idx - 1]
        mu_hi = self.mu_grid[idx]

        frac = ((u_flat - cdf_lo) / (cdf_hi - cdf_lo).clamp(min=1e-12)).clamp(0.0, 1.0)
        mu_out = mu_lo + frac * (mu_hi - mu_lo)

        return mu_out.reshape(shape)


def sample_vmf_cdf(mu: Tensor, kappa: Tensor, cdf_table: VMFRadialCDF) -> Tensor:
    """Sample from vMF using precomputed CDF table (no rejection loop).

    Same interface as sample_vmf but uses inverse-CDF sampling for the radial
    (cosine) component. Much faster on GPU, slightly approximate due to grid
    interpolation.

    Args:
        mu: (..., d) mean directions (unit norm)
        kappa: (...) or scalar, concentration (broadcastable with mu[..., 0])
        cdf_table: VMFRadialCDF instance

    Returns:
        samples: same shape as mu, on S^{d-1}
    """
    d = mu.shape[-1]
    shape = mu.shape[:-1]
    device = mu.device

    if not isinstance(kappa, Tensor):
        kappa = torch.tensor(kappa, device=device, dtype=mu.dtype)
    while kappa.dim() < len(shape):
        kappa = kappa.unsqueeze(-1)
    kappa = kappa.expand(shape)

    # Sample cosine via inverse CDF
    u = torch.rand(shape, device=device, dtype=mu.dtype)
    w = cdf_table.quantile(u, kappa)  # (...) cosine values

    # Construct sample: x = w * mu + sqrt(1 - w^2) * v
    mu_flat = mu.reshape(-1, d)
    z_rand = torch.randn(*shape, d, device=device, dtype=mu.dtype).reshape(-1, d)

    # Random tangent direction orthogonal to mu
    z_perp = z_rand - (z_rand * mu_flat).sum(dim=-1, keepdim=True) * mu_flat
    v = F.normalize(z_perp, dim=-1).reshape(*shape, d)

    w_expanded = w.unsqueeze(-1)
    sin_w = torch.sqrt((1 - w_expanded**2).clamp(min=0))
    samples = w_expanded * mu + sin_w * v

    return samples