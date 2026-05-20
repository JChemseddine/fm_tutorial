# Copied from https://github.com/JChemseddine/spherical (paper-release-anon)
# Paper: Spherical Flows for Sampling Discrete Distributions (arXiv:2605.05629)

"""CDCD-style piecewise linear time warp (Stancevic et al., 2024).

Fits an unnormalized CDF F_tilde to the observed loss curve via MSE.
The normalized CDF F is used for inverse-transform sampling of noise levels.

Key properties:
  - F_tilde(kappa) ≈ E[CE | kappa] — directly fit to observed loss
  - F(kappa) = F_tilde(kappa) / F_tilde(kappa_max) — normalized CDF for sampling
  - Trivial to invert (swap input/output roles)
  - Piecewise linear, monotone by construction
  - 2N parameters (input logits + output logits)

No EMA tracker, no anchor, no inversion chain. The warp IS the loss curve.
"""

import io
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CDCDWarp(nn.Module):
    """CDCD-style piecewise linear warp.

    Learns the loss curve F_tilde(kappa) via MSE. Samples noise levels
    from the normalized CDF using EMA-smoothed parameters.
    """

    def __init__(
        self,
        kappa_max: float,
        num_bins: int = 64,
        warmup_steps: int = 5000,
        ema_decay: float = 0.999,
        min_bin_frac: float = 1e-3,
        noise_increasing: bool = False,
    ):
        """
        Args:
            kappa_max: Maximum noise parameter value (fixed, not learned).
            num_bins: Number of piecewise linear bins (N).
            warmup_steps: Steps to ramp from uniform to learned schedule.
            ema_decay: EMA decay for warp parameters (used during sampling).
            min_bin_frac: Minimum bin size as fraction of 1/N (numerical stability).
            noise_increasing: If True, the input parameter already increases with
                noise (like VE's sigma). No internal flip needed. If False (default),
                the parameter increases with signal (like vMF kappa, geodesic t),
                and an internal flip maps to CDCD's noise-increasing convention.
        """
        super().__init__()
        self.kappa_max = kappa_max
        self.noise_increasing = noise_increasing
        self.N = num_bins
        self.warmup_steps = warmup_steps
        self.ema_decay = ema_decay
        self.min_bin_frac = min_bin_frac

        self.register_buffer('_step', torch.tensor(0, dtype=torch.long))

        # Init both logits to -log(N) → uniform bins → identity CDF
        init_val = -math.log(num_bins)
        self.l_t = nn.Parameter(torch.full((num_bins,), init_val))  # input logits
        self.l_u = nn.Parameter(torch.full((num_bins,), init_val))  # output logits

        # EMA copies (used for sampling — no grad flows through these)
        self.register_buffer('l_t_ema', torch.full((num_bins,), init_val))
        self.register_buffer('l_u_ema', torch.full((num_bins,), init_val))

    def _bin_edges(self, l_t: Tensor, l_u: Tensor, normalize_output: bool = True
                   ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute bin edges from logits.

        Args:
            l_t: (N,) input logits
            l_u: (N,) output logits
            normalize_output: If True, softmax on l_u (normalized CDF).
                              If False, exp on l_u (unnormalized = loss curve).

        Returns:
            e_t: (N+1,) input bin edges in [0, 1]
            e_u: (N+1,) output bin edges
            w_t: (N,) input bin widths
            w_u: (N,) output bin widths
        """
        # Input bins: always softmax (partition of [0,1])
        w_t = F.softmax(l_t, dim=0)
        w_t = w_t + self.min_bin_frac / self.N  # enforce minimum size
        w_t = w_t / w_t.sum()  # renormalize

        # Output bins: softmax (normalized) or exp (unnormalized)
        if normalize_output:
            w_u = F.softmax(l_u, dim=0)
            # No min_bin_frac for output — softmax is always positive,
            # and the floor creates artificial mass in dead CDF regions
        else:
            w_u = torch.exp(l_u)

        e_t = torch.cat([w_t.new_zeros(1), torch.cumsum(w_t, dim=0)])  # (N+1,)
        e_u = torch.cat([w_u.new_zeros(1), torch.cumsum(w_u, dim=0)])  # (N+1,)

        return e_t, e_u, w_t, w_u

    def _eval_piecewise(self, x: Tensor, e_in: Tensor, e_out: Tensor
                        ) -> Tuple[Tensor, Tensor]:
        """Evaluate piecewise linear function and its derivative.

        Maps x (in input-edge space) to y (in output-edge space) via
        linear interpolation within each bin.

        Args:
            x: (...) values in [e_in[0], e_in[-1]]
            e_in: (N+1,) input bin edges (monotone increasing)
            e_out: (N+1,) output bin edges (monotone increasing)

        Returns:
            y: (...) interpolated values
            dy_dx: (...) piecewise constant derivative
        """
        x_clamped = x.clamp(e_in[0], e_in[-1])

        # Find bin index: searchsorted gives first edge > x
        idx = torch.searchsorted(e_in[1:].contiguous(), x_clamped.contiguous())
        idx = idx.clamp(0, self.N - 1)

        # Linear interpolation within bin
        e_in_lo = e_in[idx]
        e_in_hi = e_in[idx + 1]
        e_out_lo = e_out[idx]
        e_out_hi = e_out[idx + 1]

        w_in = (e_in_hi - e_in_lo).clamp(min=1e-12)
        frac = (x_clamped - e_in_lo) / w_in

        y = e_out_lo + frac * (e_out_hi - e_out_lo)
        dy_dx = (e_out_hi - e_out_lo) / w_in

        return y, dy_dx

    def _to_noise(self, kappa: Tensor) -> Tensor:
        """Map input parameter to noise-increasing space for internal CDF.

        If noise_increasing=True (VE sigma): already noise-increasing, no flip.
        If noise_increasing=False (vMF kappa, geodesic t): flip to match CDCD.
        """
        if self.noise_increasing:
            return kappa
        return self.kappa_max - kappa

    def eval_unnormalized(self, kappa: Tensor) -> Tensor:
        """Evaluate F_tilde(kappa) — the unnormalized loss curve estimate.

        Internally operates in noise-increasing space so F_tilde is
        monotonically increasing, matching the CDCD paper convention.

        Args:
            kappa: (...) noise parameter values in [0, kappa_max]

        Returns:
            F_tilde: (...) estimated loss values
        """
        e_t, e_u, _, _ = self._bin_edges(self.l_t, self.l_u, normalize_output=False)
        t_prime = self._to_noise(kappa) / self.kappa_max  # flip + map to [0, 1]
        F_tilde, _ = self._eval_piecewise(t_prime, e_t, e_u)
        return F_tilde

    @torch.no_grad()
    def forward(self, u: Tensor) -> Tuple[Tensor, Tensor]:
        """Map u ∈ [0,1] → (kappa, F_prime). Uses EMA params, no grad.

        Evaluates the INVERSE of the normalized CDF: swap input/output roles.

        During warmup, blends with uniform (identity) mapping.

        Args:
            u: (...) uniform samples in [0, 1]

        Returns:
            kappa: (...) noise parameter values (detached)
            F_prime: (...) CDF derivative dF/dkappa (for importance weights)
        """
        u = u.clamp(0.0, 1.0)

        # Normalized CDF: input=t, output=u. Inverse: input=u, output=t.
        # So swap: use l_u_ema as "input" logits, l_t_ema as "output" logits.
        e_u, e_t, w_u, w_t = self._bin_edges(self.l_u_ema, self.l_t_ema, normalize_output=True)

        # Evaluate inverse CDF in noise-increasing space
        if self.noise_increasing:
            # VE: parameter already increases with noise. CDF maps u→noise directly.
            # u=0 → low noise (clean), u=1 → high noise. Output increases with u.
            t_noise, dt_du = self._eval_piecewise(u, e_u, e_t)
            kappa = self.kappa_max * t_noise
        else:
            # vMF/geodesic: parameter increases with signal. Flip u so output
            # increases with u (kappa: low=noisy, high=clean).
            u_flipped = 1.0 - u
            t_noise, dt_du = self._eval_piecewise(u_flipped, e_u, e_t)
            kappa = self.kappa_max * (1.0 - t_noise)

        # F'(kappa) for importance weighting
        F_prime = (1.0 / (dt_du * self.kappa_max)).clamp(min=1e-8)

        # Warmup blend (both increase with u)
        alpha = min(1.0, self._step.item() / max(1, self.warmup_steps))
        kappa_uniform = self.kappa_max * u
        kappa = (1.0 - alpha) * kappa_uniform + alpha * kappa

        return kappa, F_prime

    @torch.no_grad()
    def importance_weight(self, kappa: Tensor) -> Tensor:
        """Compute importance weight 1/F'(kappa) for CE loss correction.

        Args:
            kappa: (...) noise parameter values

        Returns:
            w: (...) importance weights (detached, normalized to mean=1)
        """
        # Evaluate normalized CDF derivative at kappa (in noise-increasing space)
        e_t, e_u, _, _ = self._bin_edges(
            self.l_t_ema, self.l_u_ema, normalize_output=True)
        t_prime = self._to_noise(kappa) / self.kappa_max
        _, F_prime_kappa = self._eval_piecewise(t_prime, e_t, e_u)

        # importance weight = 1/F'(kappa), normalized to mean=1
        w = (1.0 / F_prime_kappa.clamp(min=1e-8))
        w = w / w.mean().clamp(min=1e-8)
        return w

    def fit_loss(self, kappa: Tensor, observed_ce: Tensor,
                 ce_floor: float = 0.0, power: float = 1.0) -> Tensor:
        """Compute MSE fitting loss for the unnormalized CDF.

        With power=1 (default): MSE(F_tilde(kappa), CE) — standard CDCD.
        With power!=1: MSE(S(F_norm) * F_max, CE) where S(x) = x^p.
        This shapes the CE(u) curve: CE(u) = CE_max * u^p (u in noise-increasing space).
        During sampling (noisy→clean, s = 1-u): CE(s) = CE_max * (1-s)^p.
          p=1: linear (equal entropy per step)
          p>1: fast early resolution (most entropy gone in first steps, slow refinement)
          p<1: slow early resolution (CE stays high, sharp drop near clean end)

        Args:
            kappa: (B,) noise parameter values (detached)
            observed_ce: (B,) observed CE loss per sample (detached)
            ce_floor: floor below which CE is treated as 0
            power: warp target power p (1.0 = standard CDCD)

        Returns:
            mse_loss: scalar MSE loss (has grad for warp params)
        """
        observed = (observed_ce - ce_floor).clamp(min=0.0) if ce_floor > 0 else observed_ce

        if power == 1.0:
            F_tilde = self.eval_unnormalized(kappa)
            return F.mse_loss(F_tilde, observed)

        # Shaped target: S(F_norm(kappa)) * F_max ≈ CE(kappa)
        F_tilde = self.eval_unnormalized(kappa)
        # F_max = F_tilde at max-noise end.
        # For VE (noise_increasing=True): max noise = kappa_max (sigma_max).
        # For vMF/geodesic (noise_increasing=False): max noise = 0 (kappa=0).
        noise_end = self.kappa_max if self.noise_increasing else 0.0
        kappa_noise_end = torch.tensor([noise_end], device=kappa.device, dtype=kappa.dtype)
        F_max = self.eval_unnormalized(kappa_noise_end).clamp(min=1e-8)
        F_norm = (F_tilde / F_max).clamp(0.0, 1.0)
        prediction = F_norm.pow(power) * F_max
        return F.mse_loss(prediction, observed)

    def step(self) -> None:
        """Update EMA of warp parameters and increment step counter."""
        self._step += 1
        d = self.ema_decay
        self.l_t_ema.mul_(d).add_(self.l_t.data, alpha=1 - d)
        self.l_u_ema.mul_(d).add_(self.l_u.data, alpha=1 - d)

    @torch.no_grad()
    def get_kappa_max(self) -> float:
        """Return kappa_max (fixed, not learned)."""
        return self.kappa_max

    @torch.no_grad()
    def visualize(self, num_inference_steps: int = 100):
        """Create diagnostic plot of the learned warp.

        Returns a PIL Image with 4 panels:
        1. F_tilde(kappa) — unnormalized loss curve
        2. F'(kappa) — sampling PDF
        3. Training kappa distribution
        4. Inference step sizes
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image

        device = self.l_t.device
        kappa_grid = torch.linspace(0, self.kappa_max, 500, device=device)

        # Panel 1: F_tilde (loss curve)
        F_tilde = self.eval_unnormalized(kappa_grid)

        # Panel 2: F' (PDF) using EMA params
        e_t, e_u, _, _ = self._bin_edges(
            self.l_t_ema, self.l_u_ema, normalize_output=True)
        t_prime = self._to_noise(kappa_grid) / self.kappa_max
        _, F_prime = self._eval_piecewise(t_prime, e_t, e_u)

        # Panel 3: Training kappa dist
        u_samples = torch.linspace(0.0, 1.0, 10000, device=device)
        kappa_train, _ = self.forward(u_samples)

        # Panel 4: Inference steps
        S = num_inference_steps
        u_steps = torch.linspace(0.0, 1.0, S + 1, device=device)
        kappa_steps, _ = self.forward(u_steps)

        # Plot
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))

        kappa_np = kappa_grid.cpu().numpy()

        ax = axes[0]
        ax.plot(kappa_np, F_tilde.cpu().numpy(), color='steelblue', linewidth=2)
        ax.set_xlabel('kappa')
        ax.set_ylabel('CE')
        ax.set_title(f'F_tilde (loss curve), kmax={self.kappa_max:.0f}')

        ax = axes[1]
        ax.plot(kappa_np, F_prime.cpu().numpy(), color='coral', linewidth=2)
        ax.set_xlabel('kappa')
        ax.set_ylabel("F'(kappa)")
        ax.set_title("Sampling PDF")

        ax = axes[2]
        kappa_train_np = kappa_train.cpu().numpy()
        ax.hist(kappa_train_np, bins=50, density=True, color='steelblue',
                alpha=0.7, edgecolor='white', linewidth=0.5, label='Learned')
        ax.axhline(1.0 / self.kappa_max, color='k', linestyle='--',
                    alpha=0.5, label='Uniform')
        ax.set_xlabel('kappa')
        ax.set_ylabel('Density')
        ax.set_title('Training kappa dist')
        ax.legend()

        ax = axes[3]
        dk = (kappa_steps[1:] - kappa_steps[:-1]).cpu().numpy()
        u_mid = ((u_steps[1:] + u_steps[:-1]) / 2).cpu().numpy()
        ax.bar(u_mid, dk, width=1.0 / S * 0.9,
               color='seagreen', alpha=0.7, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('u')
        ax.set_ylabel('delta_kappa')
        ax.set_title(f'Inference Steps ({S})')

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)

    def state_dict_extra(self):
        """Extra state for checkpointing."""
        return {'_step': self._step.clone()}

    def load_state_dict_extra(self, d):
        """Restore extra state."""
        if '_step' in d:
            self._step.copy_(d['_step'])