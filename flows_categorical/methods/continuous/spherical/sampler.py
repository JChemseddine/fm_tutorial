# Copied from https://github.com/JChemseddine/spherical (paper-release-anon)
# Paper: Spherical Flows for Sampling Discrete Distributions (arXiv:2605.05629)

"""ODE sampler on the sphere using vMF-derived velocity.

Softmax velocity (exact):
    v_target = vmf_velocity_target(h, W_E, probs, psi_table, kappa)
    v = v_target    (uniform or warp-aware kappa stepping)

Velocity head:
    D_hat = velocity_head(backbone_features)
    v = proj_tan(D_hat, h_t)

Predictor-corrector (pc_softmax / pc_velocity_head):
    Predict: Euler step with velocity (ODE)
    Correct: Langevin step with score (project noise + score onto tangent plane)
    Score = κ * Σ_k p(k|h) * proj_tan(w_k, h) — free from same forward pass.

Integration modes:
  1. Uniform kappa (default): kappa_n = n * kappa_max / S, dk = const.
  2. Warp-aware (warp_aware_sampling=True): step uniformly in u,
     evaluate F(u_i) at S+1 grid points to get exact kappa values.
     dk_i = kappa_{i+1} - kappa_i (no first-order approximation).

Source distribution: uniform on sphere (kappa(0) = 0).
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional, TYPE_CHECKING
from tqdm import tqdm

from .path import SphericalPath

if TYPE_CHECKING:
    from flows_categorical.schedule.cdcd_warp import CDCDWarp
    from .vmf import PsiTable


class SphericalODESampler:
    """ODE/SDE sampler on S^{d-1} for discrete generation (vMF or geodesic noise process)."""

    def __init__(self, model: torch.nn.Module, config: 'Config',
                 psi_table: 'PsiTable' = None,
                 warp: Optional['CDCDWarp'] = None,
                 noise_process: str = "vmf",
                 vmf_cdf_table=None):
        self.model = model
        self.embed_dim = model.embed_dim
        self.sequence_length = model.sequence_length
        self.kappa_max = config.flow.kappa_max
        self.num_steps = config.flow.sampling_steps
        self.method = config.flow.sampler_method
        self.psi_table = psi_table
        self.warp = warp
        self.vmf_cdf_table = vmf_cdf_table
        self.time_conditioning = getattr(config.model, 'time_conditioning', False)
        self.warp_aware_sampling = config.flow.warp_aware_sampling
        self.noise_process = noise_process
        self.path = SphericalPath()
        self.blank_token_id = config.flow.blank_token_id

        # Corrector config (for pc_* methods)
        self.corrector_steps = config.flow.corrector_steps
        self.corrector_interval = config.flow.corrector_interval
        self.corrector_epsilon = config.flow.corrector_epsilon
        # Back-compat: prefer new unified flag, fall back to legacy kappa-specific one.
        self.corrector_scaling = getattr(
            config.flow, 'corrector_scaling',
            getattr(config.flow, 'corrector_kappa_scaling', True))

        # (kappa scaling removed — learned_tau only)

        # Geodesic has no score — disable corrector and refuse SDE
        if noise_process == "geodesic":
            if self.corrector_steps > 0:
                import warnings
                warnings.warn("Langevin corrector disabled for geodesic (no score available)")
                self.corrector_steps = 0
            if self.method in ("sde_softmax", "sde_velocity_head",
                               "pc_sde_softmax", "pc_sde_velocity_head"):
                raise ValueError(
                    f"SDE sampling not supported for geodesic (no score function). "
                    f"Use softmax or velocity_head instead, got '{self.method}'."
                )

        # SDE config
        self.sde_sigma = config.flow.sde_sigma

        # Exponential map retraction (optional, helps at few-step)
        self.use_expmap = getattr(config.flow, 'use_expmap', False)

    def _sigma(self, kappa_t, batch_size=None):
        """Return sigma for model conditioning: kappa_t / kappa_max expanded to (B,), or None.

        Normalize to [0,1] so the sinusoidal TimestepEmbedder (max_period=10000) is not
        fed raw high-range kappa values (aliasing in top-freq channels).
        """
        if not self.time_conditioning:
            return None
        device = next(self.model.parameters()).device
        if isinstance(kappa_t, (int, float)):
            B = batch_size or 1
            return torch.full((B,), kappa_t / self.kappa_max, device=device)
        kappa_t = kappa_t / self.kappa_max
        if kappa_t.dim() == 0:
            B = batch_size or 1
            return kappa_t.expand(B)
        return kappa_t

    def _geodesic_time_scale(self, kappa_t):
        """1/(1-t) scaling for geodesic velocity. kappa_t IS t (kappa_max=1)."""
        if self.noise_process != "geodesic":
            return 1.0
        if isinstance(kappa_t, Tensor):
            return (1.0 / (1.0 - kappa_t).clamp(min=1e-3)).unsqueeze(0).unsqueeze(-1)
        return 1.0 / max(1.0 - kappa_t, 1e-3)

    def _get_kappa_max_value(self, device: torch.device):
        """Get the effective kappa_max (scalar or per-position).

        Returns:
            kappa_max: scalar float, scalar tensor, or (L,) tensor
        """
        if self.warp is not None:
            return self.warp.get_kappa_max()  # scalar or (L,)
        return self.kappa_max

    def _sphere_step(self, h: Tensor, displacement: Tensor) -> Tensor:
        """Step on sphere from tangent displacement: retraction (default) or exponential map.

        Args:
            h: (B, L, d) current point on sphere (unit norm)
            displacement: (B, L, d) tangent vector (dk*v, or drift+noise for SDE, or eps*score+noise for PC)
        """
        if self.use_expmap:
            speed = displacement.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            return torch.cos(speed) * h + torch.sin(speed) * (displacement / speed)
        return F.normalize(h + displacement, dim=-1)

    def _softmax_step(self, h: Tensor, kappa_t: float, dk: float,
                      clue_mask: Optional[Tensor] = None,
                      W_E: Optional[Tensor] = None) -> Tensor:
        """Euler step using exact vMF velocity on the sphere."""
        h_prime = self.model(h, clue_mask=clue_mask, sigma=self._sigma(kappa_t, h.shape[0]))
        if W_E is None:
            W_E = self.model.get_W_E()

        logits = self.model.compute_logits(h_prime, W_E=W_E)
        probs = F.softmax(logits, dim=-1)

        if self.noise_process == "geodesic":
            v_target = self.path.slerp_velocity_target(h, W_E, probs)
        else:
            v_target = self.path.vmf_velocity_target(h, W_E, probs, self.psi_table, kappa_t)

        v_target = v_target * self._geodesic_time_scale(kappa_t)
        return self._sphere_step(h, dk * v_target)

    def _velocity_head_step(self, h: Tensor, kappa_t: float, dk: float,
                            clue_mask: Optional[Tensor] = None,
                            W_E: Optional[Tensor] = None) -> Tensor:
        """Euler step using learned velocity head on the sphere."""
        h_prime, D_hat = self.model(h, clue_mask=clue_mask, sigma=self._sigma(kappa_t), return_backbone=True)

        D_hat_tan = self.path.proj_tan(D_hat, h)
        D_hat_tan = D_hat_tan * self._geodesic_time_scale(kappa_t)

        return self._sphere_step(h, dk * D_hat_tan)

    def _sde_softmax_step(self, h: Tensor, kappa_t: float, dk: float,
                          clue_mask: Optional[Tensor] = None,
                          W_E: Optional[Tensor] = None) -> Tensor:
        """Reverse SDE Euler-Maruyama step: fused drift (one matmul) + tangent noise."""
        h_prime = self.model(h, clue_mask=clue_mask, sigma=self._sigma(kappa_t, h.shape[0]))
        if W_E is None:
            W_E = self.model.get_W_E()

        logits = self.model.compute_logits(h_prime, W_E=W_E)
        probs = F.softmax(logits, dim=-1)

        # Fused drift = velocity + σ² · score in one matmul
        sde_drift = self.path.vmf_sde_drift(
            h, W_E, probs, self.psi_table, kappa_t, self.sde_sigma ** 2
        )

        # Stochastic noise on tangent plane
        noise = torch.randn_like(h)
        noise_tan = self.path.proj_tan(noise, h)

        displacement = dk * sde_drift + self.sde_sigma * (dk ** 0.5) * noise_tan
        return self._sphere_step(h, displacement)

    def _sde_velocity_head_step(self, h: Tensor, kappa_t: float, dk: float,
                                clue_mask: Optional[Tensor] = None,
                                W_E: Optional[Tensor] = None) -> Tensor:
        """Reverse SDE step using velocity head for drift + exact score for correction."""
        h_prime, D_hat = self.model(h, clue_mask=clue_mask, sigma=self._sigma(kappa_t), return_backbone=True)
        if W_E is None:
            W_E = self.model.get_W_E()

        logits = self.model.compute_logits(h_prime, W_E=W_E)
        probs = F.softmax(logits, dim=-1)

        # Velocity head drift
        D_hat_tan = self.path.proj_tan(D_hat, h)
        D_hat_tan = D_hat_tan * self._geodesic_time_scale(kappa_t)

        # Score correction (score uses the algebraic trick internally)
        score = self.path.score_on_sphere(h, W_E, probs, kappa_t)

        drift = dk * D_hat_tan + self.sde_sigma ** 2 * dk * score

        noise = torch.randn_like(h)
        noise_tan = self.path.proj_tan(noise, h)

        displacement = drift + self.sde_sigma * (dk ** 0.5) * noise_tan
        return self._sphere_step(h, displacement)

    def _flow_map_step(self, h: Tensor, kappa_s: float, kappa_t: float,
                       clue_mask: Optional[Tensor] = None,
                       W_E: Optional[Tensor] = None) -> Tensor:
        """Exact CDF transport step (flow map inference).

        Uses model posterior at kappa_s, then transports exactly to kappa_t
        via CDF inversion. More accurate than Euler at large step sizes.
        """
        h_prime = self.model(h, clue_mask=clue_mask, sigma=self._sigma(kappa_s, h.shape[0]))
        if W_E is None:
            W_E = self.model.get_W_E()

        logits = self.model.compute_logits(h_prime, W_E=W_E)
        probs = F.softmax(logits, dim=-1)

        v_target = self.path.few_step_velocity_target(
            h, W_E, probs, kappa_s, kappa_t, self.vmf_cdf_table
        )
        dk = kappa_t - kappa_s if isinstance(kappa_s, (int, float)) else (kappa_t - kappa_s)
        return self.path.exp_map(h, dk * v_target)

    def _langevin_correct(self, h: Tensor, kappa_t: float, epsilon: float,
                          clue_mask: Optional[Tensor] = None,
                          W_E: Optional[Tensor] = None) -> Tensor:
        """One Langevin correction step using score on the sphere.

        h_new = sphere_step(h, ε_eff · score + √(2ε_eff) · noise_tan)
        where score = κ · Σ_k p(k|h) · proj_tan(w_k, h).

        With corrector_scaling: ε_eff = ε · (1 − u)², u = κ/κ_max.
        Shrinks the step toward the clean end (u=1) for uniform numerics.
        """
        h_prime = self.model(h, clue_mask=clue_mask, sigma=self._sigma(kappa_t, h.shape[0]))
        if W_E is None:
            W_E = self.model.get_W_E()

        logits = self.model.compute_logits(h_prime, W_E=W_E)
        probs = F.softmax(logits, dim=-1)

        score = self.path.score_on_sphere(h, W_E, probs, kappa_t)

        if self.corrector_scaling:
            u = kappa_t / self.kappa_max
            one_minus_u_sq = (1.0 - u) ** 2
            eps = epsilon * one_minus_u_sq
        else:
            eps = epsilon

        noise = torch.randn_like(h)
        noise_tan = self.path.proj_tan(noise, h)

        displacement = eps * score + (2.0 * eps) ** 0.5 * noise_tan
        return self._sphere_step(h, displacement)

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        device: torch.device,
        clue_mask: Optional[Tensor] = None,
        clue_values: Optional[Tensor] = None,
        return_intermediates: bool = False,
        verbose: bool = False,
    ) -> Dict:
        """Generate samples via ODE integration on S^{d-1}.

        Source: uniform on sphere (kappa(0) = 0).
        Integration: uniform grid in kappa-space [0, kappa_max].

        For pc_softmax / pc_velocity_head methods, Langevin correction
        steps are applied after the predict step at configured intervals.

        Args:
            num_samples: Number of samples (B)
            device: torch.device
            clue_mask: (B, L) binary mask (1=clue), optional
            clue_values: (B, L) token indices for clue positions, optional
            return_intermediates: Whether to record trajectory snapshots
            verbose: Whether to show progress bar

        Returns:
            Dict with 'tokens', 'logits', 'h_final',
            and optionally 'intermediates'
        """
        L = self.sequence_length
        B = num_samples

        # Initialize: uniform on sphere (kappa=0 -> uniform vMF)
        h = F.normalize(torch.randn(B, L, self.embed_dim, device=device), dim=-1)

        # Prepare clue embeddings for pinning
        W_E = self.model.get_W_E()
        clue_embeddings = None
        if clue_mask is not None and clue_values is not None:
            clue_embeddings = self.path.get_target_embeddings(W_E, clue_values)

        S = self.num_steps
        use_warp_steps = (self.warp_aware_sampling and self.warp is not None)

        # Determine kappa_final for final readout
        kappa_final = self._get_kappa_max_value(device)

        # Precompute kappa grid: S+1 points from kappa=0 to kappa_max
        if use_warp_steps:
            # Uniform in u-space: evaluate warp CDF at S+1 points.
            # Clamp top end to 1-eps to avoid the steep cliff at u=1 where
            # the warp has to reach kappa_max (would cause a huge final Euler step).
            eps = 1e-3
            u_grid = torch.linspace(0.0, 1.0 - eps, S + 1, device=device)
            kappa_grid, _ = self.warp(u_grid)  # (S+1,) or (S+1, L)
        else:
            # Uniform in kappa-space
            if isinstance(kappa_final, Tensor) and kappa_final.dim() >= 1:
                kappa_max_scalar = kappa_final.max().item()
            elif isinstance(kappa_final, Tensor):
                kappa_max_scalar = kappa_final.item()
            else:
                kappa_max_scalar = float(kappa_final)
            kappa_grid = torch.linspace(0.0, kappa_max_scalar, S + 1, device=device)

        is_pc = self.method in ("pc_softmax", "pc_velocity_head",
                                 "pc_sde_softmax", "pc_sde_velocity_head")
        is_sde = self.method in ("sde_softmax", "sde_velocity_head",
                                  "pc_sde_softmax", "pc_sde_velocity_head")

        intermediates = []

        steps_iter = range(S)
        if verbose:
            steps_iter = tqdm(steps_iter, desc="Sampling", leave=False)

        for step_idx in steps_iter:
            kappa_t = kappa_grid[step_idx]
            dk = kappa_grid[step_idx + 1] - kappa_grid[step_idx]

            # Reshape for broadcast with h: (B, L, D) when per-position
            if dk.dim() >= 1:
                dk = dk.unsqueeze(-1)       # (L, 1)
                # kappa_t stays (L,) — broadcast in compute_logits
            else:
                kappa_t = kappa_t.item()
                dk = dk.item()

            # --- Predict ---
            if self.method == "flow_map":
                kappa_next = kappa_grid[step_idx + 1]
                if kappa_next.dim() == 0:
                    kappa_next = kappa_next.item()
                h = self._flow_map_step(h, kappa_t, kappa_next, clue_mask, W_E=W_E)
            elif self.method in ("softmax", "pc_softmax"):
                h = self._softmax_step(h, kappa_t, dk, clue_mask, W_E=W_E)
            elif self.method in ("velocity_head", "pc_velocity_head"):
                h = self._velocity_head_step(h, kappa_t, dk, clue_mask, W_E=W_E)
            elif self.method in ("sde_softmax", "pc_sde_softmax"):
                h = self._sde_softmax_step(h, kappa_t, dk, clue_mask, W_E=W_E)
            elif self.method in ("sde_velocity_head", "pc_sde_velocity_head"):
                h = self._sde_velocity_head_step(h, kappa_t, dk, clue_mask, W_E=W_E)

            # --- Correct (Langevin with score, only for PC methods) ---
            if is_pc and self.corrector_steps > 0:
                if (step_idx + 1) % self.corrector_interval == 0:
                    kappa_after = kappa_grid[step_idx + 1]
                    if kappa_after.dim() == 0:
                        kappa_after = kappa_after.item()
                    for _ in range(self.corrector_steps):
                        h = self._langevin_correct(
                            h, kappa_after, self.corrector_epsilon, clue_mask,
                            W_E=W_E
                        )

            # Pin clue positions to clean embeddings on sphere
            if clue_mask is not None and clue_embeddings is not None:
                h = torch.where(
                    clue_mask.unsqueeze(-1).expand_as(h), clue_embeddings, h
                )

            if return_intermediates:
                kappa_next = kappa_grid[step_idx + 1]
                kappa_snap = kappa_next.mean().item() if kappa_next.dim() >= 1 else kappa_next.item()
                # Snapshot logits at the current state h (one fwd pass, with proper
                # clue_mask + sigma so the model sees the same context as the sampler).
                h_prime_snap = self.model(h, clue_mask=clue_mask,
                                          sigma=self._sigma(kappa_snap, h.shape[0]))
                logits_snap = self.model.compute_logits(h_prime_snap, W_E=W_E)
                intermediates.append({
                    'h_t': h.detach().clone(),
                    'logits': logits_snap.detach().clone(),
                    'kappa': kappa_snap,
                    'step': step_idx + 1,
                    'time': (step_idx + 1) / S,
                })

        # Final readout — condition on kappa_max (the end of the schedule)
        kappa_last = kappa_grid[-1]
        kappa_scalar = kappa_last.item() if isinstance(kappa_last, Tensor) and kappa_last.dim() == 0 else kappa_last
        h_prime = self.model(h, clue_mask=clue_mask, sigma=self._sigma(kappa_scalar, h.shape[0]))
        logits = self.model.compute_logits(h_prime, W_E=W_E)
        if self.blank_token_id >= 0:
            logits[:, :, self.blank_token_id] = float('-inf')
        tokens = logits.argmax(dim=-1)  # (B, L)

        # Restore clue tokens
        if clue_mask is not None and clue_values is not None:
            tokens = torch.where(clue_mask, clue_values, tokens)

        result = {
            'tokens': tokens,
            'logits': logits,
            'h_final': h.detach(),
        }

        if return_intermediates:
            result['intermediates'] = intermediates

        return result

    @torch.no_grad()
    def sample_batch_from_loader(
        self,
        val_loader,
        num_samples: int,
        device: torch.device,
        return_intermediates: bool = False,
        verbose: bool = False,
    ) -> Dict:
        """Sample using partial puzzles from a validation dataloader.

        Args:
            val_loader: DataLoader yielding dicts with 'input' and 'label'
            num_samples: Number of samples to generate
            device: torch.device
            return_intermediates: Record trajectory snapshots
            verbose: Show progress

        Returns:
            Dict with 'tokens', 'logits', 'h_final', 'ground_truth',
            'clue_mask', 'input_puzzles', and optionally 'intermediates'
        """
        all_inputs = []
        all_labels = []
        val_iter = iter(val_loader)

        collected = 0
        while collected < num_samples:
            try:
                batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                batch = next(val_iter)

            if isinstance(batch, dict):
                all_inputs.append(batch['input'])
                all_labels.append(batch['label'])
                collected += batch['input'].shape[0]

        input_puzzles = torch.cat(all_inputs, dim=0)[:num_samples].to(device)
        ground_truth = torch.cat(all_labels, dim=0)[:num_samples].to(device)

        clue_mask = (input_puzzles != 0)
        clue_values = input_puzzles.clone()

        result = self.sample(
            num_samples=num_samples,
            device=device,
            clue_mask=clue_mask,
            clue_values=clue_values,
            return_intermediates=return_intermediates,
            verbose=verbose,
        )

        result['ground_truth'] = ground_truth
        result['clue_mask'] = clue_mask
        result['input_puzzles'] = input_puzzles

        return result