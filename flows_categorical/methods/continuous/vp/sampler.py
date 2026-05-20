# Copied from https://github.com/JChemseddine/spherical (paper-release-anon)
# Paper: Spherical Flows for Sampling Categorical Data (arXiv:2605.05629)

"""ODE sampler for VP-style linear-interpolation flow in Euclidean embedding space.

Softmax velocity: v = (D - h) / (1-t), D = W_E @ softmax(logits).
Integration in t-space from t=0 (noise) to t=1 (clean).
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional, TYPE_CHECKING
from tqdm import tqdm

if TYPE_CHECKING:
    from flows_categorical.schedule.cdcd_warp import CDCDWarp


class VPODESampler:
    """ODE sampler in R^r for linear flow matching."""

    def __init__(self, model: torch.nn.Module, config: 'Config',
                 warp: Optional['CDCDWarp'] = None, **kwargs):
        self.model = model
        self.embed_dim = model.embed_dim
        self.sequence_length = model.sequence_length
        self.num_steps = config.flow.sampling_steps
        self.method = config.flow.sampler_method
        self.warp = warp
        self.warp_aware_sampling = config.flow.warp_aware_sampling
        self.blank_token_id = config.flow.blank_token_id
        self.time_conditioning = getattr(config.model, 'time_conditioning', False)

        # Corrector config (for pc_softmax)
        self.corrector_steps = config.flow.corrector_steps
        self.corrector_interval = config.flow.corrector_interval
        self.corrector_epsilon = config.flow.corrector_epsilon
        self.corrector_scaling = getattr(
            config.flow, 'corrector_scaling',
            getattr(config.flow, 'corrector_sigma_scaling', True))
        # Singular-endpoint floor (analogous to VE's sigma_min): avoid t=1 where
        # velocity (D−h)/(1−t) and score (t·D−h)/(1−t)² diverge.
        self.t_eps = getattr(config.flow, 't_eps', 1e-3)

    def _langevin_correct_fm(self, h: Tensor, t_i, W_E: Tensor,
                             clue_mask: Optional[Tensor] = None,
                             clue_embeddings: Optional[Tensor] = None) -> Tensor:
        """One Langevin correction step in R^d for linear flow matching.

        Marginal: p_t(h) = Σ_k π_k · N(h; t·w_k, (1-t)²·I).
        Score ∇_h log p_t(h) = (t·D − h) / (1 − t)² where D = W_E @ softmax(logits).

        With corrector_scaling: ε_eff = ε · (1 − t)², so per-step update
        ε_eff · score = ε · (t·D − h) stays bounded as t → 1 (clean end).
        """
        B = h.shape[0]
        if clue_mask is not None and clue_embeddings is not None:
            h_input = torch.where(clue_mask.unsqueeze(-1).expand_as(h),
                                  clue_embeddings, h)
        else:
            h_input = h

        if self.time_conditioning:
            t_input = t_i.expand(B) if t_i.dim() == 0 else t_i
        else:
            t_input = None
        h_prime = self.model(h_input, clue_mask=clue_mask, sigma=t_input)
        logits = self.model.compute_logits(h_prime, W_E=W_E)
        probs = F.softmax(logits, dim=-1)
        D = torch.matmul(probs, W_E.T)

        # score = (t·D − h) / (1 − t)²
        if t_i.dim() == 0:
            t_val = t_i.item()
            one_minus_t = max(1.0 - t_val, 1e-4)
            one_minus_t_sq = one_minus_t ** 2
            score = (t_val * D - h) / one_minus_t_sq
        else:
            one_minus_t = (1.0 - t_i).clamp(min=1e-4).unsqueeze(0).unsqueeze(-1)
            one_minus_t_sq = one_minus_t.pow(2)
            score = (t_i.unsqueeze(0).unsqueeze(-1) * D - h) / one_minus_t_sq

        if self.corrector_scaling:
            eps = self.corrector_epsilon * one_minus_t_sq
        else:
            eps = self.corrector_epsilon

        noise = torch.randn_like(h)
        h_new = h + eps * score + (2.0 * eps) ** 0.5 * noise

        if clue_mask is not None and clue_embeddings is not None:
            h_new = torch.where(clue_mask.unsqueeze(-1).expand_as(h_new),
                                clue_embeddings, h_new)
        return h_new

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
        """Generate samples via ODE integration in R^r.

        Integration from t=0 (noise) to t~1 (clean), signal-increasing.
        """
        L = self.sequence_length
        B = num_samples
        r = self.embed_dim
        S = self.num_steps

        # Initialize: h ~ N(0, I) (at t=0, h = (1-0)*z + 0*w = z)
        h = torch.randn(B, L, r, device=device)

        # Prepare clue embeddings
        W_E = self.model.get_W_E()
        clue_embeddings = None
        if clue_mask is not None and clue_values is not None:
            clue_embeddings = W_E[:, clue_values].permute(1, 2, 0)

        # Build t grid: from 0 to 1 - t_eps (signal-increasing)
        eps = self.t_eps
        use_warp_steps = (self.warp_aware_sampling and self.warp is not None)
        if use_warp_steps:
            # Warp outputs t directly (signal-increasing, 0->1). No flip needed.
            u_grid = torch.linspace(0.0, 1.0, S + 1, device=device)
            t_grid, _ = self.warp(u_grid)  # (S+1,), 0->1 signal-increasing
            t_grid = t_grid.clamp(max=1.0 - eps)
        else:
            t_grid = torch.linspace(0.0, 1.0 - eps, S + 1, device=device)

        intermediates = []

        steps_iter = range(S)
        if verbose:
            steps_iter = tqdm(steps_iter, desc="Sampling (FM)", leave=False)

        for step_idx in steps_iter:
            t_i = t_grid[step_idx]
            t_next = t_grid[step_idx + 1]
            dt = t_next - t_i  # positive (moving toward clean)

            # Forward pass (no input preconditioning for FM)
            if clue_mask is not None and clue_embeddings is not None:
                h_input = torch.where(
                    clue_mask.unsqueeze(-1).expand_as(h),
                    clue_embeddings, h)
            else:
                h_input = h

            if self.time_conditioning:
                t_input = t_i.expand(B) if t_i.dim() == 0 else t_i
            else:
                t_input = None
            h_prime = self.model(h_input, clue_mask=clue_mask, sigma=t_input)
            logits = self.model.compute_logits(h_prime, W_E=W_E)
            probs = F.softmax(logits, dim=-1)  # (B, L, V)

            # Denoiser: D = W_E @ probs (posterior mean embedding)
            D = torch.matmul(probs, W_E.T)  # (B, L, r)

            # Velocity: v = (D - h) / (1 - t)
            denom = max((1.0 - t_i).item(), 1e-4)
            v = (D - h) / denom

            # Euler step
            h = h + dt * v

            # Langevin correction (pc_softmax only)
            is_pc = self.method in ("pc_softmax", "pc_softmax_warp")
            if is_pc and self.corrector_steps > 0 and (step_idx + 1) % self.corrector_interval == 0:
                for _ in range(self.corrector_steps):
                    h = self._langevin_correct_fm(h, t_next, W_E, clue_mask, clue_embeddings)

            # Pin clue positions
            if clue_mask is not None and clue_embeddings is not None:
                h = torch.where(
                    clue_mask.unsqueeze(-1).expand_as(h),
                    clue_embeddings, h)

            if return_intermediates:
                t_val = t_next.item() if t_next.dim() == 0 else t_next.mean().item()
                intermediates.append({
                    'h_t': h.detach().clone(),
                    't': t_val,
                    'step': step_idx + 1,
                    'time': (step_idx + 1) / S,
                })

        # Final readout at t~1
        t_final = torch.ones(B, device=device) if self.time_conditioning else None
        h_prime = self.model(h, clue_mask=clue_mask, sigma=t_final)
        logits = self.model.compute_logits(h_prime, W_E=W_E)
        if self.blank_token_id >= 0:
            logits[:, :, self.blank_token_id] = float('-inf')
        tokens = logits.argmax(dim=-1)

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
        """Sample using partial puzzles from a validation dataloader."""
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