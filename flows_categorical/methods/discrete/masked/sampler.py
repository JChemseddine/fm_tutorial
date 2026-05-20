# Copied from https://github.com/JChemseddine/spherical (paper-release-anon)
# Paper: Spherical Flows for Sampling Discrete Distributions (arXiv:2605.05629)

"""Sampler for Discrete Flow Matching (CTMC transition sampling).
Based on MDLM (Sahoo et al., 2024): https://github.com/kuleshov-group/mdlm

At each step, for each masked position, sample from the CTMC reverse
transition kernel q(x_s | x_t, x_0):
  - With probability (move_chance_t - move_chance_s) * p_x0[v], unmask to token v
  - With probability move_chance_s, stay masked

Masking rate follows a power schedule: mask_rate(t) = t^p where p is
config.flow.mask_schedule_power. Integration goes from t=1 (fully masked)
to t=eps (nearly unmasked).
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional
from tqdm import tqdm


def _sample_categorical(categorical_probs: Tensor) -> Tensor:
    """Sample from categorical distribution using Gumbel-max trick."""
    gumbel_norm = (
        1e-10
        - (torch.rand_like(categorical_probs) + 1e-10).log()
    )
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


class MaskedFlowSampler:
    """CTMC transition sampler for Discrete Flow Matching."""

    def __init__(self, model: torch.nn.Module, config: 'Config', **kwargs):
        self.model = model
        self.mask_token_id = model.mask_token_id
        self.vocab_size = model.vocab_size
        self.sequence_length = model.sequence_length
        self.num_steps = config.flow.sampling_steps
        self.blank_token_id = config.flow.blank_token_id
        self.mask_schedule_power = getattr(config.flow, 'mask_schedule_power', 1.0)

    def _masking_rate(self, t: Tensor) -> Tensor:
        """Masking rate at time t: t^p."""
        return t ** self.mask_schedule_power

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
        """Generate samples via CTMC reverse transition sampling.

        Start fully masked at t=1, step to t=eps with uniform steps in t.
        At each step, for masked positions, sample from q(x_s | x_t, x_0).
        """
        B = num_samples
        L = self.sequence_length
        S = self.num_steps

        # Start fully masked
        x = torch.full((B, L), self.mask_token_id, device=device, dtype=torch.long)

        # Pin clue positions
        if clue_mask is not None and clue_values is not None:
            x[clue_mask] = clue_values[clue_mask]

        intermediates = []

        # Time steps: uniform from t=1 to t~0
        eps = 1e-5
        dt = (1.0 - eps) / S
        ts = torch.linspace(1.0, eps + dt, S, device=device)  # t values at each step

        steps_iter = range(S)
        if verbose:
            steps_iter = tqdm(steps_iter, desc="Sampling (masked CTMC)", leave=False)

        for step_idx in steps_iter:
            t = ts[step_idx]

            # Forward pass: get model predictions
            logits = self.model(x, clue_mask=clue_mask)  # (B, L, V)

            if self.blank_token_id >= 0:
                logits[:, :, self.blank_token_id] = float('-inf')

            # p(x_0 | x_t) from model
            p_x0 = F.softmax(logits, dim=-1)  # (B, L, V)

            # Masking rates at current and next time
            t_tensor = torch.full((B, 1, 1), t.item(), device=device)
            s_tensor = torch.full((B, 1, 1), (t - dt).clamp(min=0).item(), device=device)

            move_chance_t = self._masking_rate(t_tensor)  # (B, 1, 1)
            move_chance_s = self._masking_rate(s_tensor)  # (B, 1, 1)

            # q(x_s | x_t, x_0): transition probabilities
            # For each token v: prob of unmasking to v = p_x0[v] * (move_chance_t - move_chance_s)
            # Prob of staying masked = move_chance_s
            # Note: mask_token_id = vocab_size, so we need V+1 columns
            q_xs = torch.zeros(B, L, self.vocab_size + 1, device=device)
            q_xs[:, :, :self.vocab_size] = p_x0 * (move_chance_t - move_chance_s)
            q_xs[:, :, self.mask_token_id] = move_chance_s.squeeze(-1)  # (B, L)

            # Sample from transition distribution
            x_next = _sample_categorical(q_xs)  # (B, L)

            # Keep already-unmasked positions unchanged
            copy_flag = (x != self.mask_token_id).to(x.dtype)
            x = (copy_flag * x + (1 - copy_flag) * x_next).long()

            # Re-pin clue positions
            if clue_mask is not None and clue_values is not None:
                x[clue_mask] = clue_values[clue_mask]

            if return_intermediates:
                intermediates.append({
                    'x_t': x.detach().clone(),
                    'step': step_idx + 1,
                    'time': (t - dt).clamp(min=0).item(),
                    'mask_rate': (x == self.mask_token_id).float().mean().item(),
                })

        # Final readout: one more model call, take argmax for any remaining masked positions
        logits = self.model(x, clue_mask=clue_mask)
        if self.blank_token_id >= 0:
            logits[:, :, self.blank_token_id] = float('-inf')

        # For any positions still masked, use argmax from final logits
        still_masked = (x == self.mask_token_id)
        if still_masked.any():
            final_tokens = logits.argmax(dim=-1)
            x = torch.where(still_masked, final_tokens, x)

        if clue_mask is not None and clue_values is not None:
            x = torch.where(clue_mask, clue_values, x)

        result = {
            'tokens': x,
            'logits': logits,
            'h_final': None,  # No continuous state for masked method
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