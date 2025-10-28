"""
Utility helpers for the flow-matching notebooks.

This module provides reusable implementations of distributions, networks,
training loops, and visualization functions used throughout the course.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

try:
    import ot  # type: ignore
except ImportError:
    ot = None

try:
    from geomloss import SamplesLoss  # type: ignore
except ImportError:
    SamplesLoss = None

from matplotlib.animation import FuncAnimation
from IPython.display import HTML


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BatchSampler = Callable[[int, torch.device, torch.dtype], torch.Tensor]
LatentSampler = Callable[[Tuple[int, ...]], torch.Tensor]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# TARGET DISTRIBUTIONS
# ============================================================================

class BaseDistribution2D:
    """Minimal interface shared by the toy 2D distributions."""

    has_log_prob: bool = False

    def sample(
        self,
        n: int,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Analytic log-density not available for this distribution.")


class CheckerboardStripes(BaseDistribution2D):
    """Axis-aligned checkerboard with alternating occupied squares."""

    has_log_prob: bool = True

    def __init__(self, low: float = -4.0, high: float = 4.0) -> None:
        self.low = float(low)
        self.high = float(high)

    def _pick_square(self, n: int, device, dtype) -> torch.Tensor:
        low_i = int(math.floor(self.low))
        high_i = int(math.floor(self.high))
        coords = torch.cartesian_prod(
            torch.arange(low_i, high_i, device=device),
            torch.arange(low_i, high_i, device=device),
        )
        mask = (coords.sum(dim=-1) % 2 == 0)
        valid = coords[mask]
        idx = torch.randint(0, valid.shape[0], (n,), device=device)
        return valid[idx].to(dtype)

    def sample(
        self,
        n: int,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        device = torch.device(device) if device is not None else DEVICE
        dtype = dtype or torch.get_default_dtype()
        squares = self._pick_square(n, device, dtype)
        offsets = torch.rand(n, 2, device=device, dtype=dtype)
        return squares + offsets

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        area = (self.high - self.low) ** 2
        log_const = -math.log(area / 2.0)
        inside = (x[..., 0] >= self.low) & (x[..., 0] <= self.high)
        inside &= (x[..., 1] >= self.low) & (x[..., 1] <= self.high)
        squares = torch.floor(x[..., 0]) + torch.floor(x[..., 1])
        inside &= (squares.long() % 2 == 0)
        out = x.new_full(x.shape[:-1], float("-inf"))
        out[inside] = log_const
        return out


class GridGMM9(BaseDistribution2D):
    """Nine-component Gaussian mixture on a 3x3 grid."""

    has_log_prob: bool = True

    def __init__(self, spacing: float = 1.0, var: float = 0.01, weights: Optional[list[float]] = None) -> None:
        coords = (-float(spacing), 0.0, float(spacing))
        self.means = torch.tensor([(x, y) for x in coords for y in coords], dtype=torch.get_default_dtype())
        if weights is None:
            weights = [0.01, 0.1, 0.3, 0.2, 0.02, 0.15, 0.02, 0.15, 0.05]
        w = torch.tensor(weights, dtype=torch.get_default_dtype())
        self.weights = (w / w.sum()).tolist()
        self.var = float(var)
        self._log_weights: Optional[torch.Tensor] = None

    def sample(
        self,
        n: int,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        device = torch.device(device) if device is not None else DEVICE
        dtype = dtype or torch.get_default_dtype()
        cat = torch.distributions.Categorical(probs=torch.tensor(self.weights, device=device, dtype=dtype))
        idx = cat.sample((n,))
        means = self.means.to(device=device, dtype=dtype)
        noise = torch.randn(n, 2, device=device, dtype=dtype) * math.sqrt(self.var)
        return means[idx] + noise

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        if self._log_weights is None or self._log_weights.device != x.device:
            self._log_weights = torch.log(torch.tensor(self.weights, device=x.device, dtype=x.dtype))
        means = self.means.to(device=x.device, dtype=x.dtype)
        diff = x[:, None, :] - means[None, :, :]
        quad = (diff ** 2).sum(dim=-1) / self.var
        log_comp = -0.5 * (quad + 2 * math.log(2 * math.pi * self.var))
        return torch.logsumexp(self._log_weights + log_comp, dim=-1)


class TwoMoons(BaseDistribution2D):
    """2D two moons distribution - sample only (no analytic log-prob)."""

    has_log_prob: bool = False

    def __init__(self, noise: float = 0.05) -> None:
        self.noise = float(noise)

    def sample(
        self,
        n: int,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        device = torch.device(device) if device is not None else DEVICE
        dtype = dtype or torch.get_default_dtype()
        n1 = n // 2
        n2 = n - n1
        # Moon 1: centered at (0,0), radius 1, theta ∈ [0,π]
        theta1 = math.pi * torch.rand(n1, device=device, dtype=dtype)
        x1 = torch.stack([torch.cos(theta1), torch.sin(theta1)], dim=-1)
        # Moon 2: shift/flip to interleave
        theta2 = math.pi * torch.rand(n2, device=device, dtype=dtype)
        x2 = torch.stack([1.0 - torch.cos(theta2), 1.0 - torch.sin(theta2) - 0.5], dim=-1)
        x = torch.cat([x1, x2], dim=0)
        if self.noise > 0:
            x = x + self.noise * torch.randn_like(x)
        return x


class NealFunnel2D(BaseDistribution2D):
    """Neal's funnel distribution with analytic log-density."""

    has_log_prob: bool = True

    def __init__(self, sigma1: float = 3.0, alpha: float = 1.0) -> None:
        self.sigma1 = float(sigma1)
        self.alpha = float(alpha)

    def sample(
        self,
        n: int,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        device = torch.device(device) if device is not None else DEVICE
        dtype = dtype or torch.get_default_dtype()
        x1 = self.sigma1 * torch.randn(n, 1, device=device, dtype=dtype)
        std2 = torch.exp(0.5 * self.alpha * x1)
        x2 = std2 * torch.randn(n, 1, device=device, dtype=dtype)
        return torch.cat([x1, x2], dim=-1)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., 0], x[..., 1]
        var2 = torch.exp(self.alpha * x1)
        term1 = -0.5 * ((x1 / self.sigma1) ** 2 + math.log(2 * math.pi * self.sigma1 ** 2))
        term2 = -0.5 * (x2 ** 2 / var2 + math.log(2 * math.pi) + self.alpha * x1)
        return term1 + term2


class ZScoreWrapper(BaseDistribution2D):
    """Wrap a base sampler to operate in z-scored coordinates for stability."""

    has_log_prob: bool = True

    def __init__(self, base: NealFunnel2D) -> None:
        self.base = base
        self.mean = torch.tensor([0.0, 0.0])
        self.std = torch.tensor(
            [
                base.sigma1,
                math.exp(0.25 * (base.alpha ** 2) * (base.sigma1 ** 2)),
            ]
        )

    def sample(self, n: int, *, device=None, dtype=None) -> torch.Tensor:
        raw = self.base.sample(n, device=device, dtype=dtype)
        mean = self.mean.to(device=raw.device, dtype=raw.dtype)
        std = self.std.to(device=raw.device, dtype=raw.dtype)
        return (raw - mean) / std

    def to_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from z-scored coordinates back to raw funnel coordinates."""
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return x * std + mean

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        raw = x * std + mean
        base_logp = self.base.log_prob(raw)
        log_det = torch.log(std.abs()).sum()
        return base_logp - log_det


def get_distribution(name: str, **kwargs) -> BaseDistribution2D:
    """Instantiate one of the benchmark 2D distributions."""
    name_l = name.lower()
    if name_l in {"checker", "checkerboard"}:
        return CheckerboardStripes(**kwargs)
    if name_l in {"gridgmm", "gridgmm9"}:
        return GridGMM9(**kwargs)
    if name_l in {"twomoons", "two_moons"}:
        return TwoMoons(**kwargs)
    if name_l in {"funnel", "nealfunnel"}:
        return ZScoreWrapper(NealFunnel2D(**kwargs))
    raise ValueError(f"Unknown distribution '{name}'.")


# ============================================================================
# VELOCITY NETWORK
# ============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Fourier-style embedding for scalar time inputs."""

    def __init__(self, embed_dim: int, max_freq: float = 1000.0) -> None:
        super().__init__()
        half = embed_dim // 2
        freq = torch.logspace(0, math.log10(max_freq), steps=half)
        self.register_buffer("freq", freq)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1).float()
        angles = t * self.freq.view(1, -1)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class VelocityMLP(nn.Module):
    """Time-conditioned MLP that predicts the velocity field."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        time_embed_dim: int = 64,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        # Build simple feedforward network
        layers = []
        layers.append(nn.Linear(input_dim + time_embed_dim, hidden_dim))
        layers.append(nn.SiLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(hidden_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_emb = self.time_embed(t)
        if t_emb.shape[0] == 1 and x.shape[0] > 1:
            t_emb = t_emb.expand(x.shape[0], -1)

        h = torch.cat([x, t_emb], dim=-1)
        return self.net(h)


# ============================================================================
# SAMPLING AND PLOTTING UTILITIES
# ============================================================================

def draw_samples(distribution: BaseDistribution2D, n: int = 4096) -> torch.Tensor:
    """Sample a batch from the target distribution and detach to CPU."""
    samples = distribution.sample(n, device=DEVICE, dtype=torch.float32)
    return samples.detach().cpu()


def make_batch_sampler(sampler_like) -> BatchSampler:
    """Adapt a distribution or callable into a batch sampler returning fresh draws."""

    def sample(n: int, *, device: torch.device = DEVICE, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        target_device = torch.device(device)
        sample_attr = getattr(sampler_like, "sample", None)
        if sample_attr is None:
            try:
                result = sampler_like(n, device=target_device, dtype=dtype)
            except TypeError:
                result = sampler_like(n)
        else:
            try:
                result = sample_attr(n, device=target_device, dtype=dtype)
            except TypeError:
                try:
                    result = sample_attr((n,))
                except TypeError:
                    result = sample_attr(n)
        if not isinstance(result, torch.Tensor):
            result = torch.as_tensor(result, dtype=dtype)
        return result.to(device=target_device, dtype=dtype)

    return sample


def _to_numpy(array: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert tensor or array to numpy array."""
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return array


def _compute_limits(
    arrays: tuple[np.ndarray, ...],
    margin: float = 0.02,
    min_extent: float = 0.1,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute axis limits with padding for multiple arrays."""
    data = np.concatenate(arrays, axis=0)
    x_min, y_min = data.min(axis=0)
    x_max, y_max = data.max(axis=0)
    range_x = max(x_max - x_min, min_extent)
    range_y = max(y_max - y_min, min_extent)
    pad_x = margin * range_x
    pad_y = margin * range_y
    return (x_min - pad_x, x_max + pad_x), (y_min - pad_y, y_max + pad_y)


def plot_points(
    ax,
    samples: torch.Tensor | np.ndarray,
    title: str,
    color: str,
    *,
    alpha: float = 0.6,
    limits: Optional[tuple[tuple[float, float], tuple[float, float]]] = None,
) -> None:
    """Scatter plot helper with consistent styling."""
    pts = _to_numpy(samples)
    ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=alpha, color=color, linewidths=0)
    ax.set_title(title)
    if limits is None:
        limits = _compute_limits((pts,))
    (x_min, x_max), (y_min, y_max) = limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', 'box')
 
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))


def plot_source_target_trajectories(
    source: torch.Tensor,
    trajectories: torch.Tensor,
    *,
    ax=None,
    target: Optional[torch.Tensor] = None,
    background: Optional[torch.Tensor] = None,
    title: Optional[str] = None,
    max_paths: int = 128,
    latent_color: str = "#000000",
    endpoint_color: str = "#1f77b4",
    path_color: str = "#90EE90",
) -> plt.Axes:
    """Overlay latent samples, trajectories, and endpoints (with optional background).

    Args:
        source: Latent samples at t=1 (N, 2)
        trajectories: Full trajectory (T, N, 2)
        ax: Matplotlib axis to plot on
        target: Optional target samples to show
        background: Optional background samples
        title: Plot title
        max_paths: Maximum number of paths to draw
        latent_color: Color for latent points (default: black)
        endpoint_color: Color for trajectory endpoints (default: blue)
        path_color: Color for trajectory paths (default: light green)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    src = _to_numpy(source)
    traj = _to_numpy(trajectories)
    if traj.ndim != 3:
        raise ValueError("Trajectories tensor must have shape (T, N, 2).")
    finals = traj[-1]

    to_concat = [src, finals]
    if background is not None:
        bg = _to_numpy(background)
        to_concat.append(bg)
    if target is not None:
        to_concat.append(_to_numpy(target))

    limits = _compute_limits(tuple(to_concat))

    ax.scatter(src[:, 0], src[:, 1], s=10, alpha=0.45, color=latent_color, label="latents $z$")

    if background is not None:
        ax.scatter(bg[:, 0], bg[:, 1], s=8, alpha=0.15, color="#b0bec5", label="target samples")

    if target is None:
        target = finals
    tgt = _to_numpy(target)
    ax.scatter(tgt[:, 0], tgt[:, 1], s=16, alpha=0.7, color=endpoint_color, label="traj endpoint")

    n_paths = min(max_paths, traj.shape[1])
    for idx in range(n_paths):
        path = traj[:, idx, :]
        ax.plot(path[:, 0], path[:, 1], color=path_color, alpha=0.35, linewidth=1.0)

    (x_min, x_max), (y_min, y_max) = limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", "box")
 
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, loc="upper right")
    return ax



def plot_transport_lines(
    source: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    assignment: np.ndarray,
    *,
    ax=None,
    title: Optional[str] = None,
    source_color: str = "#9467bd",
    target_color: str = "#1f77b4",
    line_color: str = "#ff7f0e",
    alpha: float = 0.35,
) -> plt.Axes:
    """Plot straight-line transport couplings between source and target samples."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    src = _to_numpy(source)
    tgt = _to_numpy(target)
    ax.scatter(src[:, 0], src[:, 1], s=12, alpha=0.6, color=source_color, label="source")
    ax.scatter(tgt[:, 0], tgt[:, 1], s=12, alpha=0.6, color=target_color, label="target")

    for i, j in enumerate(assignment):
        xs = [src[i, 0], tgt[int(j), 0]]
        ys = [src[i, 1], tgt[int(j), 1]]
        ax.plot(xs, ys, color=line_color, alpha=alpha, linewidth=0.8)

    limits = _compute_limits((src, tgt))
    (x_min, x_max), (y_min, y_max) = limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, loc="upper right")
    return ax


def plot_velocity_field(
    model: nn.Module,
    t: float,
    xlim: tuple[float, float] = (-4, 4),
    ylim: tuple[float, float] = (-4, 4),
    n_grid: int = 20,
    ax=None,
    title: Optional[str] = None,
    scale: float = 1.0,
) -> plt.Axes:
    """Visualize the learned velocity field at a specific time t using a quiver plot."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    
    # Create grid
    x = np.linspace(*xlim, n_grid)
    y = np.linspace(*ylim, n_grid)
    X, Y = np.meshgrid(x, y)
    positions = np.stack([X.ravel(), Y.ravel()], axis=-1)
    
    # Compute velocities
    pos_tensor = torch.from_numpy(positions).float().to(DEVICE)
    t_tensor = torch.full((pos_tensor.shape[0], 1), t, device=DEVICE)
    
    with torch.no_grad():
        velocities = model(t_tensor, pos_tensor).cpu().numpy()
    
    U = velocities[:, 0].reshape(n_grid, n_grid)
    V = velocities[:, 1].reshape(n_grid, n_grid)
    
    # Plot
    ax.quiver(X, Y, U, V, alpha=0.7, scale=scale, scale_units='xy')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Velocity Field at t={t:.2f}')
    
    return ax


def animate_flow(
    model: nn.Module,
    latents: torch.Tensor,
    flow_T: float,
    steps: int = 100,
    interval: int = 50,
    xlim: tuple[float, float] = (-5, 5),
    ylim: tuple[float, float] = (-5, 5),
) -> HTML:
    """Create an animation of flow trajectories over time.

    Args:
        model: Trained velocity network
        latents: Initial latent points (N, 2)
        flow_T: Total flow time
        steps: Number of integration steps
        interval: Milliseconds between frames
        xlim, ylim: Plot limits

    Returns:
        HTML animation object for display in Jupyter
    """
    # Generate full trajectories
    with torch.no_grad():
        trajectories = euler_integrate(
            model, latents, flow_T, steps=steps, return_path=True
        )

    traj_np = trajectories.cpu().numpy()  # (T, N, 2)

    # Setup figure
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Initialize scatter plot
    scat = ax.scatter([], [], s=10, alpha=0.6, color='#1f77b4')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)

    def init():
        scat.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        return scat, time_text

    def update(frame):
        # Update positions
        pos = traj_np[frame]
        scat.set_offsets(pos)

        # Update time text
        t = flow_T * (1 - frame / (steps - 1))
        time_text.set_text(f't = {t:.3f}')

        return scat, time_text

    anim = FuncAnimation(
        fig, update, init_func=init,
        frames=steps, interval=interval, blit=True
    )

    plt.close(fig)
    return HTML(anim.to_jshtml())


def _analytic_funnel_x2_pdf(x2_grid: np.ndarray, scale1: float, gh_n: int = 80) -> np.ndarray:
    """Compute the marginal p(x2) via Gauss-Hermite quadrature for Neal's funnel.

    Args:
        x2_grid: Grid of x2 values to evaluate density at
        scale1: Standard deviation parameter (sigma1) of the funnel
        gh_n: Number of Gauss-Hermite quadrature nodes

    Returns:
        Probability densities at x2_grid points
    """
    from numpy.polynomial.hermite import hermgauss

    x2 = x2_grid.astype(np.float64)
    nodes, weights = hermgauss(gh_n)
    x1_vals = (np.sqrt(2.0) * scale1) * nodes
    w_norm = weights / np.sqrt(np.pi)

    var = np.exp(x1_vals)
    inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)
    var_col = var[:, None]
    coef = inv_sqrt_2pi / np.sqrt(var_col)
    expo = np.exp(-(x2[None, :] ** 2) / (2.0 * var_col))
    pdf_matrix = coef * expo
    pdf = (w_norm[:, None] * pdf_matrix).sum(axis=0)
    return np.maximum(pdf, 1e-300)


def _compute_log_joint_grid_funnel(
    sampler,
    x1_range: tuple[float, float],
    x2_range: tuple[float, float],
    n1: int = 300,
    n2: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute log probability on a grid for funnel visualization.

    Args:
        sampler: Distribution sampler with log_prob method (usually the base funnel)
        x1_range: (min, max) for x1 dimension
        x2_range: (min, max) for x2 dimension
        n1: Number of grid points in x1
        n2: Number of grid points in x2

    Returns:
        x1_lin: x1 grid points
        x2_lin: x2 grid points
        logp: Log probabilities on the grid (n1, n2)
    """
    x1_lin = np.linspace(*x1_range, n1)
    x2_lin = np.linspace(*x2_range, n2)
    X2, X1 = np.meshgrid(x2_lin, x1_lin, indexing="xy")
    grid = np.stack([X1.ravel(), X2.ravel()], axis=-1)
    grid_tensor = torch.from_numpy(grid).to(dtype=torch.float64, device=DEVICE)
    with torch.no_grad():
        logp = sampler.log_prob(grid_tensor).cpu().numpy().reshape(n1, n2)
    return x1_lin, x2_lin, logp


def plot_funnel_with_marginals(
    samples: torch.Tensor,
    target_sampler,
    *,
    ax=None,
    title: str = "Generated samples on funnel log-density",
    bins_main: int = 200,
    bins_marginal: int = 120,
):
    """Create funnel plot with marginal distributions and log-density heatmap.

    This advanced plotting function creates a comprehensive visualization of Neal's funnel:
    - Main plot: Log-density heatmap with generated samples overlaid
    - Top marginal: Histogram of x2 (heavy-tailed dimension) with analytic curve
    - Right marginal: Histogram of x1 (light-tailed dimension)

    Args:
        samples: Generated samples to visualize (N, 2)
        target_sampler: Target distribution (ZScoreWrapper around NealFunnel2D)
        ax: Ignored (function creates its own figure with subplots)
        title: Title for the main plot
        bins_main: Number of bins for density heatmap
        bins_marginal: Number of bins for marginal histograms

    Returns:
        Figure object
    """
    if ax is not None:
        raise ValueError("This helper creates its own figure; pass ax=None.")

    def to_raw(tensor: torch.Tensor) -> torch.Tensor:
        """Convert from z-scored back to raw funnel coordinates."""
        if hasattr(target_sampler, "to_raw"):
            return target_sampler.to_raw(tensor)
        return tensor

    base_sampler = target_sampler.base if hasattr(target_sampler, "base") else target_sampler

    samp_tensor = samples if isinstance(samples, torch.Tensor) else torch.from_numpy(samples)
    gen_raw = to_raw(samp_tensor).to(torch.float64)
    true_raw = base_sampler.sample(gen_raw.shape[0]).to(torch.float64)

    X2_MIN, X2_MAX = -999.0, 999.0
    X1_MIN, X1_MAX = -20.0, 20.0

    x1_d = gen_raw[:, 0].cpu().numpy()
    x2_d = gen_raw[:, 1].cpu().numpy()
    x1_m = true_raw[:, 0].cpu().numpy()
    x2_m = true_raw[:, 1].cpu().numpy()

    bins_x1 = np.linspace(X1_MIN, X1_MAX, bins_marginal)
    bins_x2 = np.linspace(X2_MIN, X2_MAX, bins_marginal)

    fig = plt.figure(figsize=(6, 6), dpi=120)
    gs = plt.GridSpec(4, 4, figure=fig, hspace=0.05, wspace=0.05)
    ax_main = fig.add_subplot(gs[1:, :3])
    ax_top = fig.add_subplot(gs[0, :3], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, 3], sharey=ax_main)

    teal = "#7fb8c8"
    red = "#e74c3c"

    # Compute and plot log density
    _, _, logp = _compute_log_joint_grid_funnel(
        base_sampler, (X1_MIN, X1_MAX), (X2_MIN, X2_MAX), n1=320, n2=360
    )
    ax_main.set_facecolor("black")
    cmap = plt.cm.magma.copy()
    cmap.set_under("black")
    log_floor = -20.0
    vmax = float(np.max(logp))
    ax_main.imshow(
        logp,
        origin="lower",
        extent=[X2_MIN, X2_MAX, X1_MIN, X1_MAX],
        aspect="auto",
        cmap=cmap,
        vmin=log_floor,
        vmax=vmax,
    )

    # Scatter generated samples
    ax_main.scatter(x2_d, x1_d, s=6, alpha=0.5, color=teal, linewidths=0, edgecolors="none")
    ax_main.set_xlabel(r"$x_2$", color="white")
    ax_main.set_ylabel(r"$x_1$", color="white")
    ax_main.set_xlim(X2_MIN, X2_MAX)
    ax_main.set_ylim(X1_MIN, X1_MAX)
    ax_main.set_title(title, color="white")

    # Top marginal (x2) with log scale
    ax_top.set_yscale("log")
    ax_top.hist(x2_m, bins=bins_x2, density=True, histtype="step", color=red, linewidth=2.0, label="True")
    ax_top.hist(x2_d, bins=bins_x2, density=True, color=teal, alpha=0.35, edgecolor=teal, label="Generated")
    x2_centers = 0.5 * (bins_x2[:-1] + bins_x2[1:])
    scale1 = float(getattr(base_sampler, "sigma1", torch.tensor(3.0)))
    px2 = _analytic_funnel_x2_pdf(x2_centers, scale1=scale1, gh_n=80)
    ax_top.plot(x2_centers, px2, color="#1f77b4", linewidth=2.2, alpha=0.95, label="Analytic")
    ax_top.tick_params(labelbottom=False)
    ax_top.legend(fontsize=8, loc="upper right")
    ax_top.set_ylabel(r"$p(x_2)$", fontsize=9)

    # Right marginal (x1)
    ax_right.hist(
        x1_d,
        bins=bins_x1,
        density=True,
        orientation="horizontal",
        color=teal,
        alpha=0.35,
        edgecolor=teal,
    )
    ax_right.hist(
        x1_m,
        bins=bins_x1,
        density=True,
        orientation="horizontal",
        histtype="step",
        color=red,
        linewidth=2.0,
    )
    ax_right.tick_params(labelleft=False)
    ax_right.set_xlabel(r"$p(x_1)$", fontsize=9)

    return fig


def plot_latent_colored_by_endpoint_norm(
    latents: torch.Tensor,
    endpoints: torch.Tensor,
    *,
    title: str = "Latent colored by ||x||",
    cmap: str = "viridis",
):
    """Two-panel figure showing latent points colored by endpoint norm.

    This visualization helps understand the transport difficulty:
    - Left panel: Raw latent samples (uncolored)
    - Right panel: Latent samples colored by the norm of their endpoints

    Interpretation:
    - Smooth color gradient → Uniform, easy transport
    - Sharp color changes → Network must stretch/compress heavily

    Args:
        latents: Latent points at t=1 (N, 2)
        endpoints: Corresponding endpoints at t=0 (N, 2)
        title: Title for the right panel
        cmap: Matplotlib colormap name

    Returns:
        Figure object
    """
    L = np.atleast_2d(_to_numpy(latents))
    X = np.atleast_2d(_to_numpy(endpoints))
    if X.shape[0] == 1:
        norms = np.full(L.shape[0], np.linalg.norm(X[0]))
    else:
        norms = np.linalg.norm(X, axis=1)

    radius = float(np.max(np.abs(L))) * 1.1
    if not np.isfinite(radius) or radius == 0.0:
        radius = 1.0
    limits = (-radius, radius)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=120, constrained_layout=True)

    axes[0].scatter(L[:, 0], L[:, 1], s=4, alpha=0.35, color="#888888", linewidths=0)
    axes[0].set_title("Latent samples", fontsize=12)
    axes[0].set_xlabel("$z_1$", fontsize=11)
    axes[0].set_ylabel("$z_2$", fontsize=11)
    axes[0].set_aspect("equal", "box")
    axes[0].set_xlim(*limits)
    axes[0].set_ylim(*limits)
    axes[0].grid(True, alpha=0.2)

    h = axes[1].scatter(L[:, 0], L[:, 1], s=5, c=norms, cmap=cmap, alpha=0.7, linewidths=0)
    axes[1].set_title(title, fontsize=12)
    axes[1].set_xlabel("$z_1$", fontsize=11)
    axes[1].set_ylabel("$z_2$", fontsize=11)
    axes[1].set_aspect("equal", "box")
    axes[1].set_xlim(*limits)
    axes[1].set_ylim(*limits)
    axes[1].grid(True, alpha=0.2)
    cbar = fig.colorbar(h, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("||x||", fontsize=10)

    return fig


def _make_1d_sampler(name: str, device: torch.device, **kwargs) -> Callable[[Tuple[int, ...]], torch.Tensor]:
    """Helper to create a 1D sampler for a single dimension.

    Args:
        name: Distribution name ("gaussian", "uniform", "student_t")
        device: Device to create tensors on
        **kwargs: Distribution-specific parameters

    Returns:
        Sampler function for 1D samples
    """
    name_l = name.lower()

    if name_l in {"gaussian", "normal"}:
        mean = kwargs.get("mean", 0.0)
        std = kwargs.get("std", 1.0)
        def sample(shape: Tuple[int, ...]) -> torch.Tensor:
            return torch.randn(*shape, device=device) * std + mean
        return sample

    elif name_l in {"uniform", "uni"}:
        low = kwargs.get("low", -2.0)
        high = kwargs.get("high", 2.0)
        def sample(shape: Tuple[int, ...]) -> torch.Tensor:
            return torch.rand(*shape, device=device) * (high - low) + low
        return sample

    elif name_l in {"student_t", "studentt"}:
        df = kwargs.get("df", 4.0)
        scale = kwargs.get("scale", 1.0)
        loc = kwargs.get("loc", 0.0)
        def sample(shape: Tuple[int, ...]) -> torch.Tensor:
            # Create distribution with parameters on the correct device
            dist = torch.distributions.StudentT(
                df=torch.tensor(df, device=device),
                loc=torch.tensor(loc, device=device),
                scale=torch.tensor(scale, device=device)
            )
            return dist.sample(shape)
        return sample

    else:
        raise ValueError(f"Unknown distribution '{name}'")


def make_latent_sampler(
    name: str | list[tuple[str, dict]],
    device: torch.device,
    dim: int,
    **kwargs
) -> Callable[[Tuple[int, ...]], torch.Tensor]:
    """Factory for latent distributions with optional per-dimension control.

    This function supports both IID sampling (same distribution across all dimensions)
    and non-IID sampling (different distributions per dimension) for componentwise
    noise adaptation to target geometry.

    Args:
        name: Either:
            - String: IID distribution across all dimensions ("gaussian", "uniform", "student_t")
            - List of (dist_name, params) tuples: Different distribution per dimension
        device: Device to create tensors on
        dim: Dimensionality
        **kwargs: Parameters for IID case only (e.g., df=4.0, scale=1.0)

    Examples:
        # IID Gaussian (both dimensions N(0,1))
        >>> sampler = make_latent_sampler("gaussian", DEVICE, 2)

        # IID Student-t with custom df
        >>> sampler = make_latent_sampler("student_t", DEVICE, 2, df=4.0, scale=1.0)

        # Non-IID: different std per dimension
        >>> sampler = make_latent_sampler([
        ...     ("gaussian", {"std": 1.0}),
        ...     ("gaussian", {"std": 2.0})
        ... ], DEVICE, 2)

        # Non-IID: different distributions per dimension (for funnel)
        >>> sampler = make_latent_sampler([
        ...     ("student_t", {"df": 20, "scale": 1}),  # lighter tails for x1
        ...     ("student_t", {"df": 4, "scale": 1})    # heavier tails for x2
        ... ], DEVICE, 2)

    Returns:
        Sampler function that takes shape (batch_size, dim) and returns samples
    """
    # IID case: same distribution for all dimensions
    if isinstance(name, str):
        name_l = name.lower()

        if name_l in {"gaussian", "normal"}:
            mean = kwargs.get("mean", 0.0)
            std = kwargs.get("std", 1.0)
            def sample(shape: Tuple[int, ...]) -> torch.Tensor:
                return torch.randn(*shape, device=device) * std + mean
            return sample

        elif name_l in {"uniform", "uni"}:
            low = kwargs.get("low", -2.0)
            high = kwargs.get("high", 2.0)
            def sample(shape: Tuple[int, ...]) -> torch.Tensor:
                return torch.rand(*shape, device=device) * (high - low) + low
            return sample

        elif name_l in {"student_t", "studentt"}:
            df = kwargs.get("df", 4.0)
            scale = kwargs.get("scale", 1.0)
            loc = kwargs.get("loc", 0.0)
            def sample(shape: Tuple[int, ...]) -> torch.Tensor:
                # Create distribution with parameters on the correct device
                dist = torch.distributions.StudentT(
                    df=torch.tensor(df, device=device),
                    loc=torch.tensor(loc, device=device),
                    scale=torch.tensor(scale, device=device)
                )
                return dist.sample(shape)
            return sample

        else:
            raise ValueError(f"Unknown latent distribution '{name}'")

    # Non-IID case: different distribution per dimension
    elif isinstance(name, list):
        if len(name) != dim:
            raise ValueError(
                f"Length of name list ({len(name)}) must match dim ({dim}). "
                f"Provide one (dist_name, params) tuple per dimension."
            )

        # Create individual samplers for each dimension
        samplers = []
        for spec in name:
            if isinstance(spec, tuple) and len(spec) == 2:
                dist_name, params = spec
            else:
                raise ValueError(
                    f"Each element must be a (dist_name, params) tuple, got {spec}"
                )

            # Create 1D sampler with specified parameters
            sampler_1d = _make_1d_sampler(dist_name, device, **params)
            samplers.append(sampler_1d)

        def sample(shape: Tuple[int, ...]) -> torch.Tensor:
            if len(shape) != 2 or shape[1] != dim:
                raise ValueError(
                    f"Expected shape (batch_size, {dim}), got {shape}"
                )

            batch_size = shape[0]
            # Sample each dimension independently
            components = [sampler((batch_size, 1)) for sampler in samplers]
            return torch.cat(components, dim=1)

        return sample

    else:
        raise ValueError(
            f"name must be str (for IID) or list of tuples (for non-IID), got {type(name)}"
        )


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    """EMA update helper for model parameters."""
    with torch.no_grad():
        for p_ema, p in zip(ema_model.parameters(), model.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def compute_ema(values: list[float], alpha: float = 0.05) -> list[float]:
    """Compute exponential moving average for loss plotting.

    Args:
        values: List of scalar values (e.g., training losses)
        alpha: Smoothing factor (0 < alpha <= 1). Lower = more smoothing.
               Typical values: 0.01 (very smooth), 0.05 (moderate)

    Returns:
        List of EMA values with same length as input
    """
    if not values:
        return []
    ema = [values[0]]
    for val in values[1:]:
        ema.append(alpha * val + (1 - alpha) * ema[-1])
    return ema


def euler_integrate(
    model: nn.Module,
    z0: torch.Tensor,
    flow_T: float,
    *,
    steps: int = 120,
    return_path: bool = False,
) -> torch.Tensor:
    """Integrate the learned velocity field with explicit Euler method."""
    times = torch.linspace(flow_T, 0.0, steps, device=z0.device)
    x = z0
    if return_path:
        states = [x]
    for i in range(len(times) - 1):
        t_curr = times[i].expand(z0.shape[0], 1)
        dt = times[i + 1] - times[i]
        velocity = model(t_curr / flow_T, x)
        x = x + dt * velocity
        if return_path:
            states.append(x)
    if return_path:
        return torch.stack(states, dim=0)
    return x


# ============================================================================
# QUANTITATIVE METRICS
# ============================================================================

def compute_w2_distance(
    samples1: torch.Tensor,
    samples2: torch.Tensor,
    blur: float = 0.01,
) -> float:
    """Compute Sinkhorn divergence (approximation of W2) between two sample sets.
    
    Args:
        samples1: First sample set (N, D)
        samples2: Second sample set (M, D)
        blur: Regularization parameter for Sinkhorn
        
    Returns:
        Sinkhorn divergence value
    """
    if SamplesLoss is None:
        raise ImportError("geomloss is required for W2 distance computation. Install with: pip install geomloss")
    
    loss = SamplesLoss("sinkhorn", p=2, blur=blur, backend="tensorized")
    
    # Ensure same device
    device = samples1.device
    samples2 = samples2.to(device)
    
    with torch.no_grad():
        distance = loss(samples1, samples2)
    
    return float(distance.item())


def compute_distribution_metrics(
    generated_samples: torch.Tensor,
    target_samples: torch.Tensor,
    blur: float = 0.01,
) -> dict:
    """Compute multiple distribution comparison metrics.
    
    Args:
        generated_samples: Generated samples (N, D)
        target_samples: Target distribution samples (M, D)
        blur: Regularization for Sinkhorn
        
    Returns:
        Dictionary with metric names and values
    """
    metrics = {}
    
    # Sinkhorn divergence (W2 approximation)
    if SamplesLoss is not None:
        try:
            metrics['sinkhorn_div'] = compute_w2_distance(generated_samples, target_samples, blur=blur)
        except Exception as e:
            print(f"Warning: Could not compute Sinkhorn divergence: {e}")
    
    # Simple statistics
    gen_mean = generated_samples.mean(dim=0)
    target_mean = target_samples.mean(dim=0)
    metrics['mean_error'] = float(torch.norm(gen_mean - target_mean).item())
    
    gen_std = generated_samples.std(dim=0)
    target_std = target_samples.std(dim=0)
    metrics['std_error'] = float(torch.norm(gen_std - target_std).item())
    
    return metrics


# ============================================================================
# OPTIMAL TRANSPORT UTILITIES
# ============================================================================

def minibatch_ot_pairing(
    x0: torch.Tensor,
    x1: torch.Tensor,
    *,
    entropic_eps: Optional[float] = None,
    hard_match: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute minibatch OT pairing between two batches using POT library.
    
    Args:
        x0: Source batch (B, D)
        x1: Target batch (B, D)
        entropic_eps: Ignored (always uses exact EMD)
        hard_match: Ignored (pairing via row-wise argmax of EMD plan)
    
    Returns:
        idx1: (B,) indices so that pair i is (x0[i], x1[idx1[i]])
        P: (B, B) EMD transport plan (torch.Tensor)
    """
    if ot is None:
        raise ImportError("POT (Python Optimal Transport) is required. Install with: pip install pot")

    if x0.shape[0] != x1.shape[0]:
        raise ValueError(
            f"x0 and x1 must have same batch size. "
            f"Got x0: {x0.shape[0]}, x1: {x1.shape[0]}"
        )

    device = x0.device
    with torch.no_grad():
        C = torch.cdist(x0, x1, p=2).pow(2).cpu().numpy()
        a = ot.unif(C.shape[0])
        b = ot.unif(C.shape[1])
        P_np = ot.emd(a, b, C)
        P = torch.tensor(P_np, dtype=torch.float32, device=device)
        idx1 = torch.argmax(P, dim=1)

    return idx1, P


# ============================================================================
# TRAINING CONFIGURATIONS AND LOOPS
# ============================================================================

@dataclass
class FlowConfig:
    """Configuration for flow matching training."""
    target_sampler: BatchSampler
    latent_sampler: LatentSampler
    flow_T: float = 1.0
    seed: int = 3
    lr: float = 5e-4
    ema_decay: float = 0.9
    batch_size: int = 128
    steps: int = 50000
    log_every: int = 200
    grad_clip: float = 1.0
    pairing: Literal["none", "minibatch_ot"] = "none"
    minibatch_ot_fn: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None


@dataclass
class FlowRun:
    """Container for training artifacts."""
    config: FlowConfig
    model: nn.Module
    ema: nn.Module
    losses: List[float]
    pairing_costs: List[float]


def train_flow(cfg: FlowConfig) -> FlowRun:
    """Train a 2D flow-matching model with optional OT pairing."""
    set_seed(cfg.seed)
    model = VelocityMLP(input_dim=2).to(DEVICE)
    ema_model = VelocityMLP(input_dim=2).to(DEVICE)
    ema_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    losses: List[float] = []
    pairing_costs: List[float] = []

    for step in range(cfg.steps):
        x_data = cfg.target_sampler(cfg.batch_size, device=DEVICE, dtype=torch.float32)
        z = cfg.latent_sampler((cfg.batch_size, 2))

        if cfg.pairing == "minibatch_ot":
            if cfg.minibatch_ot_fn is None:
                raise ValueError("`minibatch_ot_fn` must be provided when pairing='minibatch_ot'.")
            cost_matrix = torch.cdist(x_data, z, p=2).pow(2)
            indices, transport_plan = cfg.minibatch_ot_fn(x_data, z)
            pairing_costs.append(float((transport_plan * cost_matrix).sum().item()))
            z = z[indices]

        t = torch.rand(cfg.batch_size, 1, device=DEVICE)
        x_t = (1.0 - t) * x_data + t * z
        velocity_target = z - x_data

        pred = model(t, x_t)
        loss = ((pred - velocity_target) ** 2).sum(dim=1).mean()

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        update_ema(ema_model, model, cfg.ema_decay)

        losses.append(loss.detach().cpu().item())
        if (step + 1) % cfg.log_every == 0:
            msg = f"step {step + 1:5d} | loss = {losses[-1]:.6f}"
            if cfg.pairing == "minibatch_ot" and pairing_costs:
                msg += f" | OT cost ≈ {pairing_costs[-1]:.4f}"
            print(msg)

    return FlowRun(cfg, model, ema_model, losses, pairing_costs)
