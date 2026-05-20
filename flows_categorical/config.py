# Copied from https://github.com/JChemseddine/spherical (paper-release-anon)
# Paper: Spherical Flows for Sampling Categorical Data (arXiv:2605.05629)

"""Configuration system using Python dataclasses."""

from dataclasses import dataclass, asdict, field, fields
from typing import Optional
import json


def _filter_kwargs(cls, d: dict) -> dict:
    """Filter dict to only keys that are valid fields of a dataclass."""
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in valid}


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    hidden_size: int = 256      # Transformer internal dimension
    embed_dim: int = 10         # Embedding/latent dimension r (= V for Stage 0)
    n_blocks: int = 6
    n_heads: int = 8
    dropout: float = 0.1
    vocab_size: int = 10        # Sudoku: 0-9 (0=blank, 1-9=digits)
    sequence_length: int = 81   # 9x9 Sudoku grid
    compile: bool = False       # torch.compile optimization
    learned_tau: bool = True         # Output norm = learned temperature (always on)
    velocity_head_hidden: int = 256  # Hidden dim of velocity head (0 = disabled)
    backbone: str = "dit"            # "dit" (our DiT) or "mdlm_dit" (MDLM DiT with adaLN)
    cond_dim: int = 128              # adaLN conditioning dim (only used by mdlm_dit)
    per_position_bias: bool = False  # Per-position logit bias (L, V). OFF for large vocab (LM1B).
    time_conditioning: bool = False  # Pass noise level to adaLN (True) or zeros (False).
    scale_embeddings: bool = False   # VE: scale W_E by sqrt(embed_dim) for d-independent SNR.


@dataclass
class FlowConfig:
    """Flow matching configuration (shared across all noise processes)."""
    noise_process: str = "vmf"         # "vmf", "ve", "masked", "geodesic"
    use_warp: bool = True              # Enable learnable time warp (ignored for masked)
    kappa_max: float = 50.0            # Max vMF concentration (kappa(t) = kappa_max * t)
    psi_grid_size: int = 1000          # Psi table resolution per axis
    vmf_sample_method: str = "cdf"     # "rejection" (Wood's, exact) or "cdf" (table-based, faster)
    S: int = 32                        # Denoising steps at inference
    lambda_mse: float = 1.0            # MSE loss weight for velocity head
    sampling_steps: int = 32           # ODE integration steps (alias for S)
    sampler_method: str = "softmax"    # "softmax" or "velocity_head"
    num_eval_samples: int = 100        # Samples to generate during validation
    num_vis_samples: int = 3           # Samples to visualize during validation
    time_warp_enabled: bool = False    # Enable Gaussian CDF warp (adaptive kappa schedule)
    time_warp_warmup: int = 5000       # Steps to ramp from linear to learned warp
    warp_bins: int = 16                # Number of piecewise linear bins
    warp_lr: float = 1e-4              # Separate learning rate for warp parameters
    warp_ema_decay: float = 0.99       # EMA decay for warp parameters
    warp_target_power: float = 1.0     # CE(u) = CE_max * (1-u)^p. p>1: fast early resolution, p<1: slow
    warp_ce_floor: float = 0.0         # Clamp target CE from below (0 = no floor). E.g. 0.001
    warp_aware_sampling: bool = False   # Use warp F'(u) for per-position step sizes at inference
    importance_weighting: bool = True   # Reweight CE by 1/F'(kappa) to correct for non-uniform sampling
    corrector_steps: int = 0           # Langevin correction steps per correction (0 = disabled)
    corrector_interval: int = 1        # Correct every Nth predict step (1 = every step)
    corrector_epsilon: float = 0.01    # Langevin step size ε
    corrector_scaling: bool = True         # Scale ε by (1-u)² at inference, u = progress ∈ [0,1] (noise→clean). Bounds Langevin step at all trajectory points.
    # Deprecated, kept for back-compat with old configs (samplers fall back to these if corrector_scaling is not set):
    corrector_kappa_scaling: bool = False  # (vmf, legacy) ε / max(κ,1)
    corrector_sigma_scaling: bool = False  # (VE, legacy) ε · min(σ,1)
    blank_token_id: int = -1           # Token to mask at final readout (-1 = no masking, 0 = mask token 0 for Sudoku)
    # VE diffusion params
    sigma_max: float = 50.0            # Max noise level for VE diffusion
    sigma_min: float = 0.01            # Min noise level for VE diffusion (singular endpoint floor)
    # FM time-cap (analogous to VE's sigma_min: avoid t=1 where velocity (D-h)/(1-t) blows up)
    t_eps: float = 1e-3                # Time cap: t ∈ [0, 1 - t_eps]. Default 1e-3 matches paper.
    # SDE params
    sde_sigma: float = 1.0             # Diffusion coefficient for reverse SDE sampling
    # Sphere retraction
    use_expmap: bool = False           # True = exponential map (exact), False = normalize retraction (default)
    # Masked diffusion params
    mask_token_id: int = -1            # [MASK] token index for discrete flow matching (-1 = auto, uses vocab_size)
    mask_schedule_power: float = 1.0   # Masking rate = t^p (1.0 = linear, >1 = late masking)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    grad_clip: float = 1.0
    seed: int = 42
    eta_min_ratio: float = 0.1  # For cosine annealing LR
    ema_decay: float = 0.0      # EMA decay on model params (0 = disabled). Disables weight_decay when active.
    grad_accum_steps: int = 1   # Gradient accumulation steps (effective batch = batch_size * world_size * grad_accum_steps)
    use_bf16: bool = False      # Enable bfloat16 mixed precision (no GradScaler needed)


@dataclass
class LoggingConfig:
    """Logging and checkpointing configuration."""
    log_freq: int = 100
    save_freq: int = 10000
    sample_freq: int = 5000
    wandb_enabled: bool = True
    wandb_project: str = "spherical-flow"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    enable_diagnostics: bool = True


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset_type: str = "sudoku"  # "sudoku" or "lm1b"
    data_dir: str = "data/sudoku-extreme-full"
    return_inputs: bool = True  # Return input-label pairs (for conditional generation)
    subset: str = "all"
    tokenizer_name: Optional[str] = None  # Tokenizer for text datasets (e.g. "bert-base-uncased" for LM1B)


@dataclass
class Config:
    """Main configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    checkpoint_dir: str = "./checkpoints"
    device: str = "cuda"
    resume_from: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> 'Config':
        """Create Config from dictionary. Ignores unknown keys for backwards compatibility."""
        # Accept old "score" key as alias for "flow"
        flow_dict = d.get("flow", d.get("score", {}))
        return cls(
            model=ModelConfig(**_filter_kwargs(ModelConfig, d.get("model", {}))),
            flow=FlowConfig(**_filter_kwargs(FlowConfig, flow_dict)),
            training=TrainingConfig(**_filter_kwargs(TrainingConfig, d.get("training", {}))),
            logging=LoggingConfig(**_filter_kwargs(LoggingConfig, d.get("logging", {}))),
            data=DataConfig(**_filter_kwargs(DataConfig, d.get("data", {}))),
            checkpoint_dir=d.get("checkpoint_dir", "./checkpoints"),
            device=d.get("device", "cuda"),
            resume_from=d.get("resume_from", None),
        )

    def to_dict(self) -> dict:
        """Convert Config to dictionary."""
        return asdict(self)

    @classmethod
    def from_json(cls, path: str) -> 'Config':
        """Load Config from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str) -> None:
        """Save Config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.model.n_heads > 0, "n_heads must be positive"
        assert self.model.hidden_size % self.model.n_heads == 0, \
            f"hidden_size ({self.model.hidden_size}) must be divisible by n_heads ({self.model.n_heads})"
        assert self.model.embed_dim > 0, "embed_dim must be positive"

        assert self.training.batch_size > 0, "batch_size must be positive"
        assert self.training.learning_rate > 0, "learning_rate must be positive"
        assert self.training.max_steps > 0, "max_steps must be positive"

        _valid_noise = ("vmf", "ve", "masked", "geodesic", "vp")
        assert self.flow.noise_process in _valid_noise, \
            f"noise_process must be one of {_valid_noise}, got '{self.flow.noise_process}'"
        assert self.flow.kappa_max > 0, "kappa_max must be positive"
        assert self.flow.sampling_steps > 0, "sampling_steps must be positive"
        _valid_methods = ("softmax", "velocity_head", "pc_softmax", "pc_velocity_head",
                          "sde_softmax", "sde_velocity_head", "pc_sde_softmax", "pc_sde_velocity_head")
        assert self.flow.sampler_method in _valid_methods, \
            f"sampler_method must be one of {_valid_methods}, got '{self.flow.sampler_method}'"