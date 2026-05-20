# Copied from https://github.com/JChemseddine/spherical (paper-release-anon)
# Paper: Spherical Flows for Sampling Categorical Data (arXiv:2605.05629)

"""Method factory: dispatches to the correct training class based on noise_process."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flows_categorical.config import Config


def create_method(model, config: 'Config', **kwargs):
    """Create a training orchestrator for the configured noise process.

    Args:
        model: The backbone model
        config: Full config object
        **kwargs: Extra args forwarded to the training class (psi_table, warp, etc.)

    Returns:
        Training orchestrator with a .training_step() method.
    """
    noise = config.flow.noise_process

    # Training orchestrators (flow.py files) were not copied into the tutorial — inference only.
    raise NotImplementedError(
        f"create_method (training) not available in fm_tutorial — see the full source repo "
        f"https://github.com/JChemseddine/spherical for training code. Inference-only here. "
        f"Got noise_process={noise!r}."
    )


def create_sampler(model, config: 'Config', **kwargs):
    """Create a sampler for the configured noise process.

    Args:
        model: The backbone model
        config: Full config object
        **kwargs: Extra args (psi_table, warp, etc.)

    Returns:
        Sampler with a .sample() method.
    """
    noise = config.flow.noise_process

    if noise == "vmf":
        from .continuous.spherical.sampler import SphericalODESampler
        return SphericalODESampler(model, config, **kwargs)
    elif noise == "vp":
        from .continuous.vp.sampler import VPODESampler
        return VPODESampler(model, config, **kwargs)
    elif noise == "masked":
        from .discrete.masked.sampler import MaskedFlowSampler
        return MaskedFlowSampler(model, config, **kwargs)
    else:
        raise ValueError(
            f"Unknown or unsupported noise_process: {noise!r}. "
            f"fm_tutorial inference supports: vmf, vp, masked. "
            f"For ve / geodesic see the full source repo https://github.com/JChemseddine/spherical."
        )