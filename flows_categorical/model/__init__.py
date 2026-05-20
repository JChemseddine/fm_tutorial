# Copied from https://github.com/JChemseddine/spherical (paper-release-anon)
# Paper: Spherical Flows for Sampling Categorical Data (arXiv:2605.05629)

"""Model factory: dispatches to the correct backbone based on noise_process."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flows_categorical.config import Config


def create_model(config: 'Config'):
    """Create the backbone model for the configured noise process.

    All continuous methods (vMF, geodesic, VE) use ContinuousTransformer.
    Masked diffusion uses MaskedTransformer.
    Both share the same DDiTBlock architecture with adaLN conditioning.
    """
    noise = config.flow.noise_process

    if noise in ("vmf", "vp"):
        from .backbone import ContinuousTransformer
        return ContinuousTransformer(config.model.vocab_size, config.model)
    elif noise == "masked":
        from .backbone import MaskedTransformer
        return MaskedTransformer(config.model.vocab_size, config.model)
    else:
        raise ValueError(
            f"Unknown or unsupported noise_process: {noise!r}. "
            f"fm_tutorial inference supports: vmf, vp, masked. "
            f"For ve / geodesic see the full source repo https://github.com/JChemseddine/spherical."
        )