"""
darksnail: High-Speed Optical/IR Seeker & Lie-Algebraic Triad Primitives.

Exports the core mathematical and tracking modules:
- `tracker`: `DarkSnailHighSpeedTracker`, `TrackerConfig`, `InnovationSpikeGateConfig`, `verify_formal_theorems`
- `shell`: `BeltramiCallosalShell`, `PullbackCanvasResult`, `compute_grid_prompt_zeno_flux`, `extract_beltrami_lh_latents`
- `readout`: `SkewSymmetricTriadOperator`, `verify_pointwise_orthogonality`,
  `DarkSnailCanvasReadout`, `SlowFastTriadCascade`, `RoPE2DMedoidPooler`,
  `ContinuousSpectralFieldReadout`, `get_continuous_spectral_and_ray_basis`
- `arc3`: `TemporalClockContentSplitter`, `compute_marginal_wasserstein1_distance`
"""

__version__ = "0.1.0"

from .arc3 import (
    TemporalClockContentSplitter,
    compute_marginal_wasserstein1_distance,
)
from .readout import (
    ContinuousSpectralFieldReadout,
    DarkSnailCanvasReadout,
    RoPE2DMedoidPooler,
    SkewSymmetricTriadOperator,
    SlowFastTriadCascade,
    get_continuous_spectral_and_ray_basis,
    verify_pointwise_orthogonality,
)
from .shell import (
    BeltramiCallosalShell,
    PullbackCanvasResult,
    compute_grid_prompt_zeno_flux,
    extract_beltrami_lh_latents,
    extract_beltrami_pullback_canvas,
    extract_grid_beltrami_shape_descriptors,
    infer_arc_output_shape_beltrami,
)
from .tracker import (
    DarkSnailHighSpeedTracker,
    InnovationSpikeGateConfig,
    TrackerConfig,
    verify_formal_theorems,
)

__all__ = [
    "__version__",
    "BeltramiCallosalShell",
    "PullbackCanvasResult",
    "compute_grid_prompt_zeno_flux",
    "extract_beltrami_lh_latents",
    "extract_beltrami_pullback_canvas",
    "extract_grid_beltrami_shape_descriptors",
    "infer_arc_output_shape_beltrami",
    "SkewSymmetricTriadOperator",
    "verify_pointwise_orthogonality",
    "DarkSnailCanvasReadout",
    "SlowFastTriadCascade",
    "RoPE2DMedoidPooler",
    "ContinuousSpectralFieldReadout",
    "get_continuous_spectral_and_ray_basis",
    "TemporalClockContentSplitter",
    "compute_marginal_wasserstein1_distance",
    "DarkSnailHighSpeedTracker",
    "TrackerConfig",
    "InnovationSpikeGateConfig",
    "verify_formal_theorems",
]
