"""
uqdiff
------
Epistemic uncertainty quantification for diffusion models via Laplace approximation.

Core modules:
  - uqdiff.utils     : timecodes, robust_chol, randn_like_gen
  - uqdiff.schedules : cosine schedule, compute_alpha
  - uqdiff.ddpm      : forward/reverse diffusion process
  - uqdiff.laplace   : Laplace fitting, γ², BayesDiff, FLARE
"""

from uqdiff.utils import (
    logsnr_from_abar,
    prep_time_stats,
    time_code_from_abar,
    timecode_from_tindex,
    timecode_from_tnorm,
    randn_like_gen,
    robust_chol,
)

from uqdiff.schedules import (
    cosine_beta_schedule,
    make_schedules,
    compute_alpha,
)

from uqdiff.ddpm import (
    q_sample,
    p_sample_step,
    sample_ddpm,
)

__all__ = [
    # utils
    "logsnr_from_abar",
    "prep_time_stats",
    "time_code_from_abar",
    "timecode_from_tindex",
    "timecode_from_tnorm",
    "randn_like_gen",
    "robust_chol",
    # schedules
    "cosine_beta_schedule",
    "make_schedules",
    "compute_alpha",
    # ddpm
    "q_sample",
    "p_sample_step",
    "sample_ddpm",
]