"""
uqdiff.laplace
--------------
Laplace approximation for epistemic uncertainty in diffusion models.
"""

from uqdiff.laplace.core import (
    LaplaceWrapper,
    DiffusionShim,
    make_laplace_dataset,
    make_laplace_loader,
    build_llla,
    build_subnet,
    build_full,
)

from uqdiff.laplace.precision import (
    gamma2_diag,
    gamma2_full,
    pack_params,
    get_cholesky,
)

from uqdiff.laplace.bayesdiff import (
    sample_bayesdiff,
)

from uqdiff.laplace.flare import (
    project_llla,
    project_full,
    sample_flare_full,
)

__all__ = [
    # core
    "LaplaceWrapper",
    "DiffusionShim",
    "make_laplace_dataset",
    "make_laplace_loader",
    "build_llla",
    "build_subnet",
    "build_full",
    # precision
    "gamma2_diag",
    "gamma2_full",
    "pack_params",
    "get_cholesky",
    # samplers
    "sample_bayesdiff",
    "project_llla",
    "project_full",
    "sample_flare_full",
]