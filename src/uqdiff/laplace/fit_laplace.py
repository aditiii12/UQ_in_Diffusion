"""
Unified Laplace approximation builder and fitter.

Supports:
  - "diag"  : Last-Layer Laplace, diagonal Hessian (LLLA)
  - "full"  : Full-parameter Laplace, full Hessian
  - "flare" : Randomized subset Laplace, diagonal Hessian over random params

Usage
-----
from uqdiff.laplace.fit_laplace import build_and_fit_laplace

la, wrapped = build_and_fit_laplace(
    ema_model, X, abar, ls_mu, ls_sd,
    T=1000, data_dim=10, device="cuda",
    hessian="diag",   # or "full" or "flare"
    N_pairs=100_000,
)
"""

import copy
import torch
import torch.nn as nn
from laplace import Laplace
from typing import Optional

from uqdiff.laplace.wrapper import LaplaceWrapper
from uqdiff.laplace.dataset import make_laplace_dataset, make_laplace_loader


def build_and_fit_laplace(
    ema_model: nn.Module,
    X,
    abar: torch.Tensor,
    ls_mu: torch.Tensor,
    ls_sd: torch.Tensor,
    T: int,
    data_dim: int,
    device: str = "cuda",
    hessian: str = "diag",      # "diag" (LLLA) | "full" | "flare"
    N_pairs: int = 100_000,
    batch: int = 4096,
    optimize_prior: bool = True,
    verbose: bool = True,
    # FLARE-specific
    flare_frac: float = 0.1,
    flare_seed: Optional[int] = None,
    flare_hessian_structure: str = "diag",
):
    """
    Build, fit, and optionally optimize a Laplace approximation.

    Parameters
    ----------
    ema_model               : trained EMA ScoreNet
    X                       : (N, data_dim) training data (numpy or tensor)
    abar                    : (T,) cumulative alpha schedule
    ls_mu/ls_sd             : logSNR normalization stats
    T                       : number of diffusion steps
    data_dim                : sequence length
    device                  : "cuda" | "cpu"
    hessian                 : "diag" (LLLA), "full", or "flare"
    N_pairs                 : number of (x_t, eps) pairs for Hessian fitting
    batch                   : DataLoader batch size
    optimize_prior          : whether to run optimize_prior_precision after fit
    verbose                 : print progress
    flare_frac              : (FLARE only) fraction of params to include
    flare_seed              : (FLARE only) seed for random param selection
    flare_hessian_structure : (FLARE only) "diag" or "full" over selected subset

    Returns
    -------
    la      : fitted Laplace object
    wrapped : LaplaceWrapper (use wrapped.net to access ScoreNet)
    """
    if hessian not in ("diag", "full", "flare"):
        raise ValueError(f"hessian must be 'diag', 'full', or 'flare', got '{hessian}'")

    # delegate FLARE to its own module
    if hessian == "flare":
        from uqdiff.laplace.flare import build_flare
        la, wrapped, _ = build_flare(
            ema_model, X, abar, ls_mu, ls_sd,
            T=T, data_dim=data_dim, device=device,
            frac=flare_frac, hessian_structure=flare_hessian_structure,
            N_pairs=N_pairs, batch=batch,
            optimize_prior=optimize_prior,
            seed=flare_seed, verbose=verbose,
        )
        return la, wrapped

    # ── Build wrapped model ───────────────────────────────────────────────
    base = copy.deepcopy(ema_model).eval().to(device)

    if hessian == "diag":
        for p in base.parameters():
            p.requires_grad_(False)

        wrapped = LaplaceWrapper(base, abar, ls_mu, ls_sd, data_dim).to(device)

        last_layer_name = None
        for name, m in wrapped.named_modules():
            if isinstance(m, nn.Linear) and m.out_features == data_dim:
                last_layer_name = name

        if last_layer_name is None:
            raise RuntimeError(f"No Linear(..., {data_dim}) found in wrapped model.")

        for name, m in wrapped.named_modules():
            if name == last_layer_name:
                for p in m.parameters():
                    p.requires_grad_(True)
                break

        subset    = "last_layer"
        hessian_  = "diag"
        la_kwargs = {"last_layer_name": last_layer_name}

    else:  # full
        for p in base.parameters():
            p.requires_grad_(True)

        wrapped = LaplaceWrapper(base, abar, ls_mu, ls_sd, data_dim).to(device)
        for p in wrapped.parameters():
            p.requires_grad_(True)

        subset    = "all"
        hessian_  = "full"
        la_kwargs = {}

    # ── Build dataset + loader ────────────────────────────────────────────
    if verbose:
        print(f"Building Laplace dataset (N_pairs={N_pairs}) …")
    X_lap, Y_lap, _ = make_laplace_dataset(
        X, abar, T, N_pairs=N_pairs, data_dim=data_dim, device=device,
    )
    loader = make_laplace_loader(X_lap, Y_lap, batch=batch)

    # ── Build Laplace object ──────────────────────────────────────────────
    la = Laplace(
        wrapped,
        likelihood="regression",
        subset_of_weights=subset,
        hessian_structure=hessian_,
        **la_kwargs,
    )

    # ── Fit ───────────────────────────────────────────────────────────────
    if verbose:
        print(f"Fitting Laplace [{hessian}] ({len(loader)} batches) …")
    la.fit(loader)

    if optimize_prior:
        if verbose:
            print("Optimizing prior precision …")
        la.optimize_prior_precision(method="marglik")  # type: ignore[attr-defined]

    if verbose:
        n_params = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
        pp = la.prior_precision
        pp_mean = float(pp.mean()) if hasattr(pp, "mean") else float(pp) # type: ignore
        print(f"Done. Trainable params={n_params:,}  prior_precision={pp_mean:.4f}")

    return la, wrapped














