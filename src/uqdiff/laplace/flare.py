import torch
import torch.nn as nn
from laplace import Laplace
from typing import Optional

from uqdiff.laplace.wrapper import LaplaceWrapper
from uqdiff.laplace.dataset import make_laplace_dataset, make_laplace_loader


def select_random_params(
    model: nn.Module,
    frac: float = 0.1,
    seed: Optional[int] = None,
) -> list[str]:
    """
    Randomly select a fraction of named parameters to include in the Hessian.

    Parameters
    ----------
    model : nn.Module
    frac  : fraction of total parameters to select (0 < frac <= 1.0)
    seed  : RNG seed for reproducibility

    Returns
    -------
    selected_names : list of parameter names with requires_grad=True
    """
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    all_params = [(name, p) for name, p in model.named_parameters()]
    total = len(all_params)
    k = max(1, int(round(frac * total)))

    perm = torch.randperm(total, generator=rng).tolist()
    selected_indices = set(perm[:k])
    selected_names = [all_params[i][0] for i in selected_indices]

    return selected_names


def build_flare(
    ema_model: nn.Module,
    X,
    abar: torch.Tensor,
    ls_mu: torch.Tensor,
    ls_sd: torch.Tensor,
    T: int,
    data_dim: int,
    device: str = "cuda",
    frac: float = 0.1,
    hessian_structure: str = "diag",   # "diag" or "full" over the selected subset
    N_pairs: int = 100_000,
    batch: int = 4096,
    optimize_prior: bool = True,
    seed: Optional[int] = None,
    verbose: bool = True,
):
    """
    Build and fit a FLARE (randomized subset) Laplace approximation.

    Parameters
    ----------
    ema_model         : trained EMA ScoreNet
    X                 : (N, data_dim) training data
    abar              : (T,) cumulative alpha schedule
    ls_mu / ls_sd     : logSNR normalization stats
    T                 : diffusion steps
    data_dim          : sequence length
    device            : "cuda" | "cpu"
    frac              : fraction of parameters to include (e.g. 0.1 = 10%)
    hessian_structure : "diag" or "full" over the selected subset
    N_pairs           : number of (x_t, ε) pairs for Hessian fitting
    batch             : DataLoader batch size
    optimize_prior    : run optimize_prior_precision after fitting
    seed              : random seed for parameter selection
    verbose           : print progress

    Returns
    -------
    la              : fitted Laplace object
    wrapped         : LaplaceWrapper
    selected_names  : list of parameter names included in Hessian
    """
    base = copy.deepcopy(ema_model).eval().to(device)

    # freeze all params first
    for p in base.parameters():
        p.requires_grad_(False)

    wrapped = LaplaceWrapper(base, abar, ls_mu, ls_sd, data_dim).to(device)

    # randomly select subset and unfreeze
    selected_names = select_random_params(wrapped, frac=frac, seed=seed)
    selected_set   = set(selected_names)

    for name, p in wrapped.named_parameters():
        if name in selected_set:
            p.requires_grad_(True)

    n_selected = sum(
        p.numel() for name, p in wrapped.named_parameters()
        if name in selected_set
    )
    n_total = sum(p.numel() for p in wrapped.parameters())

    if verbose:
        print(f"FLARE: selected {len(selected_names)}/{sum(1 for _ in wrapped.parameters())} "
              f"param tensors  ({n_selected:,}/{n_total:,} scalars, frac={frac:.2f})")

    # build dataset + loader
    if verbose:
        print(f"Building Laplace dataset (N_pairs={N_pairs}) …")
    X_lap, Y_lap, _ = make_laplace_dataset(
        X, abar, T, N_pairs=N_pairs, data_dim=data_dim, device=device,
    )
    loader = make_laplace_loader(X_lap, Y_lap, batch=batch)

    # build Laplace object
    la = Laplace(
        wrapped,
        likelihood="regression",
        subset_of_weights="all",          # we manually controlled requires_grad
        hessian_structure=hessian_structure,
    )

    # fit
    if verbose:
        print(f"Fitting FLARE [{hessian_structure}] ({len(loader)} batches) …")
    la.fit(loader)

    if optimize_prior:
        if verbose:
            print("Optimizing prior precision …")
        la.optimize_prior_precision(method="marglik")  # type: ignore[attr-defined]

    if verbose:
        pp = la.prior_precision
        pp_mean = float(pp.mean()) if hasattr(pp, "mean") else float(pp)
        print(f"Done. prior_precision={pp_mean:.4f}")

    return la, wrapped, selected_names






















