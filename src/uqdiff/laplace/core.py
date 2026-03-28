"""
uqdiff/laplace/core.py
-----------------------
Everything needed to fit Laplace on a trained score model:

  - LaplaceWrapper   : wraps ScoreNet(xt, t_code) -> Laplace-compatible [x_t, t_norm] input
  - DiffusionShim    : minimal betas container for BayesDiff samplers
  - make_laplace_dataset / make_laplace_loader : post-training Laplace fitting data
  - build_llla       : last-layer diagonal Laplace
  - build_subnet     : random subnetwork full Laplace (practical for large models)
  - build_full       : full Hessian Laplace (feasible for small models)
"""

from __future__ import annotations
import copy
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from laplace import Laplace

from uqdiff.utils import timecode_from_tindex


# ---------------------------------------------------------------------------
# LaplaceWrapper
# ---------------------------------------------------------------------------

class LaplaceWrapper(nn.Module):
    """
    Wraps a score model so Laplace sees input x_cat = [x_t, t_norm].

    Args:
        net      : ScoreNet (or any nn.Module with forward(xt, t_code) -> eps)
        abar     : (T,) cumulative alphas
        ls_mu    : logSNR mean from prep_time_stats(abar)
        ls_sd    : logSNR std  from prep_time_stats(abar)
        data_dim : dimensionality of x_t
    """
    def __init__(
        self,
        net: nn.Module,
        abar: torch.Tensor,
        ls_mu: torch.Tensor,
        ls_sd: torch.Tensor,
        data_dim: int,
    ):
        super().__init__()
        self.net      = net
        self.data_dim = int(data_dim)
        self.register_buffer("abar",  torch.as_tensor(abar,  dtype=torch.float32).detach().clone())
        self.register_buffer("ls_mu", torch.as_tensor(ls_mu, dtype=torch.float32).detach().clone())
        self.register_buffer("ls_sd", torch.as_tensor(ls_sd, dtype=torch.float32).detach().clone())

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        xt     = x_cat[:, :self.data_dim]
        t_norm = x_cat[:, self.data_dim]
        T      = self.abar.numel()
        t_idx  = (t_norm * T).clamp_(0, T - 1).long()
        t_code = timecode_from_tindex(t_idx, self.abar, self.ls_mu, self.ls_sd)
        return self.net(xt, t_code)


# ---------------------------------------------------------------------------
# DiffusionShim
# ---------------------------------------------------------------------------

class DiffusionShim:
    """Minimal container so BayesDiff samplers don't need a full diffusion object."""
    def __init__(self, betas: torch.Tensor):
        self.betas         = betas
        self.num_timesteps = betas.numel()


# ---------------------------------------------------------------------------
# Laplace fitting dataset
# ---------------------------------------------------------------------------

@torch.no_grad()
def make_laplace_dataset(
    X: Union[torch.Tensor, np.ndarray],
    abar: torch.Tensor,
    T: int,
    N_pairs: int,
    data_dim: int,
    device: str = "cpu",
    snr_gamma: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build (x_t, t_norm) -> eps regression dataset for Laplace fitting.

    Returns:
        X_lap : (N_pairs, data_dim + 1)
        Y_lap : (N_pairs, data_dim)
        t     : (N_pairs,) integer timestep indices
    """
    X    = torch.as_tensor(X, dtype=torch.float32, device=device)
    abar = abar.to(device)
    assert X.ndim == 2 and X.shape[1] == data_dim

    N   = X.shape[0]
    idx = torch.randint(0, N, (N_pairs,), device=device)
    x0  = X[idx]

    if snr_gamma is None:
        t = torch.randint(0, T, (N_pairs,), device=device)
    else:
        snr = abar / (1.0 - abar + 1e-8)
        w   = torch.minimum(snr, torch.full_like(snr, snr_gamma)) / (snr + 1e-8)
        p   = (w.clamp_min(0) / w.sum()).pow(2.0)
        p   = p / p.sum()
        t   = torch.multinomial(p, N_pairs, replacement=True)

    abar_t = abar[t].unsqueeze(1)
    eps    = torch.randn(N_pairs, data_dim, device=device)
    x_t    = torch.sqrt(abar_t) * x0 + torch.sqrt(1.0 - abar_t) * eps
    tnorm  = t.float() / float(T)

    X_lap = torch.cat([x_t, tnorm[:, None]], dim=1)
    Y_lap = eps
    return X_lap, Y_lap, t


def make_laplace_loader(
    X_lap: torch.Tensor,
    Y_lap: torch.Tensor,
    batch: int = 4096,
    shuffle: bool = True,
) -> DataLoader:
    ds = TensorDataset(X_lap, Y_lap)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=True)


# ---------------------------------------------------------------------------
# Laplace builders
# ---------------------------------------------------------------------------

def _find_last_linear(model: nn.Module, out_features: int):
    """Find the last Linear layer with the given out_features."""
    last_name, last_mod = None, None
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and m.out_features == out_features:
            last_name, last_mod = name, m
    assert last_mod is not None, f"No Linear with out_features={out_features} found."
    return last_name, last_mod


def build_llla(
    ema_model: nn.Module,
    abar: torch.Tensor,
    ls_mu: torch.Tensor,
    ls_sd: torch.Tensor,
    data_dim: int,
    device: str = "cuda",
):
    """
    Last-layer Laplace (diagonal Hessian).
    Freezes all params except the final Linear layer.

    Returns: (la, wrapped, last_layer_name)
    """
    net = copy.deepcopy(ema_model).eval().to(device)
    for p in net.parameters():
        p.requires_grad_(False)

    wrapped = LaplaceWrapper(
        net, abar.to(device), ls_mu.to(device), ls_sd.to(device), data_dim
    ).to(device)

    last_name, last_mod = _find_last_linear(wrapped, data_dim)
    for p in last_mod.parameters():
        p.requires_grad_(True)

    la = Laplace(
        wrapped,
        likelihood="regression",
        subset_of_weights="last_layer",
        hessian_structure="diag",
        last_layer_name=last_name,
    )
    return la, wrapped, last_name


def build_subnet(
    ema_model: nn.Module,
    abar: torch.Tensor,
    ls_mu: torch.Tensor,
    ls_sd: torch.Tensor,
    data_dim: int,
    p_keep: float = 0.40,
    device: str = "cuda",
):
    """
    Subnetwork Laplace (full Hessian on a random p_keep fraction of params).
    Practical alternative to full Hessian for larger models.

    Returns: (la_sub, wrapped, sub_idx)
    """
    try:
        from laplace.utils import RandomSubnetMask
    except ImportError:
        from laplace.utils.subnetmask import RandomSubnetMask
    try:
        from laplace import FullSubnetLaplace, DiagSubnetLaplace
    except ImportError:
        from laplace.subnetlaplace import FullSubnetLaplace, DiagSubnetLaplace

    net = copy.deepcopy(ema_model).eval().to(device)
    for p in net.parameters():
        p.requires_grad_(False)

    wrapped = LaplaceWrapper(
        net, abar.to(device), ls_mu.to(device), ls_sd.to(device), data_dim
    ).to(device)

    for p in wrapped.parameters():
        p.requires_grad_(True)

    P_total = sum(p.numel() for p in wrapped.parameters())
    n_sub   = max(1, int(p_keep * P_total))
    mask    = RandomSubnetMask(wrapped, n_params_subnet=n_sub)
    raw_idx = mask.select()

    sub_idx = torch.as_tensor(raw_idx, dtype=torch.long).view(-1).cpu()
    sub_idx = sub_idx[(sub_idx >= 0) & (sub_idx < P_total)]
    sub_idx = torch.unique(sub_idx)
    assert sub_idx.numel() > 0, "Empty subnetwork after sanitizing indices."

    try:
        la_sub = FullSubnetLaplace(
            wrapped, likelihood="regression", subnetwork_indices=sub_idx
        )
    except RuntimeError:
        la_sub = DiagSubnetLaplace(
            wrapped, likelihood="regression", subnetwork_indices=sub_idx
        )
    return la_sub, wrapped, sub_idx


def build_full(
    ema_model: nn.Module,
    abar: torch.Tensor,
    ls_mu: torch.Tensor,
    ls_sd: torch.Tensor,
    data_dim: int,
    device: str = "cuda",
):
    """
    Full Hessian Laplace over all parameters.
    Only feasible for small models (data_dim=10, hidden_dim<=128).

    Returns: (la_full, wrapped)
    """
    net = copy.deepcopy(ema_model).eval().to(device)
    for p in net.parameters():
        p.requires_grad_(False)

    wrapped = LaplaceWrapper(
        net, abar.to(device), ls_mu.to(device), ls_sd.to(device), data_dim
    ).to(device)

    for p in wrapped.parameters():
        p.requires_grad_(True)

    la_full = Laplace(
        wrapped,
        likelihood="regression",
        subset_of_weights="all",
        hessian_structure="full",
    )
    return la_full, wrapped