"""
uqdiff/laplace/precision.py
----------------------------
Computes γ²(x_t, t) = diag(J Σ J^T) — the per-output epistemic variance
from the Laplace posterior, used in BayesDiff and FLARE.

Three variants:
  - gamma2_diag    : diagonal last-layer (LLLA) — fast, closed form
  - gamma2_full    : full/subnet Hessian via vmap Jacobians — exact but slow
  - pack_params    : helper to align params with Laplace backend order
"""

from __future__ import annotations
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.func import grad, vmap, functional_call

from uqdiff.utils import timecode_from_tindex, robust_chol


# ---------------------------------------------------------------------------
# Diagonal last-layer (LLLA)
# ---------------------------------------------------------------------------

@torch.no_grad()
def gamma2_diag(
    la,
    score_model: nn.Module,   # ScoreNet with .forward_with_feat(xt, t_code) -> (eps, h)
    xt: torch.Tensor,         # (B, data_dim)
    t_idx: torch.Tensor,      # (B,) int64
    abar: torch.Tensor,
    ls_mu: torch.Tensor,
    ls_sd: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (mu_eps, gamma2):
        mu_eps  : (B, data_dim)  — predicted noise
        gamma2  : (B, data_dim)  — per-dim epistemic variance
    """
    t_code         = timecode_from_tindex(t_idx, abar, ls_mu, ls_sd)
    mu_eps, h      = score_model.forward_with_feat(xt, t_code)  # (B,D), (B,H)
    B, H           = h.shape
    D              = mu_eps.shape[1]

    prec  = la.posterior_precision.detach().to(h.device, h.dtype)
    var   = 1.0 / (prec + eps)

    W_var = var[:D * H].view(D, H)   # (D, H)
    b_var = var[D * H:]              # (D,)

    h2     = h * h                   # (B, H)
    gamma2 = (h2 @ W_var.T) + b_var.unsqueeze(0)   # (B, D)
    return mu_eps, gamma2.clamp_min(0.0)


# ---------------------------------------------------------------------------
# Full / subnet Hessian via vmap Jacobians
# ---------------------------------------------------------------------------

@torch.no_grad()
def pack_params(model: nn.Module, backend) -> tuple:
    """
    Extract params in the exact order Laplace's backend used.
    Returns (params, buffers, keys, shapes, idx_slices, total_numel)
    """
    params  = {k: p.detach().requires_grad_(True) for k, p in backend.params_dict.items()}
    buffers = {k: b for k, b in model.named_buffers()}
    keys    = list(params.keys())
    shapes  = [params[k].shape for k in keys]
    numels  = [params[k].numel() for k in keys]
    idx_slices, s = [], 0
    for n in numels:
        idx_slices.append(slice(s, s + n))
        s += n
    return params, buffers, keys, shapes, idx_slices, s


def _flatten_grads(grads_dict: dict, keys: list, params_ref: dict) -> torch.Tensor:
    flats = []
    for k in keys:
        g = grads_dict.get(k, None)
        if g is None:
            g = torch.zeros_like(params_ref[k])
        flats.append(g.reshape(-1))
    return torch.cat(flats, dim=0)


def _make_batched_grads_fn(wrapped, params_ref, buffers, keys):
    def scalar_out(p, x_cat_row, k):
        y = functional_call(wrapped, (p, buffers), (x_cat_row.unsqueeze(0),))
        return y[0, k]

    def grad_wrt_params(x_cat_row, k):
        gtree = grad(lambda p: scalar_out(p, x_cat_row, k))(params_ref)
        return _flatten_grads(gtree, keys, params_ref)

    def batched_grads(X_batch, k):
        return vmap(lambda row: grad_wrt_params(row, k))(X_batch)

    return batched_grads


@torch.no_grad()
def gamma2_full(
    wrapped: nn.Module,
    L: torch.Tensor,           # Cholesky of posterior precision (from robust_chol)
    params: dict,
    buffers: dict,
    keys: list,
    xt: torch.Tensor,          # (B, data_dim)
    t_idx: torch.Tensor,       # (B,) int64
    abar: torch.Tensor,
    max_batch: int = 256,
    timers: Optional[dict] = None,
    idx_sub: Optional[torch.Tensor] = None,   # subnetwork column indices if using subnet
) -> torch.Tensor:
    """
    Computes γ²(x_t, t) = diag(J Σ J^T) via vmap Jacobians.

    Works for both full Hessian and subnetwork Laplace (pass idx_sub for subnet).
    Returns: (B, data_dim) clipped to >= 0.
    """
    device, dtype = xt.device, xt.dtype
    abar  = abar.to(device).contiguous()
    t_idx = t_idx.to(device=device, dtype=torch.long)

    T      = abar.numel()
    t_norm = t_idx.float() / T
    x_cat  = torch.cat([xt, t_norm[:, None]], dim=1)

    D_out         = wrapped.net.out.out_features
    batched_grads = _make_batched_grads_fn(wrapped, params, buffers, keys)

    B      = x_cat.shape[0]
    gamma2 = torch.empty(B, D_out, device=device, dtype=dtype)
    stepB  = max_batch if max_batch else B

    for s in range(0, B, stepB):
        e  = min(B, s + stepB)
        Xb = x_cat[s:e]

        t0     = time.time()
        G_list = [batched_grads(Xb, k) for k in range(D_out)]
        if timers is not None:
            timers["gamma_grads"] = timers.get("gamma_grads", 0.0) + (time.time() - t0)

        if idx_sub is not None:
            idx_sub_dev = idx_sub.to(G_list[0].device)
            G_list = [G[:, idx_sub_dev] for G in G_list]

        t1     = time.time()
        U_list = [torch.cholesky_solve(G.t().contiguous(), L).t() for G in G_list]
        if timers is not None:
            timers["gamma_solve"] = timers.get("gamma_solve", 0.0) + (time.time() - t1)

        V = [(G * U).sum(dim=1).to(dtype) for G, U in zip(G_list, U_list)]
        gamma2[s:e] = torch.stack(V, dim=1)

    return gamma2.clamp_min(0.0)


def get_cholesky(la, device: str) -> torch.Tensor:
    """Extract and Cholesky-decompose the posterior precision from a Laplace object."""
    P = la.posterior_precision
    if not torch.is_tensor(P):
        P = P.to_dense()
    return robust_chol(P.to(device=device))