
"""
uqdiff/laplace/flare.py
------------------------
FLARE: Fisher-Laplace Randomized Estimator for epistemic uncertainty
in diffusion models.

The FLARE projection transports per-step epistemic variance γ²_t
to x_0 via the closed-form accumulated transport factor:

    u_proj = Σ_t (Π_{s<t} a_s)² * b_t² * γ²_t

This is cheaper than the full BayesDiff recursion (no MC Cov needed)
and works with any γ² estimator: LLLA, subnet, or full Hessian.

Two modes:
  - project_llla     : replay FLARE on a pre-recorded path using LLLA γ²
  - project_full     : replay FLARE on a pre-recorded path using full/subnet γ²
  - bayesdiff_full   : full BayesDiff + FLARE on a fresh path using full/subnet γ²
                       (used for the full-Hessian experiment on sines)
"""

from __future__ import annotations
from typing import Optional

import torch

from uqdiff.utils import timecode_from_tindex, timecode_from_tnorm, randn_like_gen
from uqdiff.laplace.precision import gamma2_diag, gamma2_full, pack_params, get_cholesky


# ---------------------------------------------------------------------------
# Schedule helper
# ---------------------------------------------------------------------------

def _compute_alpha(betas: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    betas = torch.cat([torch.zeros(1, device=betas.device, dtype=betas.dtype), betas])
    return (1 - betas).cumprod(0).index_select(0, t.long() + 1).view(-1, 1)


# ---------------------------------------------------------------------------
# FLARE projection on a fixed path (LLLA γ²)
# ---------------------------------------------------------------------------

@torch.no_grad()
def project_llla(
    la,
    model,                   # ScoreNet with .forward_with_feat
    abar: torch.Tensor,
    ls_mu: torch.Tensor,
    ls_sd: torch.Tensor,
    xt_path: torch.Tensor,   # (K, n, data_dim) pre-recorded path
    a_path: torch.Tensor,    # (K, n, 1)
    b_path: torch.Tensor,    # (K, n, 1)
    t_seq: list,
    device: str = "cuda",
    tau_gamma2: float = 1.0,
    show_progress: bool = False,
) -> torch.Tensor:
    """
    Replay FLARE projection on a fixed x_t path using LLLA γ².
    Returns u_proj (n,).
    """
    betas = torch.zeros(1, device=device)   # not needed, path is fixed
    K, n, d = xt_path.shape
    xt_path = xt_path.to(device)
    a_path  = a_path.to(device)
    b_path  = b_path.to(device)
    abar    = abar.to(device)
    ls_mu   = ls_mu.to(device)
    ls_sd   = ls_sd.to(device)

    Var_x0_proj = torch.zeros(n, d, device=device)
    cum_a       = torch.ones(n, 1, device=device)

    iters = range(K)
    if show_progress:
        from tqdm import tqdm
        iters = tqdm(iters, desc="FLARE projection (LLLA)")

    for k in iters:
        t_int = int(t_seq[-1 - k])
        t_idx = torch.full((n,), t_int, device=device, dtype=torch.long)
        xt    = xt_path[k]
        a     = a_path[k]
        b     = b_path[k]

        _, gamma2_t = gamma2_diag(la, model, xt, t_idx, abar, ls_mu, ls_sd)
        gamma2_t    = (gamma2_t * tau_gamma2).clamp_min(0.0)

        Var_x0_proj = Var_x0_proj + (cum_a ** 2) * (b ** 2) * gamma2_t
        cum_a       = cum_a * a

    return Var_x0_proj.sum(1)


# ---------------------------------------------------------------------------
# FLARE projection on a fixed path (full/subnet γ²)
# ---------------------------------------------------------------------------

@torch.no_grad()
def project_full(
    wrapped,
    la_full,
    abar: torch.Tensor,
    xt_path: torch.Tensor,   # (K, n, data_dim)
    a_path: torch.Tensor,    # (K, n, 1)
    b_path: torch.Tensor,    # (K, n, 1)
    t_seq: list,
    device: str = "cuda",
    tau_gamma2: float = 1.0,
    max_batch_gamma: int = 256,
    show_progress: bool = True,
    prog_desc: str = "FLARE projection (full Hessian)",
) -> torch.Tensor:
    """
    Replay FLARE projection on a fixed x_t path using full/subnet Hessian γ².
    Returns u_proj (n,).
    """
    abar    = abar.to(device)
    xt_path = xt_path.to(device)
    a_path  = a_path.to(device)
    b_path  = b_path.to(device)

    L               = get_cholesky(la_full, device)
    params, buffers, keys, *_ = pack_params(wrapped, la_full.backend)
    idx_sub         = getattr(la_full.backend, "subnetwork_indices", None)

    K, n, d         = xt_path.shape
    Var_x0_proj     = torch.zeros(n, d, device=device)
    cum_a           = torch.ones(n, 1, device=device, dtype=xt_path.dtype)

    iters = range(K)
    if show_progress:
        from tqdm import tqdm
        iters = tqdm(iters, desc=prog_desc)

    for k in iters:
        t_int   = int(t_seq[-1 - k])
        t_idx   = torch.full((n,), t_int, device=device, dtype=torch.long)
        xt      = xt_path[k]
        a       = a_path[k]
        b       = b_path[k]

        gamma2_t = gamma2_full(
            wrapped=wrapped, L=L,
            params=params, buffers=buffers, keys=keys,
            xt=xt, t_idx=t_idx, abar=abar,
            max_batch=max_batch_gamma,
            idx_sub=idx_sub,
        ).clamp_min(0.0) * tau_gamma2

        Var_x0_proj = Var_x0_proj + (cum_a ** 2) * (b ** 2) * gamma2_t
        cum_a       = cum_a * a

    return Var_x0_proj.sum(1)


# ---------------------------------------------------------------------------
# Full BayesDiff + FLARE on a fresh path (full/subnet γ²)
# Used for the sines experiment where full Hessian is feasible
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_flare_full(
    diffusion,
    model,                   # ScoreNet (ema_for_laplace)
    la_full,
    wrapped,
    seq: list,
    abar: torch.Tensor,
    ls_mu: torch.Tensor,
    ls_sd: torch.Tensor,
    n: int = 2000,
    device: str = "cuda",
    tau_gamma2: float = 1.0,
    max_batch_gamma: int = 256,
    rng: Optional[torch.Generator] = None,
    xt0: Optional[torch.Tensor] = None,
    show_progress: bool = True,
    print_summary: bool = True,
    # ablation controls
    drop_step_var: Optional[int] = None,
    drop_what: str = "injection",
) -> tuple:
    """
    BayesDiff epistemic recursion + FLARE projection using full/subnet γ².

    Returns:
        x0        : (n, data_dim)
        u_ep      : (n,) epistemic recursion (t1+t3)
        u_proj    : (n,) FLARE projection
        history   : dict of per-step diagnostics
    """
    from tqdm import tqdm

    betas  = diffusion.betas.to(device)
    abar   = abar.to(device)
    ls_mu  = ls_mu.to(device)
    ls_sd  = ls_sd.to(device)

    L               = get_cholesky(la_full, device)
    params, buffers, keys, *_ = pack_params(wrapped, la_full.backend)
    idx_sub         = getattr(la_full.backend, "subnetwork_indices", None)

    D_out  = wrapped.net.out.out_features
    xt     = xt0.clone() if xt0 is not None else (
        torch.randn(n, D_out, device=device, generator=rng)
        if rng is not None else torch.randn(n, D_out, device=device)
    )

    Var_xt_ep   = torch.zeros_like(xt)
    Var_x0_proj = torch.zeros_like(xt)
    cum_a       = torch.ones(n, 1, device=device, dtype=xt.dtype)

    history = {
        "t": [], "gamma_trace": [], "bgamma_trace": [],
        "ep_trace": [], "proj_trace": [],
        "cum_a2_msq": [], "w_step": [],
        "drop_step_var": drop_step_var, "drop_what": drop_what,
        "dropped_steps": [], "drop_what_invalid": [],
    }
    gamma_parts = {"gamma_grads": 0.0, "gamma_solve": 0.0}

    iters = range(len(seq) - 1, 0, -1)
    if show_progress:
        iters = tqdm(iters, desc="FLARE (full Hessian)")

    for k in iters:
        t_int  = int(seq[k])
        t_prev = int(seq[k - 1])

        t_idx  = torch.full((n,), t_int,  device=device, dtype=torch.long)
        t_idxm = torch.full((n,), t_prev, device=device, dtype=torch.long)

        abar_t   = _compute_alpha(betas, t_idx)
        abar_tm1 = _compute_alpha(betas, t_idxm)
        beta_t   = 1.0 - (abar_t / abar_tm1)
        alpha_t  = 1.0 - beta_t

        a = 1.0 / torch.sqrt(alpha_t + 1e-12)
        b = beta_t / (torch.sqrt(alpha_t + 1e-12) * torch.sqrt(1.0 - abar_t + 1e-12))

        sqrt_abar_t     = torch.sqrt(abar_t.clamp_min(1e-12))
        sqrt_one_m_abar = torch.sqrt((1.0 - abar_t).clamp_min(1e-12))
        denom2          = (1.0 - abar_t).clamp_min(1e-12)
        c1 = (torch.sqrt(abar_tm1) * beta_t) / denom2
        c2 = (torch.sqrt(alpha_t) * (1.0 - abar_tm1)) / denom2
        beta_t_tilde = ((1.0 - abar_tm1) / denom2) * beta_t.clamp_min(0.0)

        gamma2_t = gamma2_full(
            wrapped=wrapped, L=L,
            params=params, buffers=buffers, keys=keys,
            xt=xt, t_idx=t_idx, abar=abar,
            max_batch=max_batch_gamma,
            timers=gamma_parts, idx_sub=idx_sub,
        ).clamp_min(0.0) * tau_gamma2

        # ablation hook
        if drop_step_var is not None and t_int == int(drop_step_var):
            history["dropped_steps"].append(t_int)
            if drop_what == "gamma":
                gamma2_t = torch.zeros_like(gamma2_t)
            elif drop_what == "injection":
                b = torch.zeros_like(b)
            elif drop_what == "both":
                gamma2_t = torch.zeros_like(gamma2_t)
                b = torch.zeros_like(b)
            else:
                history["drop_what_invalid"].append(drop_what)

        # logging (before update)
        cum_a2_msq   = (cum_a.squeeze(1) ** 2).mean().item()
        inj_per_samp = ((b ** 2) * gamma2_t).sum(1) / float(D_out)
        w_step       = ((cum_a.squeeze(1) ** 2) * inj_per_samp).mean().item()

        history["t"].append(t_int)
        history["gamma_trace"].append((gamma2_t.sum(1) / D_out).mean().item())
        history["bgamma_trace"].append(inj_per_samp.mean().item())
        history["cum_a2_msq"].append(cum_a2_msq)
        history["w_step"].append(w_step)

        # update cumulatives
        Var_x0_proj = Var_x0_proj + (cum_a ** 2) * (b ** 2) * gamma2_t
        Var_xt_ep   = ((a ** 2).expand_as(Var_xt_ep) * Var_xt_ep
                       + (b ** 2).expand_as(gamma2_t) * gamma2_t).clamp_min(0.0)
        cum_a       = cum_a * a

        history["ep_trace"].append((Var_xt_ep.sum(1) / D_out).mean().item())
        history["proj_trace"].append((Var_x0_proj.sum(1) / D_out).mean().item())

        # DDPM step
        t_code  = timecode_from_tindex(t_idx, abar, ls_mu, ls_sd)
        mu_eps  = model(xt, t_code)
        eps_t   = mu_eps + randn_like_gen(mu_eps, rng) * torch.sqrt(gamma2_t)
        x0_pred = (xt - sqrt_one_m_abar * eps_t) / sqrt_abar_t
        x0_pred = x0_pred.clamp_(-2.0, 2.0)
        xt      = c1 * x0_pred + c2 * xt + torch.sqrt(beta_t_tilde) * randn_like_gen(xt, rng)

    if print_summary:
        gtot = gamma_parts["gamma_grads"] + gamma_parts["gamma_solve"]
        if gtot > 0:
            print(f"γ² grads: {gamma_parts['gamma_grads']:.3f}s  solves: {gamma_parts['gamma_solve']:.3f}s")

    return xt, Var_xt_ep.sum(1), Var_x0_proj.sum(1), history


















