"""
BayesDiff with Full-Hessian Laplace Approximation.

Three entry points:
  1. sample_bayesdiff_fullH_vmap   : joint sampling + uncertainty propagation
  2. project_fullH_on_fixed_path   : epistemic projection only (fixed x_t path)
  3. bayesdiff_fullH_total_on_fixed_path : full (t1+t2+t3+t4) replay on fixed path
"""

from __future__ import annotations

import torch
from typing import Optional

from uqdiff.fisher.jacobian import (
    pack_params_and_buffers_backend_order,
    robust_chol,
    gamma2_full_laplace_vmap,
)
from uqdiff.laplace.bayesdiff_llla import (
    compute_alpha,
    _estimate_cov_and_Eeps_batch,
    DiffusionShim,
)
from uqdiff.diffusion.timecode import timecode_from_tindex, timecode_from_tnorm
from uqdiff.utils.rng import randn_like_gen


@torch.no_grad()
def sample_bayesdiff_fullH_vmap(
    diffusion: DiffusionShim,
    model,
    la_full,
    wrapped,
    seq,
    abar: torch.Tensor,
    ls_mu: torch.Tensor,
    ls_sd: torch.Tensor,
    n: int,
    device: str = "cuda",
    tau_gamma2: float = 1.0,
    max_batch_gamma: int = 256,
    rng: Optional[torch.Generator] = None,
    xt0: Optional[torch.Tensor] = None,
    show_progress: bool = True,
    print_summary: bool = True,
):
    """
    Full-Hessian BayesDiff: joint DDPM sampling with per-step γ² = diag(J Σ Jᵀ).

    Tracks:
      Var_xt_ep   : epistemic recursion (t1 + t3)
      Var_x0_proj : transported epistemic sum to x0

    Returns
    -------
    x0, u_term3, u_proj, Var_x0_last, Var_x0_proj, u_ep, history
    """
    from tqdm import tqdm

    betas  = diffusion.betas.to(device)
    T      = diffusion.num_timesteps
    abar   = abar.to(device)
    ls_mu  = ls_mu.to(device)
    ls_sd  = ls_sd.to(device)

    # Cholesky of posterior precision
    P = la_full.posterior_precision
    if not torch.is_tensor(P):
        P = P.to_dense()
    L = robust_chol(P.to(device))

    params0, buffers, keys, *_ = pack_params_and_buffers_backend_order(wrapped, la_full.backend)

    D_out = wrapped.net.out.out_features
    xt = xt0 if xt0 is not None else torch.randn(n, D_out, device=device, generator=rng)

    Var_xt_last = torch.zeros_like(xt)
    Var_x0_proj = torch.zeros_like(xt)
    Var_xt_ep   = torch.zeros_like(xt)
    cum_a = torch.ones(n, 1, device=device, dtype=xt.dtype)

    history = {"t": [], "gamma_trace": [], "bgamma_trace": [], "ep_trace": [], "proj_trace": [],
               "cum_a2_msq": [], "w_step": []}
    gamma_parts = {"gamma_grads": 0.0, "gamma_solve": 0.0}

    iters = range(len(seq) - 1, 0, -1)
    if show_progress:
        iters = tqdm(iters, desc="BayesDiff (full Laplace)")

    for k in iters:
        t_int  = int(seq[k])
        t_prev = int(seq[k - 1])

        t_idx  = torch.full((n,), t_int,  device=device, dtype=torch.long)
        t_idxm = torch.full((n,), t_prev, device=device, dtype=torch.long)

        abar_t   = compute_alpha(betas, t_idx).view(-1, 1)
        abar_tm1 = compute_alpha(betas, t_idxm).view(-1, 1)
        beta_t   = 1.0 - (abar_t / abar_tm1)
        alpha_t  = 1.0 - beta_t

        sqrt_abar_t     = torch.sqrt(abar_t.clamp_min(1e-12))
        sqrt_one_m_abar = torch.sqrt((1.0 - abar_t).clamp_min(1e-12))
        denom2          = (1.0 - abar_t).clamp_min(1e-12)
        c1 = (torch.sqrt(abar_tm1) * beta_t) / denom2
        c2 = (torch.sqrt(alpha_t) * (1.0 - abar_tm1)) / denom2
        beta_t_tilde = ((1.0 - abar_tm1) / denom2) * beta_t
        a = 1.0 / torch.sqrt(alpha_t + 1e-12)
        b = beta_t / (torch.sqrt(alpha_t + 1e-12) * torch.sqrt(1.0 - abar_t + 1e-12))

        # γ² at (x_t, t)
        gamma2_t = gamma2_full_laplace_vmap(
            wrapped=wrapped, L=L, params=params0, buffers=buffers, keys=keys,
            xt=xt, t_idx=t_idx, abar=abar,
            max_batch=max_batch_gamma, timers=gamma_parts,
        ).clamp_min(0.0) * float(tau_gamma2)

        # history logging
        cum_a2_msq = (cum_a.squeeze(1) ** 2).mean().item()
        inj_per_sample = ((b ** 2) * gamma2_t).sum(dim=1) / float(D_out)
        history["t"].append(t_int)
        history["cum_a2_msq"].append(cum_a2_msq)
        history["w_step"].append(((cum_a.squeeze(1) ** 2) * inj_per_sample).mean().item())
        history["gamma_trace"].append((gamma2_t.sum(1) / D_out).mean().item())
        history["bgamma_trace"].append(inj_per_sample.mean().item())

        # cumulatives
        Var_x0_proj = Var_x0_proj + (cum_a ** 2) * (b ** 2) * gamma2_t
        Var_xt_ep   = ((a ** 2).expand_as(Var_xt_ep) * Var_xt_ep
                       + (b ** 2).expand_as(gamma2_t) * gamma2_t).clamp_min(0.0)
        cum_a = cum_a * a
        Var_xt_last = ((b ** 2).expand_as(gamma2_t) * gamma2_t).clamp_min(0.0)

        history["ep_trace"].append((Var_xt_ep.sum(1) / D_out).mean().item())
        history["proj_trace"].append((Var_x0_proj.sum(1) / D_out).mean().item())

        # DDPM path update
        t_code = timecode_from_tindex(t_idx, abar, ls_mu, ls_sd)
        mu_eps = model(xt, t_code)
        eps_t  = mu_eps + randn_like_gen(mu_eps, rng) * torch.sqrt(gamma2_t.clamp_min(0.0))
        x0_pred = (xt - sqrt_one_m_abar * eps_t) / sqrt_abar_t
        x0_pred = x0_pred.clamp_(-2.0, 2.0)
        xt = c1 * x0_pred + c2 * xt + torch.sqrt(beta_t_tilde) * randn_like_gen(xt, rng)

    if print_summary:
        g = gamma_parts["gamma_grads"] + gamma_parts["gamma_solve"]
        if g > 0:
            print(f"γ² grads: {gamma_parts['gamma_grads']:.2f}s  solves: {gamma_parts['gamma_solve']:.2f}s")

    return (
        xt,
        Var_xt_last.sum(1),
        Var_x0_proj.sum(1),
        Var_xt_last,
        Var_x0_proj,
        Var_xt_ep.sum(1),
        history,
    )


@torch.no_grad()
def project_fullH_on_fixed_path(
    wrapped,
    la_full,
    abar: torch.Tensor,
    xt_path: torch.Tensor,
    a_path: torch.Tensor,
    b_path: torch.Tensor,
    t_seq,
    device: str = "cuda",
    tau_gamma2: float = 1.0,
    max_batch_gamma: int = 256,
    show_progress: bool = True,
    prog_desc: str = "Full-Hessian projection (same path)",
) -> torch.Tensor:
    """
    Replay the epistemic projection Σ_t (Πa)² b² γ²_fullH on a pre-recorded x_t path.

    Returns
    -------
    u_proj_fullH : (n,) summed projection variance
    """
    from tqdm import tqdm

    P = la_full.posterior_precision
    if not torch.is_tensor(P):
        P = P.to_dense()
    L = robust_chol(P.to(device))
    params0, buffers, keys, *_ = pack_params_and_buffers_backend_order(wrapped, la_full.backend)

    K, n, d = xt_path.shape
    Var_x0_proj = torch.zeros(n, d, device=device, dtype=xt_path.dtype)
    cum_a = torch.ones(n, 1, device=device, dtype=xt_path.dtype)

    iters = range(K)
    if show_progress:
        iters = tqdm(iters, desc=prog_desc, leave=False)

    for k in iters:
        t_int = int(t_seq[-1 - k])
        t_idx = torch.full((n,), t_int, device=device, dtype=torch.long)
        xt = xt_path[k].to(device)
        a  = a_path[k].to(device)
        b  = b_path[k].to(device)

        gamma2_t = gamma2_full_laplace_vmap(
            wrapped=wrapped, L=L, params=params0, buffers=buffers, keys=keys,
            xt=xt, t_idx=t_idx, abar=abar.to(device), max_batch=max_batch_gamma,
        ).clamp_min(0.0) * float(tau_gamma2)

        Var_x0_proj = Var_x0_proj + (cum_a ** 2) * (b ** 2) * gamma2_t
        cum_a = cum_a * a

    return Var_x0_proj.sum(dim=1)


@torch.no_grad()
def bayesdiff_fullH_total_on_fixed_path(
    wrapped,
    la_full,
    diffusion: DiffusionShim,
    abar: torch.Tensor,
    ls_mu: torch.Tensor,
    ls_sd: torch.Tensor,
    xt_path: torch.Tensor,
    a_path: torch.Tensor,
    b_path: torch.Tensor,
    t_seq,
    device: str = "cuda",
    use_cov: bool = True,
    include_t4: bool = True,
    S_cov: int = 64,
    tau_gamma2: float = 1.0,
    rng_mc: Optional[torch.Generator] = None,
    show_progress: bool = True,
):
    """
    Full-Hessian BayesDiff total variance (t1+t2+t3+t4) replayed on a fixed x_t path.

    Returns
    -------
    x0_fixed        : (n, d)  final samples (same as xt_path[-1])
    u_bayes_fullH   : (n,)    total BayesDiff variance
    u_ep_fullH      : (n,)    epistemic-only (t1+t3)
    u_proj_fullH    : (n,)    epistemic projection
    """
    from tqdm import tqdm

    betas  = diffusion.betas.to(device)
    T      = diffusion.num_timesteps
    abar   = abar.to(device)
    ls_mu  = ls_mu.to(device)
    ls_sd  = ls_sd.to(device)

    P = la_full.posterior_precision
    if not torch.is_tensor(P):
        P = P.to_dense()
    L = robust_chol(P.to(device))
    params0, buffers, keys, *_ = pack_params_and_buffers_backend_order(wrapped, la_full.backend)

    xt_path = xt_path.to(device)
    a_path  = a_path.to(device)
    b_path  = b_path.to(device)
    K, n, d = xt_path.shape

    E_xt_full   = xt_path[0].clone()
    Var_xt_full = torch.zeros_like(E_xt_full)
    Var_xt_ep   = torch.zeros_like(E_xt_full)
    Var_x0_proj = torch.zeros_like(E_xt_full)
    cum_a       = torch.ones(n, 1, device=device, dtype=xt_path.dtype)

    iters = range(K)
    if show_progress:
        iters = tqdm(iters, desc="Full-H total (replay on fixed path)", leave=False)

    for k in iters:
        t_int  = int(t_seq[-1 - k])
        t_prev = max(t_int - 1, 0)
        xt = xt_path[k]
        a  = a_path[k]
        b  = b_path[k]

        t_idx  = torch.full((n,), t_int,  device=device, dtype=torch.long)
        t_idxm = torch.full((n,), t_prev, device=device, dtype=torch.long)

        abar_t   = compute_alpha(betas, t_idx)
        abar_tm1 = compute_alpha(betas, t_idxm)
        beta_t   = 1.0 - (abar_t / abar_tm1)
        alpha_t  = 1.0 - beta_t
        denom2   = (1.0 - abar_t).clamp_min(1e-12)
        beta_t_tilde = ((1.0 - abar_tm1) / denom2) * beta_t

        gamma2_t = gamma2_full_laplace_vmap(
            wrapped=wrapped, L=L, params=params0, buffers=buffers, keys=keys,
            xt=xt, t_idx=t_idx, abar=abar, max_batch=256,
        ).clamp_min(0.0) * float(tau_gamma2)

        Var_x0_proj = Var_x0_proj + (cum_a ** 2) * (b ** 2) * gamma2_t
        cum_a = cum_a * a

        t_norm_prev = (t_idxm.float() / float(T)).view(-1)
        Cov, E_eps_mc = _estimate_cov_and_Eeps_batch(
            score_model=wrapped.net,
            E_xt=E_xt_full, Var_xt=Var_xt_full,
            t_norm=t_norm_prev,
            abar=abar, ls_mu=ls_mu, ls_sd=ls_sd,
            S=S_cov, mc_gen=rng_mc,
        )
        E_xt_full = a.expand_as(E_xt_full) * E_xt_full - b.expand_as(E_xt_full) * E_eps_mc

        a2, b2 = a ** 2, b ** 2
        Var_xt_ep   = (a2.expand_as(Var_xt_ep) * Var_xt_ep + b2.expand_as(gamma2_t) * gamma2_t).clamp_min(0.0)
        Var_xt_full = a2.expand_as(Var_xt_full) * Var_xt_full
        if use_cov:
            Var_xt_full = Var_xt_full + (2 * (a * (-b))).expand_as(Cov) * Cov
        Var_xt_full = Var_xt_full + b2.expand_as(gamma2_t) * gamma2_t
        if include_t4:
            Var_xt_full = Var_xt_full + beta_t_tilde.expand_as(Var_xt_full)
        Var_xt_full = Var_xt_full.clamp_min(0.0)

    return (
        xt_path[-1],
        Var_xt_full.sum(1),
        Var_xt_ep.sum(1),
        Var_x0_proj.sum(1),
    )














