"""
uqdiff/laplace/bayesdiff.py
----------------------------
BayesDiff sampler using LLLA (last-layer diagonal) γ².

Implements the full variance recursion:
    V_{t-1} = a² V_t + 2ab Cov[x_t, ε_t] + b² γ²_t + β̃_t (optional)

where γ²_t comes from the diagonal last-layer Laplace posterior.

Reference: Xiao et al., "BayesDiff: Estimating Pixel-wise Uncertainty
in Diffusion via Bayesian Inference" (2023)
"""

from __future__ import annotations
from typing import Optional

import torch

from uqdiff.utils import timecode_from_tindex, timecode_from_tnorm, randn_like_gen
from uqdiff.laplace.precision import gamma2_diag


# ---------------------------------------------------------------------------
# Schedule helper (matches compute_alpha from notebooks)
# ---------------------------------------------------------------------------

def _compute_alpha(betas: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Return ᾱ_t with shape (B, 1)."""
    betas = torch.cat([torch.zeros(1, device=betas.device, dtype=betas.dtype), betas])
    return (1 - betas).cumprod(0).index_select(0, t.long() + 1).view(-1, 1)


# ---------------------------------------------------------------------------
# MC Cov[x_t, eps_t] estimator (t2 term)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _estimate_cov_and_mean_eps(
    score_model,
    E_xt: torch.Tensor,       # (B, data_dim)
    Var_xt: torch.Tensor,     # (B, data_dim)
    t_norm: torch.Tensor,     # (B,)
    abar: torch.Tensor,
    ls_mu: torch.Tensor,
    ls_sd: torch.Tensor,
    S: int = 64,
    antithetic: bool = True,
    mc_gen: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Monte Carlo estimate of Cov[x_t, eps_t] and E[eps_t].
    Returns (Cov, E_eps) both (B, data_dim).
    """
    B, d  = E_xt.shape
    std   = torch.sqrt(Var_xt.clamp_min(0) + 1e-12)

    if antithetic:
        H = max(1, S // 2)
        z = torch.randn(H, B, d, device=E_xt.device, generator=mc_gen)
        z = torch.cat([z, -z], dim=0)
        if z.shape[0] < S:
            z = torch.cat([z, torch.randn(1, B, d, device=E_xt.device, generator=mc_gen)], dim=0)
    else:
        z = torch.randn(S, B, d, device=E_xt.device, generator=mc_gen)

    S_eff = z.shape[0]
    xti   = E_xt.unsqueeze(0) + std.unsqueeze(0) * z        # (S, B, d)
    tti   = t_norm.unsqueeze(0).expand(S_eff, B)            # (S, B)

    t_code = timecode_from_tnorm(
        tti.reshape(S_eff * B), abar, ls_mu, ls_sd
    )
    eps_i = score_model(xti.reshape(S_eff * B, d), t_code).reshape(S_eff, B, d)

    E_eps  = eps_i.mean(0)
    E_xeps = (xti * eps_i).mean(0)
    Cov    = E_xeps - E_xt * E_eps

    if Var_xt.max() < 1e-10:
        Cov.zero_()
    return Cov, E_eps


# ---------------------------------------------------------------------------
# BayesDiff sampler (LLLA)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_bayesdiff(
    diffusion,              # DiffusionShim
    model,                  # ScoreNet with .forward_with_feat
    la,                     # fitted LLLA Laplace object
    seq,                    # list of int t indices e.g. list(range(T))
    abar: torch.Tensor,
    ls_mu: torch.Tensor,
    ls_sd: torch.Tensor,
    n: int = 2048,
    device: str = "cuda",
    use_cov: bool = True,       # include t2 Cov term
    include_t4: bool = True,    # include alg noise β̃ term
    S_cov: int = 64,            # MC samples for Cov estimate
    tau_gamma2: float = 1.0,    # scale on γ² (tempering)
    rng: Optional[torch.Generator] = None,
    rng_mc: Optional[torch.Generator] = None,
    xt0: Optional[torch.Tensor] = None,   # shared x_T start for comparison
    return_path: bool = False,            # if True, also return (xt_path, a_path, b_path)
    show_progress: bool = True,
) -> tuple:
    """
    Full BayesDiff variance recursion with LLLA γ².

    Returns (x0, u_bayes, u_ep, u_proj) by default.
    If return_path=True: (x0, u_bayes, u_ep, u_proj, xt_path, a_path, b_path)

    u_bayes : full BayesDiff (t1 + t2? + t3 + t4?)
    u_ep    : epistemic recursion only (t1 + t3)
    u_proj  : FLARE projection (closed-form epistemic)
    """
    betas  = diffusion.betas.to(device)
    alphas = (1.0 - betas).to(device)
    T      = diffusion.num_timesteps
    abar   = abar.to(device)
    ls_mu  = ls_mu.to(device)
    ls_sd  = ls_sd.to(device)

    xt      = xt0.clone() if xt0 is not None else torch.randn(n, abar.shape[0] if False else model.out.out_features, device=device)
    # infer data_dim from model
    data_dim = model.out.out_features

    if xt0 is not None:
        xt = xt0.clone()
    else:
        xt = torch.randn(n, data_dim, device=device, generator=rng) if rng is not None else torch.randn(n, data_dim, device=device)

    E_xt        = xt.clone()
    Var_xt_full = torch.zeros_like(xt)
    Var_xt_ep   = torch.zeros_like(xt)
    Var_x0_proj = torch.zeros_like(xt)
    cum_a       = torch.ones(n, 1, device=device, dtype=xt.dtype)

    if return_path:
        xt_path = torch.zeros(len(seq) - 1, n, data_dim, device=device)
        a_path  = torch.zeros(len(seq) - 1, n, 1, device=device)
        b_path  = torch.zeros(len(seq) - 1, n, 1, device=device)

    iters = range(len(seq) - 1, 0, -1)
    if show_progress:
        from tqdm import tqdm
        iters = tqdm(iters, desc="BayesDiff (LLLA)")

    step = 0
    for k in iters:
        t_int  = int(seq[k])
        t_prev = int(seq[k - 1])

        t_idx  = torch.full((n,), t_int,  device=device, dtype=torch.long)
        t_idxm = torch.full((n,), t_prev, device=device, dtype=torch.long)

        abar_t   = _compute_alpha(betas, t_idx)
        abar_tm1 = _compute_alpha(betas, t_idxm)
        beta_t   = 1.0 - (abar_t / abar_tm1)
        alpha_t  = 1.0 - beta_t

        sqrt_abar_t     = torch.sqrt(abar_t.clamp_min(1e-12))
        sqrt_one_m_abar = torch.sqrt((1.0 - abar_t).clamp_min(1e-12))
        denom2          = (1.0 - abar_t).clamp_min(1e-12)

        c1 = (torch.sqrt(abar_tm1) * beta_t) / denom2
        c2 = (torch.sqrt(alpha_t) * (1.0 - abar_tm1)) / denom2
        beta_t_tilde = ((1.0 - abar_tm1) / denom2) * beta_t.clamp_min(0.0)

        a = 1.0 / torch.sqrt(alpha_t + 1e-12)
        b = beta_t / (torch.sqrt(alpha_t + 1e-12) * torch.sqrt(1.0 - abar_t + 1e-12))

        # γ² from LLLA
        mu_eps, gamma2_t = gamma2_diag(la, model, xt, t_idx, abar, ls_mu, ls_sd)
        gamma2_t = (gamma2_t * tau_gamma2).clamp_min(0.0)

        # FLARE projection accumulation
        Var_x0_proj = Var_x0_proj + (cum_a ** 2) * (b ** 2) * gamma2_t
        cum_a       = cum_a * a

        # DDPM step
        eps_t   = mu_eps + randn_like_gen(mu_eps, rng) * torch.sqrt(gamma2_t)
        x0_pred = (xt - sqrt_one_m_abar * eps_t) / sqrt_abar_t
        x0_pred = x0_pred.clamp_(-2.0, 2.0)
        noise   = randn_like_gen(xt, rng) if t_int > 0 else torch.zeros_like(xt)
        xt      = c1 * x0_pred + c2 * xt + torch.sqrt(beta_t_tilde) * noise

        if return_path:
            xt_path[step] = xt
            a_path[step]  = a
            b_path[step]  = b
            step += 1

        # MC Cov
        t_norm_prev = (t_idxm.float() / float(T)).view(-1)
        Cov, E_eps_mc = _estimate_cov_and_mean_eps(
            score_model=model,
            E_xt=E_xt,
            Var_xt=Var_xt_full,
            t_norm=t_norm_prev,
            abar=abar, ls_mu=ls_mu, ls_sd=ls_sd,
            S=S_cov, mc_gen=rng_mc,
        )

        # mean recursion
        E_xt = a.expand_as(E_xt) * E_xt - b.expand_as(E_xt) * E_eps_mc

        a2, b2, ab = a ** 2, b ** 2, a * (-b)

        # epistemic recursion (t1 + t3)
        Var_xt_ep = (a2.expand_as(Var_xt_ep) * Var_xt_ep
                     + b2.expand_as(gamma2_t) * gamma2_t).clamp_min(0.0)

        # full recursion (t1 + t2? + t3 + t4?)
        Var_xt_full = a2.expand_as(Var_xt_full) * Var_xt_full
        if use_cov:
            Var_xt_full = Var_xt_full + (2 * ab).expand_as(Cov) * Cov
        Var_xt_full = Var_xt_full + b2.expand_as(gamma2_t) * gamma2_t
        if include_t4:
            Var_xt_full = Var_xt_full + beta_t_tilde.expand_as(Var_xt_full)
        Var_xt_full = Var_xt_full.clamp_min(0.0)

    x0      = xt
    u_bayes = Var_xt_full.sum(1)
    u_ep    = Var_xt_ep.sum(1)
    u_proj  = Var_x0_proj.sum(1)

    if return_path:
        return x0, u_bayes, u_ep, u_proj, xt_path, a_path, b_path
    return x0, u_bayes, u_ep, u_proj