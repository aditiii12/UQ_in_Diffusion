"""
BayesDiff with Last-Layer Laplace Approximation (LLLA, diagonal Hessian).

Propagates epistemic uncertainty through the DDPM reverse chain via:
  V_{t-1} = a_t^2 * V_t + b_t^2 * γ_t^2       (epistemic recursion, t1+t3)
  Var_x0_proj = Σ_t (Π_{s<t} a_s)^2 * b_t^2 * γ_t^2  (transported sum)
"""

from __future__ import annotations

import torch
from typing import Optional

from uqdiff.laplace.llla import llla_gamma2_diag_lastlayer
from uqdiff.diffusion.timecode import timecode_from_tnorm
from uqdiff.utils.rng import randn_like_gen


class DiffusionShim:
    """Minimal shim to pass betas + T to BayesDiff samplers."""
    def __init__(self, betas: torch.Tensor):
        self.betas = betas
        self.num_timesteps = betas.numel()


def compute_alpha(beta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Return cumulative alpha ᾱ_t with shape (B, 1)."""
    beta = torch.cat([torch.zeros(1, device=beta.device, dtype=beta.dtype), beta], dim=0)
    return (1 - beta).cumprod(0).index_select(0, t.long() + 1).view(-1, 1)


@torch.no_grad()
def _estimate_cov_and_Eeps_batch(
    score_model,
    E_xt: torch.Tensor,
    Var_xt: torch.Tensor,
    t_norm: torch.Tensor,
    abar: torch.Tensor,
    ls_mu: torch.Tensor,
    ls_sd: torch.Tensor,
    S: int = 128,
    antithetic: bool = True,
    mc_gen: Optional[torch.Generator] = None,
):
    """Monte-Carlo estimate of Cov[x_t, ε_t] and E[ε_t]."""
    B, d = E_xt.shape
    std = torch.sqrt(Var_xt.clamp_min(0) + 1e-12)

    H = max(1, S // 2)
    z = torch.randn(H, B, d, device=E_xt.device, generator=mc_gen)
    if antithetic:
        z = torch.cat([z, -z], dim=0)
        if z.shape[0] < S:
            z = torch.cat([z, torch.randn(1, B, d, device=E_xt.device, generator=mc_gen)], dim=0)

    S_eff = z.shape[0]
    xti   = E_xt.unsqueeze(0) + std.unsqueeze(0) * z   # (S_eff, B, d)
    tti   = t_norm.unsqueeze(0).expand(S_eff, B)

    t_code = timecode_from_tnorm(tti.reshape(S_eff * B), abar, ls_mu, ls_sd)
    eps_i  = score_model(xti.reshape(S_eff * B, d), t_code).reshape(S_eff, B, d)

    E_eps  = eps_i.mean(0)
    Cov    = (xti * eps_i).mean(0) - E_xt * E_eps
    if Var_xt.max() < 1e-10:
        Cov.zero_()
    return Cov, E_eps


@torch.no_grad()
def sample_bayesdiff_samepath(
    diffusion: DiffusionShim,
    model,
    la,
    seq,
    abar: torch.Tensor,
    ls_mu: torch.Tensor,
    ls_sd: torch.Tensor,
    n: int,
    data_dim: int,
    device: str = "cuda",
    use_cov: bool = True,
    include_t4: bool = True,
    S_cov: int = 64,
    tau_gamma2: float = 1.0,
    rng: Optional[torch.Generator] = None,
    rng_mc: Optional[torch.Generator] = None,
    xt0: Optional[torch.Tensor] = None,
    return_all: bool = False,
    return_path: bool = False,
    show_progress: bool = True,
):
    """
    BayesDiff reverse sampler with LLLA diagonal uncertainty propagation.

    Variance recursion terms:
      t1: a^2 * V_{t}       (transport)
      t2: 2 * a * (-b) * Cov[x_t, ε_t]   (MC covariance, optional)
      t3: b^2 * γ^2         (epistemic injection)
      t4: β̃_t               (algorithmic noise, optional)

    Parameters
    ----------
    diffusion   : DiffusionShim
    model       : ScoreNet with .forward and .forward_with_feat
    la          : fitted LLLA Laplace object
    seq         : list of integer timestep indices (e.g. list(range(T)))
    abar        : (T,) cumulative alpha schedule
    ls_mu/ls_sd : logSNR normalization stats
    n           : number of samples
    data_dim    : dimensionality (sequence length)
    use_cov     : include t2 covariance term
    include_t4  : include algorithmic noise t4 in variance recursion
    return_path : also return (xt_path, a_path, b_path) for replay

    Returns
    -------
    (x0, u_bayes, u_ep, u_proj) or extended tuple if return_all / return_path
    """
    from tqdm import tqdm

    betas  = diffusion.betas.to(device)
    alphas = (1.0 - betas).to(device)
    T      = diffusion.num_timesteps
    abar   = abar.to(device)
    ls_mu  = ls_mu.to(device)
    ls_sd  = ls_sd.to(device)

    # initialise path
    if xt0 is not None:
        xt = xt0.to(device=device, dtype=betas.dtype).clone()
    else:
        xt = torch.randn(n, data_dim, device=device, dtype=betas.dtype, generator=rng)

    E_xt        = xt.clone()
    Var_xt_full = torch.zeros_like(xt)
    Var_xt_ep   = torch.zeros_like(xt)
    Var_x0_proj = torch.zeros_like(xt)
    cum_a       = torch.ones(n, 1, device=device, dtype=xt.dtype)

    xt_path, a_path, b_path = [], [], []

    iters = range(len(seq) - 1, 0, -1)
    if show_progress:
        iters = tqdm(iters, desc="BayesDiff LLLA")

    for k in iters:
        t_int  = int(seq[k])
        t_prev = int(seq[k - 1])

        t_idx  = torch.full((n,), t_int,  device=device, dtype=torch.long)
        t_idxm = torch.full((n,), t_prev, device=device, dtype=torch.long)

        abar_t   = compute_alpha(betas, t_idx)
        abar_tm1 = compute_alpha(betas, t_idxm)
        beta_t   = 1.0 - (abar_t / abar_tm1)
        alpha_t  = 1.0 - beta_t

        sqrt_abar_t     = torch.sqrt(abar_t.clamp_min(1e-12))
        sqrt_one_m_abar = torch.sqrt((1.0 - abar_t).clamp_min(1e-12))
        denom2          = (1.0 - abar_t).clamp_min(1e-12)
        c1 = (torch.sqrt(abar_tm1) * beta_t) / denom2
        c2 = (torch.sqrt(alpha_t) * (1.0 - abar_tm1)) / denom2

        beta_t_tilde = ((1.0 - abar_tm1) / denom2) * beta_t
        beta_t_tilde = beta_t_tilde.clamp_min(0.0)

        a = 1.0 / torch.sqrt(alpha_t + 1e-12)
        b = beta_t / (torch.sqrt(alpha_t + 1e-12) * torch.sqrt(1.0 - abar_t + 1e-12))

        if return_path:
            xt_path.append(xt.detach().clone())
            a_path.append(a.detach().clone())
            b_path.append(b.detach().clone())

        # γ² from LLLA
        mu_eps, gamma2_t = llla_gamma2_diag_lastlayer(la, model, xt, t_idx, abar, ls_mu, ls_sd)
        gamma2_t = (gamma2_t * float(tau_gamma2)).clamp_min(0.0)

        # epistemic projection accumulator
        Var_x0_proj = Var_x0_proj + (cum_a ** 2) * (b ** 2) * gamma2_t
        cum_a = cum_a * a

        # DDPM path update
        eps_t   = mu_eps + randn_like_gen(mu_eps, rng) * torch.sqrt(gamma2_t)
        x0_pred = (xt - sqrt_one_m_abar * eps_t) / sqrt_abar_t
        x0_pred = x0_pred.clamp_(-2.0, 2.0)

        if t_int > 0:
            xt = c1 * x0_pred + c2 * xt + torch.sqrt(beta_t_tilde) * randn_like_gen(xt, rng)
        else:
            xt = c1 * x0_pred + c2 * xt

        # MC covariance (t2)
        t_norm_prev = (t_idxm.float() / float(T)).view(-1)
        Cov, E_eps_mc = _estimate_cov_and_Eeps_batch(
            score_model=model,
            E_xt=E_xt,
            Var_xt=Var_xt_full,
            t_norm=t_norm_prev,
            abar=abar,
            ls_mu=ls_mu,
            ls_sd=ls_sd,
            S=S_cov,
            mc_gen=rng_mc,
        )
        E_xt = a.expand_as(E_xt) * E_xt - b.expand_as(E_xt) * E_eps_mc

        a2, b2 = a ** 2, b ** 2

        # epistemic recursion (t1 + t3)
        Var_xt_ep = (a2.expand_as(Var_xt_ep) * Var_xt_ep
                     + b2.expand_as(gamma2_t) * gamma2_t).clamp_min(0.0)

        # full BayesDiff recursion
        Var_xt_full = a2.expand_as(Var_xt_full) * Var_xt_full
        if use_cov:
            Var_xt_full = Var_xt_full + (2 * (a * (-b))).expand_as(Cov) * Cov
        Var_xt_full = Var_xt_full + b2.expand_as(gamma2_t) * gamma2_t
        if include_t4:
            Var_xt_full = Var_xt_full + beta_t_tilde.expand_as(Var_xt_full)
        Var_xt_full = Var_xt_full.clamp_min(0.0)

    # scalar uncertainty scores
    u_ep    = Var_xt_ep.sum(dim=1)
    u_bayes = Var_xt_full.sum(dim=1)
    u_proj  = Var_x0_proj.sum(dim=1)

    if return_path:
        xt_path = torch.stack(xt_path, dim=0)
        a_path  = torch.stack(a_path,  dim=0)
        b_path  = torch.stack(b_path,  dim=0)
        return xt, u_bayes, u_ep, u_proj, xt_path, a_path, b_path

    if return_all:
        return xt, u_bayes, u_ep, u_proj, Var_xt_full, Var_xt_ep, Var_x0_proj
    return xt, u_bayes, u_ep, u_proj














