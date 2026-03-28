import torch
from uqdiff.diffusion.timecode import timecode_from_tindex
from uqdiff.utils.rng import randn_like_gen


@torch.no_grad()
def p_sample_eps_tau(
    model,
    xt: torch.Tensor,
    t_idx: int,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    abar: torch.Tensor,
    ls_mu: torch.Tensor,
    ls_sd: torch.Tensor,
    tau: float = 1.0,
    gen=None,
) -> torch.Tensor:
    """Single DDPM reverse step with optional temperature scaling tau."""
    device, dtype = xt.device, xt.dtype
    betas  = betas.to(device=device, dtype=dtype)
    alphas = alphas.to(device=device, dtype=dtype)
    abar   = abar.to(device=device, dtype=dtype)
    ls_mu  = ls_mu.to(device=device, dtype=dtype)
    ls_sd  = ls_sd.to(device=device, dtype=dtype)

    beta_t   = betas[t_idx]
    alpha_t  = alphas[t_idx]
    abar_t   = abar[t_idx]
    abar_tm1 = abar[t_idx - 1] if t_idx > 0 else torch.as_tensor(1.0, device=device, dtype=dtype)

    B = xt.size(0)
    t_code = timecode_from_tindex(
        torch.full((B,), t_idx, device=device, dtype=torch.long),
        abar, ls_mu, ls_sd,
    )
    eps = model(xt, t_code)

    x0_pred = (xt - torch.sqrt((1.0 - abar_t).clamp_min(0.0)) * eps) / torch.sqrt(abar_t + 1e-12)
    x0_pred = x0_pred.clamp_(-2.0, 2.0)

    denom2 = (1.0 - abar_t).clamp_min(1e-12)
    c1 = (torch.sqrt(abar_tm1) * beta_t) / denom2
    c2 = (torch.sqrt(alpha_t) * (1.0 - abar_tm1)) / denom2
    mean = c1 * x0_pred + c2 * xt

    beta_t_tilde = ((1.0 - abar_tm1) / (1.0 - abar_t).clamp_min(1e-12)) * beta_t
    beta_t_tilde = beta_t_tilde.clamp_min(0.0)

    if t_idx > 0:
        return mean + float(tau) * torch.sqrt(beta_t_tilde + 1e-12) * randn_like_gen(xt, gen)
    return mean


@torch.no_grad()
def sample_ddpm(
    model,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    abar: torch.Tensor,
    ls_mu: torch.Tensor,
    ls_sd: torch.Tensor,
    T: int,
    data_dim: int,
    n: int,
    device=None,
    gen=None,
    tau: float = 1.0,
    use_eval: bool = True,
) -> torch.Tensor:
    """Full DDPM reverse chain from x_T ~ N(0,I) to x_0."""
    if device is None:
        device = next(model.parameters()).device
    was_training = model.training
    if use_eval:
        model.eval()

    dtype = next(model.parameters()).dtype
    betas  = betas.to(device, dtype=dtype)
    alphas = alphas.to(device, dtype=dtype)
    abar   = abar.to(device, dtype=dtype)
    ls_mu  = ls_mu.to(device, dtype=dtype)
    ls_sd  = ls_sd.to(device, dtype=dtype)

    xt = (
        torch.randn((n, data_dim), device=device, dtype=dtype, generator=gen)
        if gen is not None
        else torch.randn(n, data_dim, device=device, dtype=dtype)
    )

    for t in reversed(range(T)):
        xt = p_sample_eps_tau(model, xt, t, betas, alphas, abar, ls_mu, ls_sd, tau=tau, gen=gen)

    if use_eval and was_training:
        model.train()
    return xt














