import torch


def _logsnr_from_abar(a: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    a = a.clamp(min=eps, max=1 - eps)
    return torch.log(a) - torch.log1p(-a)


@torch.no_grad()
def prep_time_stats(abar: torch.Tensor):
    """Compute mean and std of logSNR schedule for normalization."""
    ls = _logsnr_from_abar(abar)
    return ls.mean(), ls.std().clamp_min(1e-6)


def time_code_from_abar(
    abar_t: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor
) -> torch.Tensor:
    """abar_t: (B,1) -> normalized logSNR code (B,)"""
    return ((_logsnr_from_abar(abar_t) - mu) / sd).squeeze(1)


def timecode_from_tindex(
    t_idx_batch: torch.Tensor,
    abar: torch.Tensor,
    ls_mu: torch.Tensor,
    ls_sd: torch.Tensor,
) -> torch.Tensor:
    """Integer timestep indices (B,) -> normalized logSNR codes (B,)."""
    abar_t = abar.gather(0, t_idx_batch).unsqueeze(1)  # (B,1)
    return time_code_from_abar(abar_t, ls_mu, ls_sd)


def timecode_from_tnorm(
    t_norm: torch.Tensor,
    abar: torch.Tensor,
    ls_mu: torch.Tensor,
    ls_sd: torch.Tensor,
) -> torch.Tensor:
    """Continuous normalized timestep t_norm in [0,1) -> logSNR codes (B,)."""
    T = abar.numel()
    t_idx = (t_norm * T).floor().clamp_(0, T - 1).long()
    return timecode_from_tindex(t_idx, abar, ls_mu, ls_sd)














