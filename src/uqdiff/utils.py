"""
uqdiff/utils.py
---------------
Shared low-level utilities:
  - Normalized logSNR time-coding helpers
  - Generator-aware randn_like
  - Robust Cholesky decomposition
"""

from __future__ import annotations
import torch
from typing import Optional


# ---------------------------------------------------------------------------
# logSNR time-coding (must match training)
# ---------------------------------------------------------------------------

def logsnr_from_abar(a: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """log(ᾱ / (1 - ᾱ))  — raw logSNR from cumulative alpha."""
    a = a.clamp(min=eps, max=1 - eps)
    return torch.log(a) - torch.log1p(-a)


@torch.no_grad()
def prep_time_stats(abar: torch.Tensor):
    """
    Compute mean and std of logSNR over the full schedule.
    Returns (ls_mu, ls_sd) — pass these into all time-coding calls.
    """
    ls = logsnr_from_abar(abar)
    return ls.mean(), ls.std().clamp_min(1e-6)


def time_code_from_abar(
    abar_t: torch.Tensor,   # (B, 1) or (1, 1)
    mu: torch.Tensor,
    sd: torch.Tensor,
) -> torch.Tensor:
    """Normalized logSNR code: (logSNR(abar_t) - mu) / sd  →  (B,)"""
    return ((logsnr_from_abar(abar_t) - mu) / sd).squeeze(1)


def timecode_from_tindex(
    t_idx_batch: torch.Tensor,   # (B,) int64
    abar: torch.Tensor,          # (T,)
    ls_mu: torch.Tensor,
    ls_sd: torch.Tensor,
) -> torch.Tensor:
    """Map integer timestep indices → normalized logSNR codes  (B,)."""
    abar_t = abar.gather(0, t_idx_batch).unsqueeze(1)   # (B, 1)
    return time_code_from_abar(abar_t, ls_mu, ls_sd)    # (B,)


def timecode_from_tnorm(
    t_norm: torch.Tensor,   # (B,) in [0, 1]
    abar: torch.Tensor,
    ls_mu: torch.Tensor,
    ls_sd: torch.Tensor,
) -> torch.Tensor:
    """Map normalized timestep t/T → normalized logSNR codes  (B,)."""
    T = abar.numel()
    t_idx = (t_norm * T).floor().clamp_(0, T - 1).long()
    return timecode_from_tindex(t_idx, abar, ls_mu, ls_sd)


# ---------------------------------------------------------------------------
# Generator-aware randn_like
# ---------------------------------------------------------------------------

def randn_like_gen(
    x: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """torch.randn_like but respects an optional Generator for reproducibility."""
    if generator is not None:
        return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    return torch.randn_like(x)


# ---------------------------------------------------------------------------
# Robust Cholesky (used for full/subnet Laplace precision matrices)
# ---------------------------------------------------------------------------

@torch.no_grad()
def robust_chol(P: torch.Tensor, base_rel: float = 1e-8) -> torch.Tensor:
    """
    Cholesky of a symmetric positive semi-definite matrix P with automatic
    jitter for numerical stability.  Falls back to eigenvalue clamping if
    iterative jitter still fails.
    """
    Ps = 0.5 * (P + P.T)
    scale = torch.median(torch.diag(Ps)).clamp_min(1e-12)
    I = torch.eye(Ps.shape[0], device=Ps.device, dtype=Ps.dtype)
    jitter = float(base_rel) * float(scale)
    for _ in range(8):
        try:
            return torch.linalg.cholesky(Ps + jitter * I)
        except RuntimeError:
            jitter *= 10.0
    # last-resort: clamp small eigenvalues
    w, V = torch.linalg.eigh(Ps)
    w = torch.clamp(w, min=float(base_rel) * float(scale))
    return torch.linalg.cholesky((V * w.unsqueeze(0)) @ V.T + 1e-12 * I)