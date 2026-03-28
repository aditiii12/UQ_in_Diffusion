"""
uqdiff/schedules.py
--------------------
Noise schedules and diffusion coefficients.

  - cosine_beta_schedule : cosine beta schedule (Nichol & Dhariwal 2021)
  - make_schedules       : betas, alphas, abar
  - compute_alpha        : ᾱ_t for a batch of timesteps (used in samplers)
"""

from __future__ import annotations
import torch


def cosine_beta_schedule(T: int, s: float = 0.008, device=None) -> torch.Tensor:
    """
    Cosine beta schedule from Nichol & Dhariwal (2021).
    Returns betas of shape (T,) clamped to [1e-5, 0.999].
    """
    steps = T + 1
    x     = torch.linspace(0, T, steps, device=device)
    abar  = torch.cos(((x / T) + s) / (1 + s) * torch.pi / 2) ** 2
    abar  = abar / abar[0].clamp(min=1e-8)
    betas = 1 - (abar[1:] / abar[:-1])
    return betas.clamp(1e-5, 0.999)


def make_schedules(
    T: int,
    device=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (betas, alphas, abar) each of shape (T,).
    """
    betas  = cosine_beta_schedule(T, device=device)
    alphas = 1.0 - betas
    abar   = torch.cumprod(alphas, dim=0)
    return betas, alphas, abar


def compute_alpha(betas: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Return ᾱ_t for a batch of integer timestep indices.

    Args:
        betas : (T,)
        t     : (B,) int64

    Returns: (B, 1)
    """
    betas = torch.cat([torch.zeros(1, device=betas.device, dtype=betas.dtype), betas])
    return (1 - betas).cumprod(0).index_select(0, t.long() + 1).view(-1, 1)