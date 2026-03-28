import torch

def cosine_beta_schedule(T: int, s: float = 0.008, device=None) -> torch.Tensor:
    steps = T + 1
    x = torch.linspace(0, T, steps, device=device)
    abar = torch.cos(((x / T) + s) / (1 + s) * torch.pi / 2) ** 2
    abar = abar / abar[0].clamp(min=1e-8)
    betas = 1 - (abar[1:] / abar[:-1])
    return betas.clamp(1e-5, 0.999)

def make_schedules(T: int, device=None):
    betas = cosine_beta_schedule(T, device=device)
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)
    return betas, alphas, abar
