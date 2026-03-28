import copy
import torch
import torch.nn as nn


def make_ema(model: nn.Module) -> nn.Module:
    """Return a detached, non-differentiable copy for EMA tracking."""
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


@torch.no_grad()
def ema_update(model: nn.Module, ema_model: nn.Module, decay: float = 0.999) -> None:
    """In-place EMA update: ema = decay * ema + (1 - decay) * model."""
    for p, q in zip(model.parameters(), ema_model.parameters()):
        q.data.mul_(decay).add_(p.data, alpha=1.0 - decay)