import torch
from typing import Optional

def randn_like_gen(x: torch.Tensor, gen: Optional[torch.Generator] = None) -> torch.Tensor:
    # Avoid torch.randn_like(..., generator=gen) because some builds don’t support it.
    if gen is None:
        return torch.randn_like(x)
    return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=gen)