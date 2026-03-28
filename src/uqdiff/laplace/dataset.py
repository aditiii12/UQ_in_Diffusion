import torch
import numpy as np
from typing import Optional, Tuple, Union
from torch.utils.data import TensorDataset, DataLoader

@torch.no_grad()
def make_laplace_dataset(
    X: Union[torch.Tensor, np.ndarray],
    abar: torch.Tensor,
    T: int,
    N_pairs: int,
    data_dim: int,
    device: str = "cpu",
    snr_gamma: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    abar = abar.to(device)

    N = X.shape[0]
    idx = torch.randint(0, N, (N_pairs,), device=device)
    x0  = X[idx]

    if snr_gamma is None:
        t = torch.randint(0, T, (N_pairs,), device=device)
    else:
        snr = abar / (1.0 - abar + 1e-8)
        w = torch.minimum(snr, torch.full_like(snr, snr_gamma)) / (snr + 1e-8)
        p = (w.clamp_min(0) / w.sum()).pow(2.0)
        p = p / p.sum()
        t = torch.multinomial(p, N_pairs, replacement=True)

    abar_t = abar[t].unsqueeze(1)
    eps    = torch.randn(N_pairs, data_dim, device=device)

    x_t   = torch.sqrt(abar_t) * x0 + torch.sqrt(1.0 - abar_t) * eps
    tnorm = t.float() / float(T)

    X_lap = torch.cat([x_t, tnorm[:, None]], dim=1)
    Y_lap = eps

    return X_lap, Y_lap, t

def make_laplace_loader(X_lap, Y_lap, batch: int = 4096, shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(X_lap, Y_lap)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=True)