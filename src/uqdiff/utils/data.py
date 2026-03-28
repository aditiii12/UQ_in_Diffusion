import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def make_gaussian_timeseries(
    length: int = 10,
    n_samples: int = 200,
    std: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    """
    Generate a bimodal dataset of noisy sine / negative-sine sequences.

    Returns
    -------
    X : np.ndarray of shape (n_samples, length)
    """
    np.random.seed(seed)
    t = np.linspace(0, 1, length)
    base_sin = np.sin(2 * np.pi * t)

    data = []
    for _ in range(n_samples):
        base = base_sin if np.random.rand() < 0.5 else -base_sin
        data.append(base + np.random.randn(length) * std)
    return np.array(data)


def make_dataloader(
    X,
    batch_size: int = 512,
    shuffle: bool = True,
    num_workers: int = 0,
    device: str = "cpu",
) -> DataLoader:
    """
    Wrap a numpy / tensor dataset in a DataLoader.
    Data stays on CPU; pin_memory is enabled for CUDA targets.
    """
    X = torch.as_tensor(X, dtype=torch.float32)
    ds = TensorDataset(X)
    pin = "cuda" in str(device)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin,
    )














