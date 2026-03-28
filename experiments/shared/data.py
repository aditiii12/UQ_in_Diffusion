"""
experiments/shared/data.py
---------------------------
Synthetic time-series datasets used in the UQ experiments.

Sines experiment (data_dim=10):
  - make_gaussian_timeseries : bimodal sine / negative-sine

Chirp experiment (data_dim=80, finer=8):
  - make_sine
  - make_sine_plus_trend
  - make_chirp
  - make_damped_sine
  - make_beats
  - make_am
  - make_fm

All generators return float32 arrays of shape (n, length * finer).
"""

from __future__ import annotations
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch


# ---------------------------------------------------------------------------
# Shared grid helper
# ---------------------------------------------------------------------------

def _time_grid(length: int, finer: int = 1) -> np.ndarray:
    return np.linspace(0, 1, int(length) * int(finer), endpoint=False)


# ---------------------------------------------------------------------------
# Sines experiment
# ---------------------------------------------------------------------------

def make_gaussian_timeseries(
    length: int = 10,
    n_samples: int = 5000,
    std: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """
    Bimodal dataset: sine and negative-sine with Gaussian noise.
    Returns (n_samples, length) float32.
    """
    rng        = np.random.default_rng(seed)
    t          = np.linspace(0, 1, length)
    base_sin   = np.sin(2 * np.pi * t)
    data       = []
    for _ in range(n_samples):
        base = base_sin if rng.random() < 0.5 else -base_sin
        data.append(base + rng.normal(0, std, length))
    return np.array(data, dtype=np.float32)


# ---------------------------------------------------------------------------
# Chirp experiment
# ---------------------------------------------------------------------------

def make_sine(
    n: int, length: int = 10, noise_std: float = 0.05, seed: int = 0,
    amp_range=(0.6, 1.4), freq_range=(1.0, 1.0),
    phase_range=(0, 2 * np.pi), finer: int = 1,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t   = _time_grid(length, finer)
    A   = rng.uniform(*amp_range, size=n).astype(np.float32)
    f   = rng.uniform(*freq_range, size=n).astype(np.float32)
    phi = rng.uniform(*phase_range, size=n).astype(np.float32)
    X   = A[:, None] * np.sin(2 * np.pi * (f[:, None] * t + phi[:, None]))
    if noise_std > 0:
        X += rng.normal(0, noise_std, X.shape).astype(np.float32)
    return X.astype(np.float32)


def make_sine_plus_trend(
    n: int, length: int = 10, noise_std: float = 0.05, seed: int = 1,
    amp_range=(0.6, 1.4), freq_range=(1.0, 1.0),
    phase_range=(0, 2 * np.pi), slope_range=(-0.8, 0.8),
    finer: int = 1,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t   = _time_grid(length, finer)
    A   = rng.uniform(*amp_range, size=n).astype(np.float32)
    f   = rng.uniform(*freq_range, size=n).astype(np.float32)
    phi = rng.uniform(*phase_range, size=n).astype(np.float32)
    m   = rng.uniform(*slope_range, size=n).astype(np.float32)
    X   = A[:, None] * np.sin(2 * np.pi * (f[:, None] * t + phi[:, None])) \
          + m[:, None] * (t - 0.5)
    if noise_std > 0:
        X += rng.normal(0, noise_std, X.shape).astype(np.float32)
    return X.astype(np.float32)


def make_chirp(
    n: int, length: int = 10, noise_std: float = 0.02, seed: int = 2,
    amp_range=(0.6, 1.4), f0_range=(0.5, 1.0), k_range=(2.0, 5.0),
    finer: int = 1,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t   = _time_grid(length, finer)
    A   = rng.uniform(*amp_range, size=n).astype(np.float32)
    f0  = rng.uniform(*f0_range,  size=n).astype(np.float32)
    k   = rng.uniform(*k_range,   size=n).astype(np.float32)
    ph  = f0[:, None] * t + 0.5 * k[:, None] * (t ** 2)
    X   = A[:, None] * np.sin(2 * np.pi * ph)
    if noise_std > 0:
        X += rng.normal(0, noise_std, X.shape).astype(np.float32)
    return X.astype(np.float32)


def make_damped_sine(
    n: int, length: int = 10, noise_std: float = 0.03, seed: int = 10,
    amp_range=(0.8, 1.3), freq_range=(0.8, 1.5),
    decay_range=(1.0, 5.0), phase_range=(0, 2 * np.pi),
    finer: int = 1,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t   = _time_grid(length, finer)
    A   = rng.uniform(*amp_range,   size=n).astype(np.float32)
    f   = rng.uniform(*freq_range,  size=n).astype(np.float32)
    d   = rng.uniform(*decay_range, size=n).astype(np.float32)
    phi = rng.uniform(*phase_range, size=n).astype(np.float32)
    env = np.exp(-d[:, None] * t)
    X   = A[:, None] * env * np.sin(2 * np.pi * (f[:, None] * t + phi[:, None]))
    if noise_std > 0:
        X += rng.normal(0, noise_std, X.shape).astype(np.float32)
    return X.astype(np.float32)


def make_beats(
    n: int, length: int = 10, noise_std: float = 0.03, seed: int = 11,
    amp_range=(0.8, 1.2), f1_range=(1.0, 1.2), f2_delta=(0.02, 0.08),
    phase_range=(0, 2 * np.pi), finer: int = 1,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t   = _time_grid(length, finer)
    A   = rng.uniform(*amp_range,  size=n).astype(np.float32)
    f1  = rng.uniform(*f1_range,   size=n).astype(np.float32)
    df  = rng.uniform(*f2_delta,   size=n).astype(np.float32)
    f2  = f1 + df
    phi = rng.uniform(*phase_range, size=n).astype(np.float32)
    X   = 0.5 * A[:, None] * (
        np.sin(2 * np.pi * (f1[:, None] * t + phi[:, None])) +
        np.sin(2 * np.pi * (f2[:, None] * t + phi[:, None]))
    )
    if noise_std > 0:
        X += rng.normal(0, noise_std, X.shape).astype(np.float32)
    return X.astype(np.float32)


def make_am(
    n: int, length: int = 10, noise_std: float = 0.03, seed: int = 12,
    carrier_freq=(1.0, 1.2), mod_freq=(0.1, 0.25),
    mod_index=(0.3, 0.8), amp_range=(0.8, 1.2),
    phase_range=(0, 2 * np.pi), finer: int = 1,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t   = _time_grid(length, finer)
    Ac  = rng.uniform(*amp_range,    size=n).astype(np.float32)
    fc  = rng.uniform(*carrier_freq, size=n).astype(np.float32)
    fm  = rng.uniform(*mod_freq,     size=n).astype(np.float32)
    m   = rng.uniform(*mod_index,    size=n).astype(np.float32)
    phi = rng.uniform(*phase_range,  size=n).astype(np.float32)
    env = 1 + m[:, None] * np.sin(2 * np.pi * fm[:, None] * t)
    X   = Ac[:, None] * env * np.sin(2 * np.pi * (fc[:, None] * t + phi[:, None]))
    if noise_std > 0:
        X += rng.normal(0, noise_std, X.shape).astype(np.float32)
    return X.astype(np.float32)


def make_fm(
    n: int, length: int = 10, noise_std: float = 0.03, seed: int = 13,
    base_freq=(0.8, 1.1), dev_freq=(0.15, 0.4), dev_index=(0.5, 1.5),
    amp_range=(0.9, 1.1), phase_range=(0, 2 * np.pi),
    finer: int = 1,
) -> np.ndarray:
    rng   = np.random.default_rng(seed)
    t     = _time_grid(length, finer)
    A     = rng.uniform(*amp_range,  size=n).astype(np.float32)
    f0    = rng.uniform(*base_freq,  size=n).astype(np.float32)
    fd    = rng.uniform(*dev_freq,   size=n).astype(np.float32)
    beta  = rng.uniform(*dev_index,  size=n).astype(np.float32)
    phi   = rng.uniform(*phase_range, size=n).astype(np.float32)
    phase = 2 * np.pi * f0[:, None] * t + beta[:, None] * np.sin(2 * np.pi * fd[:, None] * t) + phi[:, None]
    X     = A[:, None] * np.sin(phase)
    if noise_std > 0:
        X += rng.normal(0, noise_std, X.shape).astype(np.float32)
    return X.astype(np.float32)


# ---------------------------------------------------------------------------
# DataLoader helper
# ---------------------------------------------------------------------------

def make_dataloader(
    X: np.ndarray,
    batch_size: int = 512,
    shuffle: bool = True,
    num_workers: int = 0,
    device: str = "cuda",
) -> DataLoader:
    X_t = torch.as_tensor(X, dtype=torch.float32)
    ds  = TensorDataset(X_t)
    pin = "cuda" in str(device)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        drop_last=True, num_workers=num_workers, pin_memory=pin
    )