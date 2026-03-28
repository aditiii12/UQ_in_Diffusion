import torch
import pytest
from uqdiff.diffusion.schedules import make_schedules
from uqdiff.diffusion.timecode import prep_time_stats, timecode_from_tindex, timecode_from_tnorm


@pytest.fixture
def schedule():
    _, _, abar = make_schedules(T=100)
    mu, sd = prep_time_stats(abar)
    return abar, mu, sd


def test_timecode_shape(schedule):
    abar, mu, sd = schedule
    t = torch.tensor([0, 50, 99])
    codes = timecode_from_tindex(t, abar, mu, sd)
    assert codes.shape == (3,)


def test_timecode_from_tnorm(schedule):
    abar, mu, sd = schedule
    t_norm = torch.tensor([0.0, 0.5, 0.99])
    codes  = timecode_from_tnorm(t_norm, abar, mu, sd)
    assert codes.shape == (3,)
    assert codes.isfinite().all()


def test_timecode_monotone(schedule):
    """logSNR decreases as t increases (more noise at higher t)."""
    abar, mu, sd = schedule
    T = abar.numel()
    t = torch.arange(T)
    codes = timecode_from_tindex(t, abar, mu, sd)
    diffs = codes[1:] - codes[:-1]
    assert (diffs <= 0).all()














