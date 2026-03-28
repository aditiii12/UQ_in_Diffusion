"""
experiments/shared/scorenet.py
-------------------------------
FiLM-conditioned MLP score network for 1D time series.

Architecture:
  - LogSNRTimeMLP  : maps scalar logSNR time code -> time embedding
  - ResidualBlock  : FiLM-conditioned residual block
  - ScoreNet       : full score network (xt, t_code) -> eps

ScoreNet.forward_with_feat is required by gamma2_diag in uqdiff.laplace.precision
— it returns both the output and the pre-output hidden features.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class LogSNRTimeMLP(nn.Module):
    def __init__(self, time_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim), nn.SiLU(),
        )

    def forward(self, t_code_1d: torch.Tensor) -> torch.Tensor:
        # t_code_1d: (B,) -> (B, time_dim)
        return self.net(t_code_1d[:, None])


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, time_dim: int):
        super().__init__()
        self.fc       = nn.Linear(hidden_dim, hidden_dim)
        self.norm     = nn.LayerNorm(hidden_dim)
        self.to_scale = nn.Linear(time_dim, hidden_dim)
        self.to_shift = nn.Linear(time_dim, hidden_dim)
        # FiLM ~ identity at init
        nn.init.zeros_(self.to_scale.weight); nn.init.zeros_(self.to_scale.bias)
        nn.init.zeros_(self.to_shift.weight); nn.init.zeros_(self.to_shift.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h     = F.silu(self.fc(self.norm(x)))
        scale = self.to_scale(t_emb)
        shift = self.to_shift(t_emb)
        return x + h * (1 + scale) + shift


class ScoreNet(nn.Module):
    """
    FiLM-conditioned MLP score network.

    Args:
        data_dim   : dimensionality of x_t (e.g. 10 for sines, 80 for chirp)
        hidden_dim : width of residual blocks
        time_dim   : time embedding dimension
        n_blocks   : number of residual blocks (0 = linear model)
    """
    def __init__(
        self,
        data_dim: int = 10,
        hidden_dim: int = 512,
        time_dim: int = 32,
        n_blocks: int = 2,
    ):
        super().__init__()
        self.time_mlp = LogSNRTimeMLP(time_dim)
        self.inp      = nn.Linear(data_dim, hidden_dim)
        self.blocks   = nn.ModuleList([
            ResidualBlock(hidden_dim, time_dim) for _ in range(n_blocks)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.out  = nn.Linear(hidden_dim, data_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, t_code: torch.Tensor) -> torch.Tensor:
        """
        x      : (B, data_dim)
        t_code : (B,)  normalized logSNR
        returns: (B, data_dim) predicted noise eps
        """
        t_emb = self.time_mlp(t_code)
        h     = self.inp(x)
        for blk in self.blocks:
            h = blk(h, t_emb)
        return self.out(F.silu(self.norm(h)))

    def forward_with_feat(
        self, x: torch.Tensor, t_code: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Same as forward but also returns pre-output hidden features h.
        Required by gamma2_diag for LLLA variance computation.

        Returns: (eps, h) where h: (B, hidden_dim)
        """
        t_emb = self.time_mlp(t_code)
        h     = self.inp(x)
        for blk in self.blocks:
            h = blk(h, t_emb)
        h   = F.silu(self.norm(h))
        eps = self.out(h)
        return eps, h