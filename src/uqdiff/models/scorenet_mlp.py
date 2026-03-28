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
        """t_code_1d: (B,) -> (B, time_dim)"""
        return self.net(t_code_1d[:, None])


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, time_dim: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.to_scale = nn.Linear(time_dim, hidden_dim)
        self.to_shift = nn.Linear(time_dim, hidden_dim)

        nn.init.zeros_(self.to_scale.weight); nn.init.zeros_(self.to_scale.bias)
        nn.init.zeros_(self.to_shift.weight); nn.init.zeros_(self.to_shift.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.fc(self.norm(x)))
        h = h * (1.0 + self.to_scale(t_emb)) + self.to_shift(t_emb)
        return x + h


class ScoreNet(nn.Module):
    """
    FiLM-conditioned MLP score network for 1-D time-series of length data_dim.
    Time conditioning is via normalized logSNR codes.
    """

    def __init__(
        self,
        data_dim: int,
        hidden_dim: int = 512,
        time_dim: int = 32,
        n_blocks: int = 2,
    ):
        super().__init__()
        self.time_mlp = LogSNRTimeMLP(time_dim)
        self.inp = nn.Linear(data_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, time_dim) for _ in range(n_blocks)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, data_dim)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def _encode(self, x: torch.Tensor, t_code: torch.Tensor):
        """Shared forward pass returning pre-output hidden state."""
        t_emb = self.time_mlp(t_code)
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h, t_emb)
        return F.silu(self.norm(h))

    def forward(self, x: torch.Tensor, t_code: torch.Tensor) -> torch.Tensor:
        """
        x:      (B, data_dim)
        t_code: (B,)  normalized logSNR
        returns (B, data_dim) predicted noise
        """
        return self.out(self._encode(x, t_code))

    def forward_with_feat(self, x: torch.Tensor, t_code: torch.Tensor):
        """
        Same as forward but also returns the penultimate hidden features.
        Used by LLLA uncertainty estimation.

        Returns:
            y: (B, data_dim)  predicted noise
            h: (B, hidden_dim) pre-output features
        """
        h = self._encode(x, t_code)
        return self.out(h), h














