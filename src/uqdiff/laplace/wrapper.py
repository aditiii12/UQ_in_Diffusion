import torch
import torch.nn as nn
from uqdiff.diffusion.timecode import timecode_from_tindex


class LaplaceWrapper(nn.Module):
    """
    Wrap ScoreNet(xt, t_code) so Laplace sees input:
        x_cat = [x_t, t_norm]
    """

    def __init__(self, net, abar, ls_mu, ls_sd, data_dim: int):
        super().__init__()
        self.net = net
        self.data_dim = int(data_dim)

        self.register_buffer("abar",  torch.as_tensor(abar,  dtype=torch.float32))
        self.register_buffer("ls_mu", torch.as_tensor(ls_mu, dtype=torch.float32))
        self.register_buffer("ls_sd", torch.as_tensor(ls_sd, dtype=torch.float32))

    def forward(self, x_cat: torch.Tensor):
        xt     = x_cat[:, :self.data_dim]
        t_norm = x_cat[:, self.data_dim]

        T = self.abar.numel()
        t_idx = (t_norm * T).clamp_(0, T - 1).long()

        t_code = timecode_from_tindex(t_idx, self.abar, self.ls_mu, self.ls_sd)
        return self.net(xt, t_code)