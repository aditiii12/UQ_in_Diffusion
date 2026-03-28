import copy
import torch
import torch.nn as nn
from laplace import Laplace
from uqdiff.diffusion.timecode import timecode_from_tindex


def build_llla_lastlayer_diag(
    ema_model: nn.Module,
    abar: torch.Tensor,
    ls_mu: torch.Tensor,
    ls_sd: torch.Tensor,
    data_dim: int,
    device: str = "cuda",
):
    """
    Build a Last-Layer Laplace Approximation (LLLA) with diagonal Hessian.

    Freezes all layers except the final Linear and wraps the model so that
    laplace-torch sees [x_t, t_norm] as input.

    Returns
    -------
    la              : unfitted Laplace object (call la.fit(loader) next)
    wrapped         : LaplaceWrapper around the frozen EMA model
    last_layer_name : str name of the final Linear layer
    """
    from uqdiff.laplace.wrapper import LaplaceWrapper

    ema_for_laplace = copy.deepcopy(ema_model).eval().to(device)
    for p in ema_for_laplace.parameters():
        p.requires_grad_(False)

    wrapped = LaplaceWrapper(
        ema_for_laplace,
        abar.to(device), ls_mu.to(device), ls_sd.to(device),
        data_dim=data_dim,
    ).to(device)

    # find last Linear(*, data_dim) — should be ScoreNet.out
    last_layer_name = None
    for name, m in wrapped.named_modules():
        if isinstance(m, nn.Linear) and m.out_features == data_dim:
            last_layer_name = name

    if last_layer_name is None:
        raise RuntimeError(f"No Linear(..., {data_dim}) found in wrapped model.")

    # unfreeze only the last layer for Laplace to fit
    for name, m in wrapped.named_modules():
        if name == last_layer_name:
            for p in m.parameters():
                p.requires_grad_(True)
            break

    la = Laplace(
        wrapped,
        likelihood="regression",
        subset_of_weights="last_layer",
        hessian_structure="diag",
        last_layer_name=last_layer_name,
    )
    return la, wrapped, last_layer_name


@torch.no_grad()
def llla_gamma2_diag_lastlayer(
    la,
    score_model: nn.Module,
    xt: torch.Tensor,
    t_idx: torch.Tensor,
    abar: torch.Tensor,
    ls_mu: torch.Tensor,
    ls_sd: torch.Tensor,
    eps: float = 1e-12,
):
    """
    Compute μ_ε and diagonal epistemic variance γ² for each output dimension.

    γ²_k = h² W_var[k] + b_var[k]   where Σ = diag(1 / posterior_precision)

    Parameters
    ----------
    la          : fitted LLLA Laplace object
    score_model : ScoreNet with .forward_with_feat
    xt          : (B, data_dim) noisy input
    t_idx       : (B,) integer timestep indices
    abar        : (T,) cumulative alpha schedule
    ls_mu/ls_sd : logSNR normalization stats

    Returns
    -------
    mu_eps : (B, data_dim)  predicted noise
    gamma2 : (B, data_dim)  epistemic variance per output dim
    """
    t_code = timecode_from_tindex(t_idx.long(), abar, ls_mu, ls_sd)
    mu_eps, h = score_model.forward_with_feat(xt, t_code)   # (B,D), (B,H)

    B, H = h.shape
    D = mu_eps.shape[1]

    prec = la.posterior_precision.detach().to(h.device, h.dtype)
    var  = 1.0 / (prec + eps)

    W_var = var[:D * H].view(D, H)   # (D, H)
    b_var = var[D * H:]               # (D,)

    gamma2 = (h * h) @ W_var.T + b_var.unsqueeze(0)   # (B, D)
    return mu_eps, gamma2.clamp_min(0.0)














