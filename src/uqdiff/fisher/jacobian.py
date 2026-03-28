"""
Full-Hessian epistemic uncertainty utilities.

Provides:
  - pack_params_and_buffers_backend_order : align param dict with Laplace backend
  - flatten_param_pytree                  : flatten grad dict to 1-D vector
  - make_batched_param_grads_fn           : vmap'd per-output Jacobians
  - robust_chol                           : numerically stable Cholesky
  - gamma2_full_laplace_vmap              : per-output γ² = diag(J Σ Jᵀ)
"""

import time
import torch
import torch.nn as nn
from torch.func import grad, vmap, functional_call


# ---------------------------------------------------------------------------
# Parameter packing (must match Laplace backend ordering)
# ---------------------------------------------------------------------------

@torch.no_grad()
def pack_params_and_buffers_backend_order(model: nn.Module, backend):
    """
    Return params and buffers aligned with the ordering used by the Laplace backend.
    This ensures Jacobian vectors line up with posterior_precision.
    """
    params = {
        k: p.detach().requires_grad_(True)
        for k, p in backend.params_dict.items()
    }
    buffers = dict(model.named_buffers())
    keys    = list(params.keys())
    shapes  = [params[k].shape for k in keys]
    numels  = [params[k].numel() for k in keys]

    idx_slices, s = [], 0
    for n in numels:
        idx_slices.append(slice(s, s + n))
        s += n

    return params, buffers, keys, shapes, idx_slices, s


def flatten_param_pytree(grads_dict: dict, keys: list, params_ref: dict) -> torch.Tensor:
    """Flatten a gradient dict (same ordering as keys) into a 1-D tensor."""
    flats = []
    for k in keys:
        g = grads_dict.get(k, None)
        if g is None:
            g = torch.zeros_like(params_ref[k])
        flats.append(g.reshape(-1))
    return torch.cat(flats, dim=0)


# ---------------------------------------------------------------------------
# Batched per-output Jacobians via vmap + grad
# ---------------------------------------------------------------------------

def make_batched_param_grads_fn(wrapped: nn.Module, params_ref: dict, buffers: dict, keys: list):
    """
    Return a function `batched_grads(X_batch, k)` that computes the
    per-sample Jacobian of output dimension k with respect to all parameters.
    """
    def scalar_out(pbags, x_cat_row, k: int):
        y = functional_call(wrapped, (pbags, buffers), (x_cat_row.unsqueeze(0),))
        return y[0, k]

    def grad_wrt_params(x_cat_row, k: int):
        gtree = grad(lambda p: scalar_out(p, x_cat_row, k))(params_ref)
        return flatten_param_pytree(gtree, keys, params_ref)

    def batched_grads(X_batch: torch.Tensor, k: int) -> torch.Tensor:
        return vmap(lambda row: grad_wrt_params(row, k))(X_batch)

    return batched_grads


# ---------------------------------------------------------------------------
# Numerically stable Cholesky decomposition
# ---------------------------------------------------------------------------

def robust_chol(P: torch.Tensor, base_rel: float = 1e-8) -> torch.Tensor:
    """
    Cholesky of a (nearly) PSD matrix P with adaptive jitter.
    Falls back to eigenvalue clamping if all jitter attempts fail.
    """
    Ps    = 0.5 * (P + P.T)
    scale = torch.median(torch.diag(Ps)).clamp_min(1e-12)
    I     = torch.eye(Ps.shape[0], device=Ps.device, dtype=Ps.dtype)
    jitter = float(base_rel) * float(scale)

    for _ in range(8):
        try:
            return torch.linalg.cholesky(Ps + jitter * I)
        except RuntimeError:
            jitter *= 10.0

    # last resort: clamp eigenvalues
    w, V = torch.linalg.eigh(Ps)
    w = torch.clamp(w, min=float(base_rel) * float(scale))
    return torch.linalg.cholesky((V * w.unsqueeze(0)) @ V.T + 1e-12 * I)


# ---------------------------------------------------------------------------
# γ² = diag(J Σ Jᵀ) via Cholesky solve
# ---------------------------------------------------------------------------

@torch.no_grad()
def gamma2_full_laplace_vmap(
    wrapped: nn.Module,
    L: torch.Tensor,
    params: dict,
    buffers: dict,
    keys: list,
    xt: torch.Tensor,
    t_idx: torch.Tensor,
    abar: torch.Tensor,
    max_batch: int = 256,
    out_chunk: int = 4,
    timers: dict | None = None,
) -> torch.Tensor:
    """
    Compute per-sample, per-output epistemic variance γ²_k = (J_k Σ J_kᵀ)
    using a full-Hessian Laplace posterior (Σ = L⁻ᵀ L⁻¹).

    Parameters
    ----------
    wrapped     : LaplaceWrapper (net + schedule buffers)
    L           : lower-triangular Cholesky of posterior precision P
    params      : parameter dict aligned with Laplace backend
    buffers     : buffer dict from wrapped.named_buffers()
    keys        : parameter name ordering
    xt          : (B, data_dim) current noisy samples
    t_idx       : (B,) integer timestep indices
    abar        : (T,) cumulative alpha schedule
    max_batch   : maximum sub-batch size for Jacobian computation
    out_chunk   : number of output dimensions to process per inner loop
    timers      : optional dict to accumulate timing info

    Returns
    -------
    gamma2 : (B, data_dim) epistemic variance, clamped >= 0
    """
    device, dtype = xt.device, xt.dtype
    T = abar.numel()
    t_norm = t_idx.float() / T
    x_cat  = torch.cat([xt, t_norm[:, None]], dim=1)

    D_out = wrapped.net.out.out_features
    batched_grads = make_batched_param_grads_fn(wrapped, params, buffers, keys)

    B = x_cat.shape[0]
    gamma2 = torch.empty(B, D_out, device=device, dtype=dtype)

    step_B = max_batch if max_batch else B
    for s in range(0, B, step_B):
        e  = min(B, s + step_B)
        Xb = x_cat[s:e]

        for k0 in range(0, D_out, out_chunk):
            k1 = min(D_out, k0 + out_chunk)
            ks = list(range(k0, k1))

            t0 = time.time()
            G_list = [batched_grads(Xb, k) for k in ks]
            if timers is not None:
                timers["gamma_grads"] = timers.get("gamma_grads", 0.0) + (time.time() - t0)

            t1 = time.time()
            U_list = [torch.cholesky_solve(G.t().contiguous(), L).t() for G in G_list]
            if timers is not None:
                timers["gamma_solve"] = timers.get("gamma_solve", 0.0) + (time.time() - t1)

            gamma2[s:e, k0:k1] = torch.stack(
                [(G * U).sum(dim=1).to(dtype) for G, U in zip(G_list, U_list)], dim=1
            )

    return gamma2.clamp_min(0.0)














