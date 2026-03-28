"""
Experiment: LLLA vs Full-Hessian BayesDiff on R10 time-series.

Runs both uncertainty estimators on a shared x_T start and compares:
  - u_ep   : epistemic variance (t1+t3 recursion)
  - u_proj : transported projection to x0

Outputs are saved to assets/ for downstream plotting.

Usage
-----
python experiments/timeseries/diag_fullH.py \
    --checkpoint checkpoints/ddpm_r10_final.pth \
    --out_dir assets/figures \
    --n 200 --seed 20251022
"""

import argparse
import os
import pickle

import numpy as np
import torch

from uqdiff.models.scorenet_mlp import ScoreNet
from uqdiff.diffusion.schedules import make_schedules
from uqdiff.diffusion.timecode import prep_time_stats
from uqdiff.utils.data import make_gaussian_timeseries
from uqdiff.utils.ema import make_ema
from uqdiff.laplace.llla import build_llla_lastlayer_diag
from uqdiff.laplace.dataset import make_laplace_dataset
from uqdiff.laplace.bayesdiff_llla import DiffusionShim, sample_bayesdiff_samepath
from uqdiff.laplace.bayesdiff_fullH import (
    sample_bayesdiff_fullH_vmap,
    project_fullH_on_fixed_path,
    bayesdiff_fullH_total_on_fixed_path,
)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    type=str,   default="checkpoints/ddpm_r10_final.pth")
    p.add_argument("--data_dim",      type=int,   default=10)
    p.add_argument("--hidden_dim",    type=int,   default=512)
    p.add_argument("--time_dim",      type=int,   default=32)
    p.add_argument("--n_blocks",      type=int,   default=2)
    p.add_argument("--T",             type=int,   default=1000)
    p.add_argument("--n",             type=int,   default=200)
    p.add_argument("--seed",          type=int,   default=20251022)
    p.add_argument("--N_pairs",       type=int,   default=100_000)
    p.add_argument("--n_samples",     type=int,   default=5000)
    p.add_argument("--out_dir",       type=str,   default="assets/figures")
    p.add_argument("--history_path",  type=str,   default="assets/history_fullH.pkl")
    p.add_argument("--run_fullH",     action="store_true",
                   help="Also run full-Hessian Laplace (slow!)")
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.history_path), exist_ok=True)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    X = make_gaussian_timeseries(length=args.data_dim, n_samples=args.n_samples, std=0.1)
    print(f"Data: {X.shape}")

    # ── Model ─────────────────────────────────────────────────────────────
    model     = ScoreNet(args.data_dim, args.hidden_dim, args.time_dim, args.n_blocks)
    ema_model = make_ema(model)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        ema_model.load_state_dict(ckpt["ema_model_state"])
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"⚠️  Checkpoint not found ({args.checkpoint}), using random weights.")

    ema_model.to(device).eval()

    # ── Schedules ─────────────────────────────────────────────────────────
    betas, alphas, abar = make_schedules(args.T)
    ls_mu, ls_sd = prep_time_stats(abar)
    abar   = abar.to(device)
    ls_mu  = ls_mu.to(device)
    ls_sd  = ls_sd.to(device)

    # ── LLLA ──────────────────────────────────────────────────────────────
    print("Building LLLA …")
    la, wrapped, last_layer_name = build_llla_lastlayer_diag(
        ema_model, abar, ls_mu, ls_sd, data_dim=args.data_dim, device=device,
    )
    print(f"Last layer: {last_layer_name}")

    X_lap, Y_lap, _ = make_laplace_dataset(
        X, abar, args.T, N_pairs=args.N_pairs,
        data_dim=args.data_dim, device=device,
    )
    from torch.utils.data import TensorDataset, DataLoader
    loader_lap = DataLoader(
        TensorDataset(X_lap, Y_lap), batch_size=4096, shuffle=True, drop_last=True,
    )
    print("Fitting LLLA …")
    la.fit(loader_lap)
    la.optimize_prior_precision(method="marglik")

    # ── Shared init ───────────────────────────────────────────────────────
    g_main = torch.Generator(device=device).manual_seed(args.seed)
    g_mc   = torch.Generator(device=device).manual_seed(args.seed + 1)
    g_init = torch.Generator(device=device).manual_seed(args.seed + 2)
    xt0    = torch.randn(args.n, args.data_dim, device=device, generator=g_init)

    diffusion = DiffusionShim(betas.to(device))
    seq = list(range(diffusion.num_timesteps))

    # snapshot RNG so both methods see the same noise stream
    state_main = g_main.get_state()
    state_mc   = g_mc.get_state()

    # ── LLLA BayesDiff ────────────────────────────────────────────────────
    print("Running LLLA BayesDiff …")
    g_main.set_state(state_main); g_mc.set_state(state_mc)
    x0_llla, u_full_llla, u_ep_llla, u_proj_llla, xt_path, a_path, b_path = sample_bayesdiff_samepath(
        diffusion=diffusion, model=wrapped.net, la=la, seq=seq,
        abar=abar, ls_mu=ls_mu, ls_sd=ls_sd,
        n=args.n, data_dim=args.data_dim, device=device,
        use_cov=True, include_t4=False,
        S_cov=64, tau_gamma2=1.0,
        rng=g_main, rng_mc=g_mc, xt0=xt0,
        return_path=True, show_progress=True,
    )
    print(f"LLLA: x0={x0_llla.shape}  u_ep mean={u_ep_llla.mean():.4f}")

    results = {
        "x0_llla": x0_llla.cpu(), "u_ep_llla": u_ep_llla.cpu(),
        "u_proj_llla": u_proj_llla.cpu(), "u_full_llla": u_full_llla.cpu(),
    }

    # ── Full-Hessian (optional) ───────────────────────────────────────────
    if args.run_fullH:
        try:
            from laplace import Laplace
            import copy

            print("Building full-Hessian Laplace …")
            ema_for_laplace = copy.deepcopy(ema_model).eval().to(device)
            for p in ema_for_laplace.parameters():
                p.requires_grad_(True)

            from uqdiff.laplace.wrapper import LaplaceWrapper
            wrapped_full = LaplaceWrapper(ema_for_laplace, abar, ls_mu, ls_sd, args.data_dim).to(device)
            for p in wrapped_full.parameters():
                p.requires_grad_(True)

            la_full = Laplace(wrapped_full, likelihood="regression",
                              subset_of_weights="all", hessian_structure="full")
            la_full.fit(loader_lap)
            la_full.optimize_prior_precision(method="marglik")

            print("Running full-H projection on fixed path …")
            u_proj_fullH = project_fullH_on_fixed_path(
                wrapped=wrapped_full, la_full=la_full, abar=abar,
                xt_path=xt_path, a_path=a_path, b_path=b_path, t_seq=seq,
                device=device, show_progress=True,
            )
            results["u_proj_fullH"] = u_proj_fullH.cpu()
            print(f"Full-H: u_proj mean={u_proj_fullH.mean():.4f}")

        except Exception as e:
            print(f"Full-H run failed: {e}")

    # ── Save ──────────────────────────────────────────────────────────────
    with open(args.history_path, "wb") as f:
        pickle.dump(results, f)
    print(f"✅ Results saved to {args.history_path}")


if __name__ == "__main__":
    main()














