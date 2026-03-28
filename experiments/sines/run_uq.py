"""
experiments/sines/run_uq.py
----------------------------
UQ experiment for the sines dataset (data_dim=10).

Compares:
  1. BayesDiff (LLLA)        — full variance recursion, diagonal last-layer γ²
  2. FLARE (LLLA)            — projection, diagonal last-layer γ²
  3. FLARE (Full Hessian)    — projection, full Hessian γ², replayed on LLLA path

All three run on the same x_T start and same noise stream for apples-to-apples comparison.

Usage:
    python experiments/sines/run_uq.py --checkpoint checkpoints/sines/ddpm_sines.pth
"""

from __future__ import annotations
import argparse
import pickle
import os

import numpy as np
import torch
import yaml

from uqdiff.schedules import make_schedules
from uqdiff.utils import prep_time_stats
from uqdiff.laplace.core import (
    DiffusionShim,
    make_laplace_dataset,
    make_laplace_loader,
    build_llla,
    build_full,
)
from uqdiff.laplace.bayesdiff import sample_bayesdiff
from uqdiff.laplace.flare import project_llla, project_full, sample_flare_full

# experiment-local
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.scorenet import ScoreNet
from shared.data import make_gaussian_timeseries


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to saved checkpoint (model + EMA weights)")
    p.add_argument("--config", type=str,
                   default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out_dir", type=str, default="assets/sines")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = get_args()
    device = args.device

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- data ----
    X = make_gaussian_timeseries(
        length=cfg["data"]["length"],
        n_samples=cfg["data"]["n_samples"],
        std=cfg["data"]["std"],
    )

    # ---- model ----
    ckpt      = torch.load(args.checkpoint, map_location=device)
    model_cfg = cfg["model"]
    ema_model = ScoreNet(**model_cfg).to(device)
    ema_model.load_state_dict(ckpt["ema_model_state"])
    ema_model.eval()

    # ---- schedules ----
    T               = cfg["diffusion"]["T"]
    betas, alphas, abar = make_schedules(T, device=device)
    ls_mu, ls_sd    = prep_time_stats(abar)

    # ---- Laplace dataset ----
    lap_cfg  = cfg["laplace"]
    X_lap, Y_lap, _ = make_laplace_dataset(
        X, abar, T,
        N_pairs=lap_cfg["N_pairs"],
        data_dim=model_cfg["data_dim"],
        device="cpu",
        snr_gamma=lap_cfg["snr_gamma"],
    )
    loader_lap = make_laplace_loader(X_lap, Y_lap, batch=lap_cfg["batch"])

    # ---- build + fit LLLA ----
    print("Fitting LLLA...")
    la, wrapped, last_layer_name = build_llla(
        ema_model, abar, ls_mu, ls_sd,
        data_dim=model_cfg["data_dim"], device=device
    )
    la.fit(loader_lap)
    la.optimize_prior_precision(method="marglik")
    print(f"LLLA done. Last layer: {last_layer_name}")

    # ---- build + fit Full Hessian ----
    print("Fitting Full Hessian Laplace...")
    la_full, wrapped_full = build_full(
        ema_model, abar, ls_mu, ls_sd,
        data_dim=model_cfg["data_dim"], device=device
    )
    la_full.fit(loader_lap)
    la_full.optimize_prior_precision(method="marglik")
    print("Full Hessian done.")

    # ---- shared RNGs + start ----
    seed    = cfg["experiment"]["seed"]
    g_main  = torch.Generator(device=device).manual_seed(seed)
    g_mc    = torch.Generator(device=device).manual_seed(seed + 1)
    g_init  = torch.Generator(device=device).manual_seed(seed + 2)
    n       = cfg["llla"]["n_samples"]

    xt0 = torch.randn(n, model_cfg["data_dim"], device=device, generator=g_init)

    diffusion = DiffusionShim(betas.to(device))
    seq       = list(range(diffusion.num_timesteps))

    # ---- snapshot RNG state so both methods see identical noise ----
    state_main = g_main.get_state()
    state_mc   = g_mc.get_state()

    # ---- 1. BayesDiff + FLARE (LLLA) ----
    print("Running BayesDiff (LLLA)...")
    g_main.set_state(state_main)
    g_mc.set_state(state_mc)
    llla_cfg = cfg["llla"]
    x0_llla, u_bayes_llla, u_ep_llla, u_proj_llla, xt_path, a_path, b_path = sample_bayesdiff(
        diffusion=diffusion,
        model=wrapped.net,
        la=la,
        seq=seq,
        abar=abar, ls_mu=ls_mu, ls_sd=ls_sd,
        n=n, device=device,
        use_cov=llla_cfg["use_cov"],
        include_t4=llla_cfg["include_t4"],
        S_cov=llla_cfg["S_cov"],
        tau_gamma2=llla_cfg["tau_gamma2"],
        rng=g_main, rng_mc=g_mc, xt0=xt0,
        return_path=True,
        show_progress=True,
    )

    # ---- 2. FLARE (Full Hessian) on same path ----
    print("Running FLARE (Full Hessian) on fixed path...")
    fh_cfg = cfg["full_hessian"]
    u_proj_fullH = project_full(
        wrapped=wrapped_full,
        la_full=la_full,
        abar=abar,
        xt_path=xt_path,
        a_path=a_path,
        b_path=b_path,
        t_seq=seq,
        device=device,
        tau_gamma2=fh_cfg["tau_gamma2"],
        max_batch_gamma=fh_cfg["max_batch_gamma"],
        show_progress=True,
    )

    # ---- save results ----
    results = {
        "x0_llla":        x0_llla.cpu().numpy(),
        "u_bayes_llla":   u_bayes_llla.cpu().numpy(),
        "u_ep_llla":      u_ep_llla.cpu().numpy(),
        "u_proj_llla":    u_proj_llla.cpu().numpy(),
        "u_proj_fullH":   u_proj_fullH.cpu().numpy(),
    }
    out_path = os.path.join(args.out_dir, "results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()