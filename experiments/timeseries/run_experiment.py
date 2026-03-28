"""
Unified BayesDiff experiment runner for R10 time-series.

Runs epistemic uncertainty estimation with either LLLA (diagonal last-layer)
or full-Hessian Laplace, and saves results to a pkl keyed by hessian type.

Usage
-----
# LLLA (fast, n=200)
python experiments/timeseries/run_experiment.py \
    --checkpoint checkpoints/ddpm_r10_final.pth \
    --hessian diag --n 200 --seed 20251022 \
    --out assets/results.pkl

# Full-Hessian (slow, keep n small)
python experiments/timeseries/run_experiment.py \
    --checkpoint checkpoints/ddpm_r10_final.pth \
    --hessian full --n 20 --seed 20251022 \
    --out assets/results.pkl

Both runs append to the same pkl under results["diag"] and results["full"].
"""

import argparse
import os
import pickle

import torch

from uqdiff.models.scorenet_mlp import ScoreNet
from uqdiff.diffusion.schedules import make_schedules
from uqdiff.diffusion.timecode import prep_time_stats
from uqdiff.utils.data import make_gaussian_timeseries
from uqdiff.utils.ema import make_ema
from uqdiff.laplace.fit_laplace import build_and_fit_laplace
from uqdiff.laplace.bayesdiff_llla import DiffusionShim, sample_bayesdiff_samepath
from uqdiff.laplace.bayesdiff_fullH import project_fullH_on_fixed_path


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  type=str,   required=True)
    p.add_argument("--hessian",     type=str,   default="diag", choices=["diag", "full"],
                   help="Hessian approximation: 'diag' (LLLA) or 'full'")
    p.add_argument("--data_dim",    type=int,   default=10)
    p.add_argument("--hidden_dim",  type=int,   default=16)
    p.add_argument("--time_dim",    type=int,   default=32)
    p.add_argument("--n_blocks",    type=int,   default=2)
    p.add_argument("--T",           type=int,   default=1000)
    p.add_argument("--n",           type=int,   default=200)
    p.add_argument("--seed",        type=int,   default=20251022)
    p.add_argument("--N_pairs",     type=int,   default=100_000)
    p.add_argument("--n_samples",   type=int,   default=5000)
    p.add_argument("--out",         type=str,   default="assets/results.pkl")
    return p.parse_args()


def load_model(args, device):
    model     = ScoreNet(args.data_dim, args.hidden_dim, args.time_dim, args.n_blocks)
    ema_model = make_ema(model)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    ema_model.load_state_dict(ckpt["ema_model_state"])
    ema_model.to(device).eval()
    print(f"Loaded: {args.checkpoint}")
    return ema_model


def main():
    args = get_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  hessian: {args.hessian}  |  n: {args.n}")

    # ── Data ──────────────────────────────────────────────────────────────
    X = make_gaussian_timeseries(length=args.data_dim, n_samples=args.n_samples, std=0.1)
    print(f"Data: {X.shape}")

    # ── Model ─────────────────────────────────────────────────────────────
    ema_model = load_model(args, device)

    # ── Schedules ─────────────────────────────────────────────────────────
    betas, _, abar = make_schedules(args.T)
    ls_mu, ls_sd   = prep_time_stats(abar)
    abar  = abar.to(device)
    ls_mu = ls_mu.to(device)
    ls_sd = ls_sd.to(device)

    # ── Laplace ───────────────────────────────────────────────────────────
    la, wrapped = build_and_fit_laplace(
        ema_model, X, abar, ls_mu, ls_sd,
        T=args.T, data_dim=args.data_dim, device=device,
        hessian=args.hessian, N_pairs=args.N_pairs,
    )

    # ── Shared init ───────────────────────────────────────────────────────
    g_main = torch.Generator(device=device).manual_seed(args.seed)
    g_mc   = torch.Generator(device=device).manual_seed(args.seed + 1)
    g_init = torch.Generator(device=device).manual_seed(args.seed + 2)
    xt0    = torch.randn(args.n, args.data_dim, device=device, generator=g_init)

    diffusion = DiffusionShim(betas.to(device))
    seq = list(range(diffusion.num_timesteps))

    # ── Run ───────────────────────────────────────────────────────────────
    if args.hessian == "diag":
        print("Running BayesDiff [LLLA] …")
        x0, u_bayes, u_ep, u_proj, xt_path, a_path, b_path = sample_bayesdiff_samepath(  # type: ignore[misc]
            diffusion=diffusion, model=wrapped.net, la=la, seq=seq,
            abar=abar, ls_mu=ls_mu, ls_sd=ls_sd,
            n=args.n, data_dim=args.data_dim, device=device,
            use_cov=True, include_t4=False,
            S_cov=64, tau_gamma2=1.0,
            rng=g_main, rng_mc=g_mc, xt0=xt0,
            return_path=True, show_progress=True,
        )
        print(f"u_ep mean={u_ep.mean():.4f}  u_proj mean={u_proj.mean():.4f}")

        run_result = {
            "x0":      x0.cpu(),
            "u_ep":    u_ep.cpu(),
            "u_proj":  u_proj.cpu(),
            "u_bayes": u_bayes.cpu(),
            "xt_path": xt_path.cpu(),
            "a_path":  a_path.cpu(),
            "b_path":  b_path.cpu(),
        }

    else:  # full
        print("Running BayesDiff [full-H] — projection on LLLA path …")

        # For full-H we need a fixed path — run a lightweight LLLA sample first
        # to get xt_path, then replay full-H γ² on it
        from uqdiff.laplace.llla import build_llla_lastlayer_diag
        la_diag, wrapped_diag, _ = build_llla_lastlayer_diag(
            ema_model, abar, ls_mu, ls_sd, data_dim=args.data_dim, device=device,
        )
        from uqdiff.laplace.dataset import make_laplace_dataset, make_laplace_loader
        X_lap, Y_lap, _ = make_laplace_dataset(
            X, abar, args.T, N_pairs=args.N_pairs,
            data_dim=args.data_dim, device=device,
        )
        la_diag.fit(make_laplace_loader(X_lap, Y_lap))
        la_diag.optimize_prior_precision(method="marglik")  # type: ignore[attr-defined]

        _, _, _, _, xt_path, a_path, b_path = sample_bayesdiff_samepath(  # type: ignore[misc]
            diffusion=diffusion, model=wrapped_diag.net, la=la_diag, seq=seq,
            abar=abar, ls_mu=ls_mu, ls_sd=ls_sd,
            n=args.n, data_dim=args.data_dim, device=device,
            use_cov=False, include_t4=False,
            rng=g_main, rng_mc=g_mc, xt0=xt0,
            return_path=True, show_progress=False,
        )

        u_proj = project_fullH_on_fixed_path(
            wrapped=wrapped, la_full=la, abar=abar,
            xt_path=xt_path, a_path=a_path, b_path=b_path,
            t_seq=seq, device=device, show_progress=True,
        )
        print(f"u_proj mean={u_proj.mean():.4f}")

        run_result = {
            "u_proj":  u_proj.cpu(),
            "xt_path": xt_path.cpu(),
            "a_path":  a_path.cpu(),
            "b_path":  b_path.cpu(),
        }

    # ── Save (append to existing pkl if present) ──────────────────────────
    if os.path.exists(args.out):
        with open(args.out, "rb") as f:
            all_results = pickle.load(f)
    else:
        all_results = {}

    run_result["meta"] = vars(args)
    all_results[args.hessian] = run_result

    with open(args.out, "wb") as f:
        pickle.dump(all_results, f)
    print(f"✅ Saved results['{args.hessian}'] to {args.out}")


if __name__ == "__main__":
    main()


















