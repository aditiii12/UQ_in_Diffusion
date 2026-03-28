"""
Sample from a trained DDPM and optionally run BayesDiff (LLLA) uncertainty estimation.

Usage
-----
# Plain DDPM samples
python scripts/sample.py --checkpoint checkpoints/ddpm_r10_final.pth --n 2000

# BayesDiff LLLA uncertainty
python scripts/sample.py --checkpoint checkpoints/ddpm_r10_final.pth \
    --bayesdiff --laplace_data data/X_train.npy --n 200
"""

import argparse
import os
import torch
import numpy as np

from uqdiff.models.scorenet_mlp import ScoreNet
from uqdiff.diffusion.schedules import make_schedules
from uqdiff.diffusion.timecode import prep_time_stats
from uqdiff.diffusion.ddpm import sample_ddpm
from uqdiff.utils.ema import make_ema


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    type=str,   required=True)
    p.add_argument("--data_dim",      type=int,   default=10)
    p.add_argument("--hidden_dim",    type=int,   default=512)
    p.add_argument("--time_dim",      type=int,   default=32)
    p.add_argument("--n_blocks",      type=int,   default=2)
    p.add_argument("--T",             type=int,   default=1000)
    p.add_argument("--n",             type=int,   default=2000)
    p.add_argument("--tau",           type=float, default=1.0)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--out_dir",       type=str,   default="assets")
    # BayesDiff flags
    p.add_argument("--bayesdiff",     action="store_true")
    p.add_argument("--laplace_data",  type=str,   default=None,
                   help="Path to .npy training data for Laplace fitting")
    p.add_argument("--N_pairs",       type=int,   default=100_000)
    return p.parse_args()


def main():
    args = get_args()
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    # Load model
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = ScoreNet(args.data_dim, args.hidden_dim, args.time_dim, args.n_blocks)
    ema_model = make_ema(model)
    model.load_state_dict(ckpt["model_state"])
    ema_model.load_state_dict(ckpt["ema_model_state"])
    ema_model.to(device).eval()

    betas, alphas, abar = make_schedules(args.T)
    ls_mu, ls_sd = prep_time_stats(abar)

    g = torch.Generator(device=device).manual_seed(args.seed)

    if not args.bayesdiff:
        # Plain DDPM samples
        samples = sample_ddpm(
            ema_model, betas, alphas, abar, ls_mu, ls_sd,
            T=args.T, data_dim=args.data_dim, n=args.n,
            device=device, gen=g, tau=args.tau,
        )
        out_path = os.path.join(args.out_dir, "samples.npy")
        np.save(out_path, samples.cpu().numpy())
        print(f"Saved {samples.shape} samples to {out_path}")

    else:
        # BayesDiff LLLA
        if args.laplace_data:
            X = np.load(args.laplace_data)
        else:
            from uqdiff.utils.data import make_gaussian_timeseries
            print("No --laplace_data provided, generating synthetic R10 data on the fly.")
            X = make_gaussian_timeseries(length=args.data_dim, n_samples=5000, std=0.1, seed=0)

        from uqdiff.laplace.llla import build_llla_lastlayer_diag
        from uqdiff.laplace.dataset import make_laplace_dataset
        from uqdiff.laplace.bayesdiff_llla import DiffusionShim, sample_bayesdiff_samepath
        from torch.utils.data import TensorDataset, DataLoader

        la, wrapped, _ = build_llla_lastlayer_diag(
            ema_model, abar, ls_mu, ls_sd, data_dim=args.data_dim, device=device,
        )
        X_lap, Y_lap, _ = make_laplace_dataset(
            X, abar, args.T, N_pairs=args.N_pairs, data_dim=args.data_dim, device=device,
        )
        loader_lap = DataLoader(
            TensorDataset(X_lap, Y_lap), batch_size=4096, shuffle=True, drop_last=True,
        )
        n_batches = len(loader_lap)
        print(f"[1/3] Fitting LLLA ({n_batches} batches, N_pairs={args.N_pairs}) ...")
        la.fit(loader_lap)
        print("[2/3] Optimizing prior precision ...")
        la.optimize_prior_precision(method="marglik") #type:ignore
        print("[3/3] Running BayesDiff reverse chain ...")

        diffusion = DiffusionShim(betas.to(device))
        seq = list(range(diffusion.num_timesteps))

        x0, u_bayes, u_ep, u_proj = sample_bayesdiff_samepath(. #type:ignore
            diffusion=diffusion, model=wrapped.net, la=la, seq=seq,
            abar=abar.to(device), ls_mu=ls_mu.to(device), ls_sd=ls_sd.to(device),
            n=args.n, data_dim=args.data_dim, device=device,
            rng=g, show_progress=True,
        )
        np.save(os.path.join(args.out_dir, "x0_bd.npy"),    x0.cpu().numpy())
        np.save(os.path.join(args.out_dir, "u_ep_bd.npy"),  u_ep.cpu().numpy())
        np.save(os.path.join(args.out_dir, "u_proj_bd.npy"), u_proj.cpu().numpy())
        print(f"BayesDiff samples + uncertainty saved to {args.out_dir}/")


if __name__ == "__main__":
    main()










