"""
Train a DDPM score network on synthetic Gaussian time-series.

Usage
-----
python scripts/train.py \
    --data_dim 10 --n_samples 5000 --T 1000 \
    --epochs 600 --lr 5e-4 --hidden_dim 512 \
    --save_dir checkpoints --save_name ddpm_r10_final.pth
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from uqdiff.models.scorenet_mlp import ScoreNet
from uqdiff.diffusion.schedules import make_schedules
from uqdiff.diffusion.timecode import prep_time_stats
from uqdiff.utils.data import make_gaussian_timeseries, make_dataloader
from uqdiff.utils.ema import make_ema
from uqdiff.utils.train import train_ddpm


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dim",    type=int,   default=10)
    p.add_argument("--n_samples",   type=int,   default=5000)
    p.add_argument("--std",         type=float, default=0.1)
    p.add_argument("--seed",        type=int,   default=0)
    p.add_argument("--T",           type=int,   default=1000)
    p.add_argument("--epochs",      type=int,   default=600)
    p.add_argument("--lr",          type=float, default=5e-4)
    p.add_argument("--batch_size",  type=int,   default=512)
    p.add_argument("--hidden_dim",  type=int,   default=512)
    p.add_argument("--time_dim",    type=int,   default=32)
    p.add_argument("--n_blocks",    type=int,   default=2)
    p.add_argument("--snr_gamma",   type=float, default=5.0)
    p.add_argument("--save_dir",    type=str,   default="checkpoints")
    p.add_argument("--save_name",   type=str,   default="ddpm_r10_final.pth")
    p.add_argument("--plot",        action="store_true")
    return p.parse_args()


def main():
    args = get_args()
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # Data
    X = make_gaussian_timeseries(
        length=args.data_dim, n_samples=args.n_samples,
        std=args.std, seed=args.seed,
    )
    print(f"Data shape: {X.shape}")

    loader = make_dataloader(X, batch_size=args.batch_size, device=device)

    # Model
    model     = ScoreNet(args.data_dim, args.hidden_dim, args.time_dim, args.n_blocks)
    ema_model = make_ema(model)

    # Schedules
    betas, alphas, abar = make_schedules(args.T)

    # Train
    epoch_losses, step_losses = train_ddpm(
        model=model, ema_model=ema_model,
        loader=loader,
        betas=betas, alphas=alphas, abar=abar,
        epochs=args.epochs, lr=args.lr,
        snr_gamma=args.snr_gamma, use_min_snr=True,
        device=device,
        save_dir=args.save_dir, save_name=args.save_name,
    )

    if args.plot:
        plt.figure(figsize=(6, 4))
        plt.plot(epoch_losses, marker="o", markersize=3)
        plt.title("Epoch Loss")
        plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
        plt.grid(True); plt.tight_layout()
        plt.savefig("assets/train_loss.png", dpi=150)
        plt.show()


if __name__ == "__main__":
    main()














