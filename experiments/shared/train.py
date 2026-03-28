"""
experiments/shared/train.py
----------------------------
Training loop for the score network with:
  - Min-SNR loss weighting (Hang et al. 2023)
  - EMA model tracking
  - Warmup + cosine LR schedule
  - Debug logging
"""

from __future__ import annotations
import copy
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from uqdiff.utils import prep_time_stats, time_code_from_abar


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

def make_ema(model: nn.Module) -> nn.Module:
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


@torch.no_grad()
def ema_update(model: nn.Module, ema_model: nn.Module, decay: float = 0.999):
    for p, q in zip(model.parameters(), ema_model.parameters()):
        q.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def make_warmup_cosine(
    optimizer: optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 1000,
    eta_min: float = 1e-6,
) -> optim.lr_scheduler.LambdaLR:
    base_lr    = optimizer.defaults["lr"]
    min_factor = eta_min / max(base_lr, 1e-12)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * t))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Loss curve helper
# ---------------------------------------------------------------------------

def save_loss_plot(epoch_losses, step_losses, path: str):
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)

    def ema(y, b=0.98):
        y = np.asarray(y, float)
        m, out = 0.0, []
        for v in y:
            m = b * m + (1 - b) * v
            out.append(m)
        return np.array(out)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(epoch_losses, lw=1, alpha=0.7, label="epoch avg")
    ax[0].plot(ema(epoch_losses, 0.8), lw=2, label="EMA(0.8)")
    ax[0].set_title("Epoch loss"); ax[0].grid(alpha=0.3); ax[0].legend()

    decim = max(1, len(step_losses) // 8000)
    steps = range(0, len(step_losses), decim)
    vals  = np.asarray(step_losses, float)[::decim]
    ax[1].plot(steps, vals, lw=0.8, alpha=0.35, label="step (decimated)")
    ax[1].plot(steps, ema(vals, 0.98), lw=2, label="EMA(0.98)")
    ax[1].set_title("Step loss"); ax[1].grid(alpha=0.3); ax[1].legend()

    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_score_model(
    model: nn.Module,
    ema_model: nn.Module,
    loader: DataLoader,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    abar: torch.Tensor,
    epochs: int = 200,
    lr: float = 5e-4,
    use_min_snr: bool = True,
    snr_gamma: float = 5.0,
    clip_grad: float = 0.5,
    ema_decay: float = 0.999,
    eta_min: float = 1e-6,
    warmup_steps: int = 600,
    device: str = "cuda",
    save_dir: str = "checkpoints",
    save_name: str = "ddpm_final.pth",
    # debug controls
    debug: bool = True,
    debug_first_n_epochs: int = 2,
    debug_every_steps: int = 1000,
    print_every_epoch: int = 10,
) -> tuple[list, list]:
    """
    Train a score model with min-SNR weighting, EMA, and warmup cosine LR.
    Returns (epoch_losses, step_losses).
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)

    model.to(device)
    ema_model.to(device)
    betas  = betas.to(device).float()
    alphas = alphas.to(device).float()
    abar   = abar.to(device).float()
    T      = betas.numel()

    ls_mu, ls_sd = prep_time_stats(abar)

    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)

    steps_per_epoch = len(loader)
    total_steps     = max(1, epochs * steps_per_epoch)
    scheduler       = make_warmup_cosine(opt, total_steps, warmup_steps, eta_min)

    epoch_losses, step_losses, global_step = [], [], 0
    epoch_bar = tqdm(range(epochs), desc="Epochs")

    for epoch in epoch_bar:
        running_loss, n_batches = 0.0, 0
        batch_bar = tqdm(loader, leave=False, desc=f"Epoch {epoch+1}/{epochs}")

        for b_idx, (x0_cpu,) in enumerate(batch_bar):
            x0 = x0_cpu.to(device, non_blocking=True)
            B  = x0.shape[0]

            t      = torch.randint(0, T, (B,), device=device)
            abar_t = abar[t].unsqueeze(1)

            noise  = torch.randn_like(x0)
            xt     = torch.sqrt(abar_t) * x0 + torch.sqrt(1.0 - abar_t) * noise
            t_code = time_code_from_abar(abar_t, ls_mu, ls_sd)
            pred   = model(xt, t_code)

            if use_min_snr:
                snr  = abar_t / (1.0 - abar_t + 1e-8)
                w    = torch.minimum(snr, torch.full_like(snr, snr_gamma)) / (snr + 1e-8)
                loss = (w * (pred - noise).pow(2)).mean()
            else:
                loss = (pred - noise).pow(2).mean()

            # debug logging
            do_dbg = debug and (
                (epoch < debug_first_n_epochs and b_idx == 0)
                or (global_step % debug_every_steps == 0)
            )
            if do_dbg:
                with torch.no_grad():
                    epoch_bar.write(
                        f"[dbg] epoch={epoch+1} step={global_step} "
                        f"loss={loss.item():.3e} "
                        f"lr={scheduler.get_last_lr()[0]:.2e}"
                    )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()
            scheduler.step()
            global_step += 1

            ema_update(model, ema_model, decay=ema_decay)

            running_loss += float(loss)
            n_batches    += 1
            step_losses.append(float(loss))
            batch_bar.set_postfix(loss=f"{float(loss):.3e}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_loss = running_loss / max(1, n_batches)
        epoch_losses.append(avg_loss)

        epoch_bar.write(
            f"[epoch {epoch+1:04d}/{epochs:04d}] avg_loss={avg_loss:.3e} "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )
        if (epoch + 1) % print_every_epoch == 0:
            epoch_bar.write(f"=== E{epoch+1}: avg_loss={avg_loss:.3e} ===")
        if (epoch + 1) % 50 == 0:
            save_loss_plot(epoch_losses, step_losses,
                           os.path.join(save_dir, "loss_curves_progress.png"))

        epoch_bar.set_postfix(avg_loss=f"{avg_loss:.3e}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

    torch.save(
        {"model_state": model.state_dict(), "ema_model_state": ema_model.state_dict()},
        save_path
    )
    print(f"Saved to {save_path}")
    return epoch_losses, step_losses