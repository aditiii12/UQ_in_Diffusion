import math
import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from uqdiff.utils.ema import ema_update


def make_warmup_cosine(
    optimizer: optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 1000,
    eta_min: float = 1e-6,
) -> optim.lr_scheduler.LambdaLR:
    base_lr = optimizer.defaults["lr"]
    min_factor = eta_min / max(base_lr, 1e-12)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * t))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_ddpm(
    model: nn.Module,
    ema_model: nn.Module,
    loader,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    abar: torch.Tensor,
    epochs: int = 200,
    lr: float = 5e-4,
    snr_gamma: float = 5.0,
    use_min_snr: bool = True,
    clip_grad: float = 0.5,
    ema_decay: float = 0.999,
    eta_min: float = 1e-6,
    warmup_steps: int = 600,
    device: str = "cuda",
    save_dir: str = "checkpoints",
    save_name: str = "ddpm_final.pth",
):
    """
    Train a DDPM score network with cosine schedule, min-SNR weighting,
    warmup+cosine LR schedule, and EMA.

    Returns
    -------
    epoch_losses : list[float]
    step_losses  : list[float]
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)

    model.to(device); ema_model.to(device)
    betas  = betas.to(device).float()
    alphas = alphas.to(device).float()
    abar   = abar.to(device).float()
    T = betas.numel()

    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)

    steps_per_epoch = len(loader)
    total_steps = max(1, epochs * steps_per_epoch)
    scheduler = make_warmup_cosine(opt, total_steps, warmup_steps=warmup_steps, eta_min=eta_min)

    # Precompute normalized logSNR stats for time conditioning
    from uqdiff.diffusion.timecode import prep_time_stats, time_code_from_abar
    ls_mu, ls_sd = prep_time_stats(abar)

    epoch_losses, step_losses = [], []
    epoch_bar = tqdm(range(epochs), desc="Epochs")

    for epoch in epoch_bar:
        running, n_batches = 0.0, 0
        batch_bar = tqdm(loader, leave=False, desc=f"Epoch {epoch+1}/{epochs}")

        for (x0_cpu,) in batch_bar:
            x0 = x0_cpu.to(device, non_blocking=True)
            t  = torch.randint(0, T, (x0.shape[0],), device=device)
            abar_t = abar[t].unsqueeze(1)

            noise = torch.randn_like(x0)
            xt    = torch.sqrt(abar_t) * x0 + torch.sqrt(1.0 - abar_t) * noise

            t_code = time_code_from_abar(abar_t, ls_mu, ls_sd)
            pred   = model(xt, t_code)

            if use_min_snr:
                snr = abar_t / (1.0 - abar_t + 1e-8)
                w   = torch.minimum(snr, torch.full_like(snr, snr_gamma)) / (snr + 1e-8)
                loss = ((w * (pred - noise) ** 2).sum(dim=-1)).mean()
            else:
                loss = ((pred - noise) ** 2).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()
            scheduler.step()

            ema_update(model, ema_model, decay=ema_decay)

            running += float(loss); n_batches += 1
            step_losses.append(float(loss))
            batch_bar.set_postfix(loss=f"{float(loss):.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg = running / max(1, n_batches)
        epoch_losses.append(avg)
        epoch_bar.set_postfix(avg_loss=f"{avg:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

    torch.save(
        {"model_state": model.state_dict(), "ema_model_state": ema_model.state_dict()},
        save_path,
    )
    print(f"✅ Saved to {save_path}")
    return epoch_losses, step_losses














