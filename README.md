 UQ-Diffusion

Epistemic uncertainty quantification for DDPM score networks via Laplace approximation, implemented on synthetic time-series.

Implements and compares three BayesDiff variants:

- **LLLA** — Last-Layer Laplace Approximation with diagonal Hessian (fast)
- **FLARE** — Randomized subset Laplace, diagonal Hessian over randomly selected parameters (middle ground)
- **Full-Hessian** — Full-parameter Laplace with vmap'd Jacobians (exact, expensive)

---

## Repository Structure

```
uq-diffusion/
├── src/uqdiff/
│   ├── models/
│   │   └── scorenet_mlp.py        # FiLM-conditioned MLP score network
│   ├── diffusion/
│   │   ├── schedules.py           # Cosine beta schedule
│   │   ├── timecode.py            # Normalized logSNR time conditioning
│   │   └── ddpm.py                # DDPM forward/reverse process
│   ├── laplace/
│   │   ├── wrapper.py             # LaplaceWrapper: ScoreNet → [x_t, t_norm] interface
│   │   ├── dataset.py             # Laplace fitting dataset construction
│   │   ├── fit_laplace.py         # Unified Laplace builder/fitter (diag | flare | full)
│   │   ├── flare.py               # FLARE: randomized subset Laplace
│   │   ├── llla.py                # LLLA build + γ² computation
│   │   ├── bayesdiff_llla.py      # BayesDiff sampler (LLLA, diagonal)
│   │   └── bayesdiff_fullH.py     # BayesDiff samplers (full Hessian)
│   ├── fisher/
│   │   └── jacobian.py            # vmap Jacobians, robust Cholesky, γ² (full)
│   └── utils/
│       ├── data.py                # Synthetic data generation + DataLoader
│       ├── train.py               # Training loop (min-SNR, EMA, warmup cosine LR)
│       ├── ema.py                 # EMA model helpers
│       └── rng.py                 # Generator-aware randn_like
├── scripts/
│   ├── train.py                   # CLI training script
│   └── sample.py                  # CLI sampling + optional BayesDiff UQ
├── experiments/
│   └── timeseries/
│       ├── run_experiment.py      # Unified runner (--hessian diag|flare|full)
│       └── plot_uncertainty.py    # Trajectory visualization colored by uncertainty
├── configs/
│   └── r10_default.yaml           # Default hyperparameters
├── tests/                         # pytest unit tests
├── assets/                        # Figures and saved results
└── checkpoints/                   # Model checkpoints
```

---

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.1.

---

## Quickstart

### 1. Train

```bash
python scripts/train.py \
    --hidden_dim 16 --time_dim 32 --n_blocks 2 \
    --data_dim 10 --n_samples 5000 \
    --epochs 600 --lr 5e-4 \
    --save_name ddpm_r10_final.pth
```

### 2. Sample (plain DDPM)

```bash
python scripts/sample.py \
    --checkpoint checkpoints/ddpm_r10_final.pth \
    --n 2000 --out_dir assets
```

### 3. Sample with BayesDiff (LLLA uncertainty)

```bash
python scripts/sample.py \
    --checkpoint checkpoints/ddpm_r10_final.pth \
    --bayesdiff --n 200 --out_dir assets
```

Outputs `assets/x0_bd.npy`, `assets/u_ep_bd.npy`, `assets/u_proj_bd.npy`.

---

## Experiments

### LLLA (fast)

```bash
python experiments/timeseries/run_experiment.py \
    --checkpoint checkpoints/ddpm_r10_final.pth \
    --hessian diag --n 200 --seed 20251022 \
    --out assets/results.pkl
```

### FLARE (randomized subset)

```bash
# 10% of parameters randomly selected
python experiments/timeseries/run_experiment.py \
    --checkpoint checkpoints/ddpm_r10_final.pth \
    --hessian flare --flare_frac 0.1 --flare_seed 42 \
    --n 200 --seed 20251022 --out assets/results.pkl

# 30% of parameters
python experiments/timeseries/run_experiment.py \
    --checkpoint checkpoints/ddpm_r10_final.pth \
    --hessian flare --flare_frac 0.3 --flare_seed 42 \
    --n 200 --seed 20251022 --out assets/results.pkl
```

### Full-Hessian (exact, small models only)

```bash
python experiments/timeseries/run_experiment.py \
    --checkpoint checkpoints/ddpm_r10_final.pth \
    --hessian full --n 20 --seed 20251022 \
    --out assets/results.pkl
```

All three runs append to the same `results.pkl` under `results["diag"]`, `results["flare"]`, and `results["full"]`.

### Fit Laplace once, sample many times

```bash
# Fit and save (no sampling)
python experiments/timeseries/run_experiment.py \
    --checkpoint checkpoints/ddpm_r10_final.pth \
    --hessian diag \
    --save_laplace checkpoints/la_diag.pkl \
    --skip_sampling

# Load saved Laplace and sample (fast)
python experiments/timeseries/run_experiment.py \
    --checkpoint checkpoints/ddpm_r10_final.pth \
    --hessian diag \
    --load_laplace checkpoints/la_diag.pkl \
    --n 200 --seed 20251022 --out assets/results.pkl
```

Works for all three hessian types.

### Plot uncertainty

```bash
python experiments/timeseries/plot_uncertainty.py \
    --results assets/results.pkl \
    --out assets/figures/uncertainty.pdf
```

---
