# Official Impletentation of FLARE (AISTATS 2026)

Epistemic uncertainty quantification for diffusion models via Laplace approximation (AISTATS 2026)

This repo implements **FLARE** (Fisher-Laplace Approximation for Randomized Epistemic uncertainty), which transports per-step epistemic variance from the score network's Laplace posterior to the generated samples via the closed-form DDPM transport factor. Paper Link - https://arxiv.org/pdf/2602.09170

---

## Method

At each reverse diffusion step, we compute the epistemic variance of the score prediction:

$$\gamma^2_t = \text{diag}(J_t \Sigma J_t^\top)$$

where $J_t$ is the Jacobian of the score network at $(x_t, t)$ and $\Sigma$ is the Laplace posterior covariance.

FLARE then projects this to $x_0$ via the accumulated transport factor:

$$u_{\text{proj}} = \sum_t \left(\prod_{s < t} a_s\right)^2 b_t^2 \gamma^2_t$$

This is compared against the full **BayesDiff** variance recursion:

$$V_{t-1} = a_t^2 V_t + 2a_t b_t \text{Cov}[x_t, \varepsilon_t] + b_t^2 \gamma^2_t + \tilde{\beta}_t$$

Three UQ variants are supported:
- **LLLA** — last-layer diagonal (fast, closed-form $\gamma^2$)
- **FLARE** — random subnetwork full Hessian (practical for larger models over full Hessian)
- **Bayesdiff** — methodology from Kou et. al. (2024)

---

## Structure
```
uq-diffusion/
├── src/uqdiff/                  # pip-installable UQ package
│   ├── utils.py                 # timecodes, robust_chol, randn_like_gen
│   ├── schedules.py             # cosine schedule, compute_alpha
│   ├── ddpm.py                  # forward/reverse diffusion process
│   └── laplace/
│       ├── core.py              # LaplaceWrapper, dataset, build_llla/subnet/full
│       ├── precision.py         # γ² computation (diag, vmap full, subnet)
│       ├── bayesdiff.py         # BayesDiff sampler (LLLA)
│       └── flare.py             # FLARE projection (LLLA, full, subnet)
│
├── experiments/
│   ├── shared/
│   │   ├── scorenet.py          # FiLM-conditioned MLP score network
│   │   ├── train.py             # training loop (min-SNR, EMA, warmup cosine)
│   │   └── data.py              # synthetic time-series generators
│   ├── sines/                   # R10 experiment (full Hessian feasible)
│   │   ├── config.yaml
│   │   ├── run_uq.py
│   │   └── plot_uncertainty.py
│   └── chirp/                   # R80 experiment (SubnetLaplace)
│       ├── config.yaml
│       ├── run_uq.py
│       └── plot_uncertainty.py
```

---

## Installation
```bash
git clone https://github.com/aditiii12/UQ_in_Diffusion.git
cd UQ_in_Diffusion
pip install -e ".[experiments]"
```

---

## Checkpoints

Pretrained checkpoints for both experiments are available in `checkpoints/`.

| Experiment | Dataset         | data_dim | Model                        |
|------------|-----------------|----------|------------------------------|
| sines      | sine / -sine    | 10       | ScoreNet (hidden=512, blocks=2) |
| chirp      | FM / AM / damped| 80       | ScoreNet (hidden=128, blocks=0) |

---

## Running the UQ experiments

**Sines (Full Hessian):**
```bash
python experiments/sines/run_uq.py \
    --checkpoint checkpoints/sines/ddpm_sines.pth \
    --device cuda
```

**Chirp (SubnetLaplace):**
```bash
python experiments/chirp/run_uq.py \
    --checkpoint checkpoints/chirp/ddpm_chirp.pth \
    --device cuda
```

**Plotting:**
```bash
python experiments/sines/plot_uncertainty.py --results assets/sines/results.pkl
python experiments/chirp/plot_uncertainty.py --results assets/chirp/results.pkl
```

---

## Using the package with your own model

The `uqdiff` package is model-agnostic. You only need a score model that maps `(x_t, t_code) -> eps`. Wrap it with `LaplaceWrapper`, fit Laplace, and run FLARE:
```python
from uqdiff.schedules import make_schedules
from uqdiff.utils import prep_time_stats
from uqdiff.laplace.core import (
    LaplaceWrapper, DiffusionShim,
    make_laplace_dataset, make_laplace_loader,
    build_llla,
)
from uqdiff.laplace.bayesdiff import sample_bayesdiff

# 1. build schedules
betas, alphas, abar = make_schedules(T=1000)
ls_mu, ls_sd = prep_time_stats(abar)

# 2. fit LLLA on your trained model
la, wrapped, _ = build_llla(your_ema_model, abar, ls_mu, ls_sd, data_dim=D)
X_lap, Y_lap, _ = make_laplace_dataset(X_train, abar, T=1000, N_pairs=100_000, data_dim=D)
la.fit(make_laplace_loader(X_lap, Y_lap))
la.optimize_prior_precision(method="marglik")

# 3. sample with uncertainty
diffusion = DiffusionShim(betas)
x0, u_bayes, u_ep, u_proj = sample_bayesdiff(
    diffusion, wrapped.net, la,
    seq=list(range(1000)),
    abar=abar, ls_mu=ls_mu, ls_sd=ls_sd,
    n=500, device="cuda",
)
```

---

## Citation

If citing or using the code, please cite:
```bibtex
@inproceedings{gupta2026flare,
  title     = {Quantifying Epistemic Uncertainty in Diffusion Models},
  booktitle = {AISTATS},
  year      = {2026}
}
```
