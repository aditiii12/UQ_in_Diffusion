"""
Plot R10 trajectories colored by uncertainty scores.

Loads results.pkl produced by run_experiment.py and generates a 3-panel figure:
  Panel 1: trajectories colored by LLLA BayesDiff (u_bayes)
  Panel 2: trajectories colored by LLLA epistemic (u_ep)
  Panel 3: trajectories colored by Full-H epistemic projection (u_proj)

Usage
-----
python experiments/timeseries/plot_uncertainty.py \
    --results assets/results.pkl \
    --out assets/figures/uncertainty.pdf
"""

import argparse
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=str, default="assets/results.pkl")
    p.add_argument("--out",     type=str, default="assets/figures/uncertainty.pdf")
    p.add_argument("--max_lines", type=int, default=200)
    p.add_argument("--cmap",    type=str, default="roma_r",
                   help="cmcrameri colormap name, e.g. roma_r, vik, bam")
    return p.parse_args()


def load_cmap(name: str):
    try:
        import cmcrameri.cm as cmc
        return getattr(cmc, name)
    except Exception:
        print(f"cmcrameri not available or cmap '{name}' not found, falling back to viridis.")
        return plt.get_cmap("viridis")


def to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def plot_panel(ax, X, u, cmap, title, label_size=16, tick_size=13,
               max_lines=200, lw=1.0, alpha=0.85, y_lim=None):
    X = to_numpy(X)
    u = to_numpy(u)

    N, L = X.shape
    order = np.argsort(u)
    if N > max_lines:
        step = max(1, N // max_lines)
        order = order[::step]

    Xp = X[order]
    up = u[order]
    t  = np.arange(L)

    vmin, vmax = float(np.nanmin(u)), float(np.nanmax(u))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    norm = Normalize(vmin=vmin, vmax=vmax)

    segs   = np.stack([np.column_stack([t, y]) for y in Xp], axis=0)
    colors = cmap(norm(up))
    lc     = LineCollection(segs, colors=colors, linewidths=lw, alpha=alpha) # type: ignore
    ax.add_collection(lc)

    ax.set_xlim(0, L - 1)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    else:
        pad = 0.05 * max(1e-6, Xp.max() - Xp.min())
        ax.set_ylim(Xp.min() - pad, Xp.max() + pad)

    ax.set_title(title, fontsize=label_size)
    ax.tick_params(axis="both", labelsize=tick_size)
    ax.grid(True, alpha=0.25)

    return norm


def main():
    args = get_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # ── Load results ──────────────────────────────────────────────────────
    with open(args.results, "rb") as f:
        results = pickle.load(f)

    assert "diag" in results, "No 'diag' key in results — run with --hessian diag first."
    diag = results["diag"]

    x0     = diag["x0"]
    u_bayes = diag["u_bayes"]   # LLLA BayesDiff (t1+t2+t3)
    u_ep    = diag["u_ep"]      # LLLA epistemic (t1+t3)

    has_fullH = "full" in results
    if has_fullH:
        u_proj_fullH = results["full"]["u_proj"]
    else:
        print("No 'full' key found — plotting 2 panels (LLLA only).")

    n_panels = 3 if has_fullH else 2
    titles   = ["LLLA BayesDiff", "LLLA Epistemic"]
    if has_fullH:
        titles.append("Full-H Epistemic")

    cmap = load_cmap(args.cmap)

    # shared y limits
    X_np = to_numpy(x0)
    pad  = 0.05 * max(1e-6, X_np.max() - X_np.min())
    y_lim = (X_np.min() - pad, X_np.max() + pad)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axs = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.8), sharey=True)

    norms = []
    panels = [(x0, u_bayes), (x0, u_ep)]
    if has_fullH:
        panels.append((x0, min(u_proj_fullH, float("inf"))))  # type: ignore # clip to avoid outliers dominating the color scale

    for i, (ax, (X, u), title) in enumerate(zip(axs, panels, titles)):
        norm = plot_panel(ax, X, u, cmap, title,
                          max_lines=args.max_lines, y_lim=y_lim)
        norms.append(norm)
        if i == 0:
            ax.set_ylabel("value", fontsize=16)
        ax.set_xlabel("time step", fontsize=14)

    # colorbar on rightmost panel
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norms[-1])
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs[-1], fraction=0.046, pad=0.04)
    cbar.set_ticks([])
    # cbar.outline.set_visible(False)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"✅ Saved to {args.out}")
    plt.show()


if __name__ == "__main__":
    main()