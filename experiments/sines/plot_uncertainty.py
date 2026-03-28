"""
experiments/sines/plot_uncertainty.py
--------------------------------------
Visualization for the sines UQ experiment.

Loads results.pkl from run_uq.py and produces:
  - 4-panel trajectory plot colored by uncertainty
    (BayesDiff LLLA, FLARE LLLA, FLARE Full Hessian)
  - Per-panel independent color normalization

Usage:
    python experiments/sines/plot_uncertainty.py --results assets/sines/results.pkl
"""

from __future__ import annotations
import argparse
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=str, default="assets/sines/results.pkl")
    p.add_argument("--out_dir", type=str, default="assets/sines")
    p.add_argument("--max_lines", type=int, default=600)
    p.add_argument("--cmap", type=str, default="viridis")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Core plotting function
# ---------------------------------------------------------------------------

def plot_panels(
    sets: list,            # list of (X, u) pairs
    titles: list,
    max_lines: int = 600,
    sort_by_u: bool = True,
    cmap="viridis",
    lw: float = 1.2,
    alpha: float = 0.9,
    figsize=None,
    sharey: bool = True,
    label_size: int = 18,
    tick_size: int = 15,
    save_path: str = None,
):
    n_panels = len(sets)
    if figsize is None:
        figsize = (5.0 * n_panels, 4.8)

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    Xs, us = [], []
    for X, u in sets:
        Xs.append(np.asarray(X))
        us.append(np.asarray(u))

    if sharey:
        y_min = min(X.min() for X in Xs)
        y_max = max(X.max() for X in Xs)
        y_pad = 0.05 * max(1e-6, y_max - y_min)

    fig, axs = plt.subplots(1, n_panels, figsize=figsize, sharey=sharey)
    if n_panels == 1:
        axs = [axs]

    for i, (X, u, ax, title) in enumerate(zip(Xs, us, axs, titles)):
        N, L  = X.shape
        order = np.argsort(u) if sort_by_u else np.arange(N)
        if N > max_lines:
            order = order[::max(1, N // max_lines)]

        Xp = X[order]
        up = u[order]
        t  = np.arange(L)

        vmin = float(np.nanmin(u))
        vmax = float(np.nanmax(u))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        norm = Normalize(vmin=vmin, vmax=vmax)

        segs   = np.stack([np.column_stack([t, y]) for y in Xp], axis=0)
        colors = cmap(norm(up))
        lc     = LineCollection(segs, colors=colors, linewidths=lw, alpha=alpha)
        ax.add_collection(lc)

        ax.set_xlim(0, L - 1)
        if sharey:
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
        else:
            ymin, ymax = Xp.min(), Xp.max()
            pad = 0.05 * max(1e-6, ymax - ymin)
            ax.set_ylim(ymin - pad, ymax + pad)

        ax.set_title(title, fontsize=label_size)
        ax.set_xlabel("time index", fontsize=label_size)
        if i == 0:
            ax.set_ylabel("value", fontsize=label_size)
        ax.tick_params(labelsize=tick_size)
        ax.grid(True, alpha=0.3)

        # colorbar (no ticks, gradient only)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([])
        cbar.outline.set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")
    plt.show()
    return fig, axs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.results, "rb") as f:
        res = pickle.load(f)

    x0          = res["x0_llla"]
    u_bayes     = res["u_bayes_llla"]
    u_proj_llla = res["u_proj_llla"]
    u_proj_fullH = res["u_proj_fullH"]

    plot_panels(
        sets=[
            (x0, u_bayes),
            (x0, u_proj_llla),
            (x0, u_proj_fullH),
        ],
        titles=[
            "BayesDiff (LLLA)",
            "FLARE (LLLA)",
            "FLARE (Full Hessian)",
        ],
        max_lines=args.max_lines,
        cmap=args.cmap,
        save_path=os.path.join(args.out_dir, "uncertainty_panels.pdf"),
    )


if __name__ == "__main__":
    main()