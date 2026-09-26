"""
check_indexability.py  (synthetic-stationary)

Numerically verifies Whittle indexability for each arm type via finite-horizon
backward induction over a discretized state grid.

Method:
  For each arm type m and subsidy λ, solve the single-arm MDP with modified reward:
    r^λ(x, a) = max(x, 0) - λ·a
  using finite-horizon backward induction (T steps).

  Transition: x' ~ N(alpha_m·x + beta_m·a - drift, sigma_w²)
  Approximated by a fixed quadrature grid for the noise term.

Indexability check:
  Passive set: U(λ) = {x : Q^λ(x,0) ≥ Q^λ(x,1)}
  Indexability: U(λ) non-decreasing in λ (passive set grows as subsidy rises)

  For each type we plot:
    - Q^λ(x,1) - Q^λ(x,0)  vs  x  for several λ  (threshold shifts left as λ↑)
    - Whittle index λ*(x) = inf{λ : Q^λ(x,0) ≥ Q^λ(x,1)}  vs  x
    - Fraction of states passive |U(λ)|/n_grid vs λ  (should be monotone)

Usage:
  python check_indexability.py
  python check_indexability.py --gamma 0.99 --T 50 --n_grid 100
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from math import erf as _erf

from env import TYPE_ALPHA_MEAN, TYPE_BETA_MEAN, TYPE_NAMES, AdaptRMABConfig


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF via math.erf — no scipy needed."""
    sqrt2 = 1.4142135623730951
    return np.vectorize(lambda v: 0.5 * (1.0 + _erf(float(v) / sqrt2)))(x)


# ---------------------------------------------------------------------------
# Backward induction with Gaussian transition
# ---------------------------------------------------------------------------

def _make_transition_matrix(
    x_grid: np.ndarray,
    alpha: float,
    beta: float,
    drift: float,
    sigma_w: float,
    a: int,
) -> np.ndarray:
    """
    P[i,j] = P(x' ∈ bin j | x = x_grid[i], action a)
    Bins are Voronoi cells: midpoints between adjacent grid points.
    """
    n   = len(x_grid)
    h   = x_grid[1] - x_grid[0]  # assumes uniform grid
    mu  = alpha * x_grid + beta * a - drift  # shape (n,)

    # Bin edges
    edges = np.empty(n + 1)
    edges[1:-1] = 0.5 * (x_grid[:-1] + x_grid[1:])
    edges[0]    = x_grid[0]  - h / 2
    edges[-1]   = x_grid[-1] + h / 2

    # CDF at each edge for each starting state: shape (n, n+1)
    z_lo = (edges[np.newaxis, :-1] - mu[:, np.newaxis]) / sigma_w
    z_hi = (edges[np.newaxis,  1:] - mu[:, np.newaxis]) / sigma_w
    P    = _norm_cdf(z_hi) - _norm_cdf(z_lo)
    P   /= P.sum(axis=1, keepdims=True)  # renormalize (absorb tail mass)
    return P.astype(np.float32)


def backward_induction(
    x_grid: np.ndarray,
    alpha: float,
    beta: float,
    drift: float,
    sigma_w: float,
    lam: float,
    T: int,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Finite-horizon backward induction for modified single-arm MDP.
    Returns Q0, Q1 at t=0 (shape: (n_grid,) each).
    """
    n  = len(x_grid)
    r0 = np.maximum(x_grid, 0.0)          # reward with a=0
    r1 = np.maximum(x_grid, 0.0) - lam    # reward with a=1

    P0 = _make_transition_matrix(x_grid, alpha, beta, drift, sigma_w, a=0)
    P1 = _make_transition_matrix(x_grid, alpha, beta, drift, sigma_w, a=1)

    V = np.zeros(n, dtype=np.float32)

    for _ in range(T):
        Q0 = r0 + gamma * (P0 @ V)
        Q1 = r1 + gamma * (P1 @ V)
        V  = np.maximum(Q0, Q1)

    return Q0, Q1


# ---------------------------------------------------------------------------
# Indexability check per type
# ---------------------------------------------------------------------------

def check_type(
    type_idx: int,
    x_grid: np.ndarray,
    lambdas: np.ndarray,
    cfg: AdaptRMABConfig,
    T: int,
    gamma: float,
) -> dict:
    alpha = TYPE_ALPHA_MEAN[type_idx]
    beta  = TYPE_BETA_MEAN[type_idx]
    name  = TYPE_NAMES[type_idx]

    passive_fracs = np.zeros(len(lambdas))
    # advantage[l, i] = Q^λ(x_i, 1) - Q^λ(x_i, 0)  (positive = prefer active)
    advantage_mat = np.zeros((len(lambdas), len(x_grid)), dtype=np.float32)
    whittle_idx   = np.full(len(x_grid), np.nan)

    for li, lam in enumerate(lambdas):
        Q0, Q1 = backward_induction(
            x_grid, alpha, beta, cfg.drift, cfg.sigma_w, lam, T, gamma
        )
        adv = Q1 - Q0
        advantage_mat[li] = adv
        passive_fracs[li] = float((adv <= 0).mean())

    # Whittle index λ*(x) = smallest λ where Q^λ(x,0) ≥ Q^λ(x,1)
    for xi in range(len(x_grid)):
        crossing = np.where(advantage_mat[:, xi] <= 0)[0]
        if len(crossing) > 0:
            whittle_idx[xi] = lambdas[crossing[0]]

    monotone = bool(np.all(np.diff(passive_fracs) >= -1e-4))

    return {
        "name":          name,
        "alpha":         alpha,
        "beta":          beta,
        "passive_fracs": passive_fracs,
        "advantage_mat": advantage_mat,
        "whittle_idx":   whittle_idx,
        "monotone":      monotone,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_TYPE_COLORS = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
_LAMBDA_SHOW = [0.0, 0.5, 1.0, 2.0, 3.0, 4.5]


def plot_indexability(
    results: list[dict],
    x_grid: np.ndarray,
    lambdas: np.ndarray,
    save_path: str,
):
    M    = len(results)
    fig, axes = plt.subplots(3, M, figsize=(4 * M, 11))

    for m, res in enumerate(results):
        name   = res["name"]
        color  = _TYPE_COLORS[m]
        pfrac  = res["passive_fracs"]
        adv    = res["advantage_mat"]   # (n_lam, n_grid)
        wi     = res["whittle_idx"]
        mono   = res["monotone"]

        # Row 0: advantage curves for selected lambdas
        ax = axes[0, m]
        show_lams = [lam for lam in _LAMBDA_SHOW if lam <= lambdas.max()]
        cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(show_lams)))
        for li_show, lam in enumerate(show_lams):
            li   = int(np.argmin(np.abs(lambdas - lam)))
            ax.plot(x_grid, adv[li], color=cmap[li_show],
                    lw=1.4, label=f"λ={lam:.1f}")
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel("State x")
        ax.set_ylabel("Q(x,1) − Q(x,0)")
        ax.set_title(f"Type {m}: {name}\n(α={res['alpha']:.2f}, β={res['beta']:.2f})")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        # Row 1: Whittle index vs state
        ax = axes[1, m]
        ax.plot(x_grid, wi, color=color, lw=1.8)
        ax.set_xlabel("State x")
        ax.set_ylabel("Whittle index λ*(x)")
        ax.set_title(f"Whittle Index  (type {m})")
        ax.grid(True, alpha=0.3)
        valid = ~np.isnan(wi)
        if valid.any():
            ax.set_ylim(0, lambdas.max() * 1.05)

        # Row 2: passive fraction vs lambda (monotonicity check)
        ax = axes[2, m]
        ax.plot(lambdas, pfrac, color=color, lw=2.0,
                marker="o", markersize=3)
        ax.set_xlabel("Subsidy λ")
        ax.set_ylabel("|U(λ)| / n_grid")
        status = "MONOTONE ✓" if mono else "NON-MONOTONE ✗"
        ax.set_title(f"Passive Fraction  [{status}]")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Whittle Indexability Check (per arm type)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma",   type=float, default=0.99)
    parser.add_argument("--T",       type=int,   default=100,
                        help="horizon for backward induction")
    parser.add_argument("--n_grid",  type=int,   default=100,
                        help="number of state grid points")
    parser.add_argument("--x_lo",   type=float, default=-8.0)
    parser.add_argument("--x_hi",   type=float, default=20.0)
    parser.add_argument("--lam_lo", type=float, default=0.0)
    parser.add_argument("--lam_hi", type=float, default=20.0)
    parser.add_argument("--n_lam",  type=int,   default=50)
    parser.add_argument("--out",    type=str,   default="indexability.png")
    args = parser.parse_args()

    cfg     = AdaptRMABConfig()
    x_grid  = np.linspace(args.x_lo, args.x_hi, args.n_grid, dtype=np.float32)
    lambdas = np.linspace(args.lam_lo, args.lam_hi, args.n_lam)

    print("=" * 60)
    print(f"Indexability check  T={args.T}  γ={args.gamma}")
    print(f"  x ∈ [{args.x_lo}, {args.x_hi}]  n_grid={args.n_grid}")
    print(f"  λ ∈ [{args.lam_lo:.1f}, {args.lam_hi:.1f}]  n_lam={args.n_lam}")
    print("=" * 60)

    results = []
    for m in range(cfg.M):
        res = check_type(m, x_grid, lambdas, cfg, args.T, args.gamma)
        results.append(res)
        tag = "MONOTONE" if res["monotone"] else "NON-MONOTONE"
        wi_valid = res["whittle_idx"][~np.isnan(res["whittle_idx"])]
        wi_str   = (f"λ* ∈ [{wi_valid.min():.2f}, {wi_valid.max():.2f}]"
                    if len(wi_valid) else "λ* not found in range")
        print(f"  type {m} ({res['name']:12s})  [{tag}]  {wi_str}")

    print()
    all_mono = all(r["monotone"] for r in results)
    if all_mono:
        print("All 4 types PASS monotonicity check → indexability supported.")
    else:
        print("WARNING: some types fail monotonicity → may not be Whittle-indexable.")
    print()

    plot_indexability(results, x_grid, lambdas, save_path=args.out)

    # Save Whittle index values for reference
    out_npz = Path(args.out).with_suffix(".npz")
    np.savez(out_npz,
             x_grid=x_grid,
             lambdas=lambdas,
             **{f"whittle_idx_{m}": results[m]["whittle_idx"] for m in range(cfg.M)},
             **{f"passive_fracs_{m}": results[m]["passive_fracs"] for m in range(cfg.M)})
    print(f"[data] saved {out_npz}")


if __name__ == "__main__":
    main()
