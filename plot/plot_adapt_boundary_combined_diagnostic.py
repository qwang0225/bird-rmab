"""
Combined boundary diagnostic figure for adapt-lr.

Left: posterior rank uncertainty for one boundary arm from the particle-filter
diagnostic. Right: oracle value difference between BIRD sampled Top-K actions
and the deterministic MLP actor on boundary histories.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RANK_NPZ = ROOT / "adapt_lr_particle_rank_posterior.npz"
VALUE_NPZ = ROOT / "adapt_lr_boundary_value_diagnostic.npz"
OUT_PNG = ROOT / "adapt_lr_boundary_combined_diagnostic.png"


def _kde_1d(values: np.ndarray, x_grid: np.ndarray, bandwidth: float) -> np.ndarray:
    kde = np.exp(-0.5 * ((x_grid[:, None] - values[None, :]) / bandwidth) ** 2).mean(axis=1)
    return kde / (bandwidth * np.sqrt(2.0 * np.pi))


def main() -> None:
    rank_data = np.load(RANK_NPZ)
    value_data = np.load(VALUE_NPZ)

    ranks = rank_data["ranks"]
    query_arms = rank_data["query_arms"]
    topk_probs = rank_data["topk_probs"]
    query_arm = int(query_arms[0])
    query_ranks = ranks[:, query_arm]

    diffs = value_data["bird_mean_diff"]
    mean_diff = float(diffs.mean())
    frac_positive = float((diffs > 0.0).mean())

    n_arms = ranks.shape[1]
    budget = 5

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.45, 2.55))

    ax = axes[0]
    rank_bins = np.arange(1, n_arms + 2) - 0.5
    ax.hist(
        query_ranks,
        bins=rank_bins,
        density=True,
        color="#8fb3da",
        edgecolor="white",
        linewidth=0.65,
        alpha=0.58,
    )
    ax.axvspan(0.5, budget + 0.5, color="#d95f5f", alpha=0.075, lw=0)
    ax.axvline(budget + 0.5, color="#2b2b2b", linestyle="--", linewidth=1.05)
    rank_x = np.linspace(1.0, 11.0, 240)
    rank_kde = _kde_1d(query_ranks.astype(np.float32), rank_x, bandwidth=0.55)
    ax.plot(rank_x, rank_kde, color="#204b83", linewidth=2.0)
    ax.set_xlim(0.5, 11.5)
    ax.set_xticks([1, 3, budget, 7, 9, 11])
    ax.set_xlabel(f"Oracle rank of arm {query_arm}")
    ax.set_ylabel("Posterior density")
    ax.set_title("Rank uncertainty")
    ax.grid(False)
    ax.text(
        0.04,
        0.92,
        f"Pr(a=1)={topk_probs[query_arm]:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#d8d8d8", "linewidth": 0.45, "alpha": 0.92, "pad": 2.6},
    )

    ax = axes[1]
    lo = float(np.percentile(diffs, 2))
    hi = float(np.percentile(diffs, 98))
    bins = np.linspace(lo, hi, 22) if not np.isclose(lo, hi) else 18
    ax.hist(
        diffs,
        bins=bins,
        density=True,
        color="#9fd0c8",
        edgecolor="white",
        linewidth=0.65,
        alpha=0.60,
    )
    diff_x = np.linspace(float(diffs.min()), float(diffs.max()), 240)
    bandwidth = max(float(diffs.std()) * 0.35, 1e-3)
    diff_kde = _kde_1d(diffs.astype(np.float32), diff_x, bandwidth=bandwidth)
    ax.plot(diff_x, diff_kde, color="#16766f", linewidth=2.0)
    ax.axvline(0.0, color="#2b2b2b", linestyle="--", linewidth=1.05)
    ax.axvline(mean_diff, color="#d95f5f", linewidth=1.35)
    ax.set_xlabel(r"Oracle value: BIRD $-$ MLP")
    ax.set_ylabel("Density")
    ax.set_title("Value of stochastic samples")
    ax.grid(False)
    ax.text(
        0.04,
        0.92,
        f"mean={mean_diff:.1f}\nPr(>0)={frac_positive:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#d8d8d8", "linewidth": 0.45, "alpha": 0.92, "pad": 2.6},
    )

    fig.subplots_adjust(left=0.09, right=0.985, top=0.84, bottom=0.22, wspace=0.30)
    fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {OUT_PNG}")


if __name__ == "__main__":
    main()
