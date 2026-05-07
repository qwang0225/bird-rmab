"""
Collect and plot N=20, K=5 main comparison results across environments.

Inputs:
  synthetic-stationary/comparison_N20_K5.npz
  synthetic-drifting/comparison_N20_K5.npz
  mimic-icu/comparison_N20_K5.npz

Outputs:
  main_comparison_N20_K5.png
  main_comparison_N20_K5.npz
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENTS = [
    ("synthetic-stationary", "Stationary"),
    ("synthetic-drifting", "Drifting"),
    ("mimic-icu", "MIMIC-ICU Simulator"),
]
METHOD_ORDER = ["random", "greedy", "neurwin", "ppo", "dpmd", "oracle_lookahead"]
LABELS = {
    "random": "Random",
    "greedy": "Greedy",
    "neurwin": "NeurWIN",
    "ppo": "PPO",
    "dpmd": "BIRD",
    "oracle_lookahead": "Oracle",
}
COLORS = {
    "random": "#d62728",
    "greedy": "#ff9896",
    "neurwin": "#ff7f0e",
    "ppo": "#8c564b",
    "dpmd": "#2ca02c",
    "oracle_lookahead": "#1f77b4",
}


def _load_result(env_dir: str) -> dict[str, np.ndarray]:
    path = ROOT / env_dir / "comparison_N20_K5.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    data = np.load(path, allow_pickle=True)
    names = [str(x) for x in data["method_names"]]
    return {name: np.asarray(data[name], dtype=np.float32) for name in names}


def main() -> None:
    env_results = [(env_dir, title, _load_result(env_dir)) for env_dir, title in ENVIRONMENTS]
    methods = [m for m in METHOD_ORDER if all(m in res for _, _, res in env_results)]
    if not methods:
        raise RuntimeError("No common methods found across comparison files.")

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.2), sharey=False)
    if len(env_results) == 1:
        axes = [axes]

    mean_returns = np.zeros((len(env_results), len(methods)), dtype=np.float32)
    std_returns = np.zeros_like(mean_returns)

    for env_idx, (_, title, results) in enumerate(env_results):
        ax = axes[env_idx]
        means = np.asarray([results[m].mean() for m in methods], dtype=np.float32)
        stds = np.asarray([results[m].std() for m in methods], dtype=np.float32)
        mean_returns[env_idx] = means
        std_returns[env_idx] = stds

        x = np.arange(len(methods))
        bars = ax.bar(
            x, means, yerr=stds, capsize=4,
            color=[COLORS.get(m, "#8c8c8c") for m in methods],
            edgecolor="black", linewidth=0.7,
            error_kw={"ecolor": "black", "elinewidth": 0.9, "capthick": 0.9},
        )
        if "dpmd" in methods:
            bird_bar = bars[methods.index("dpmd")]
            bird_bar.set_edgecolor("red")
            bird_bar.set_linewidth(2.4)

        ax.set_title(title)
        ax.set_xticks([])
        ax.grid(True, axis="y", alpha=0.28)
        if env_idx == 0:
            ax.set_ylabel("Episode return")

        y_top = float(np.max(means + stds))
        y_bottom = min(0.0, float(np.min(means - stds)))
        ax.set_ylim(y_bottom, y_top + 0.08 * max(1.0, y_top - y_bottom))

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[m], edgecolor="red" if m == "dpmd" else "black",
                      linewidth=2.0 if m == "dpmd" else 0.7)
        for m in methods
    ]
    fig.legend(
        handles, [LABELS.get(m, m) for m in methods],
        loc="center left", ncol=1, frameon=False, bbox_to_anchor=(0.86, 0.5),
    )
    fig.tight_layout(rect=(0, 0, 0.84, 1))

    out_png = ROOT / "main_comparison_N20_K5.png"
    out_npz = ROOT / "main_comparison_N20_K5.npz"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    np.savez(
        out_npz,
        environment_dirs=np.asarray([e[0] for e in ENVIRONMENTS]),
        environment_labels=np.asarray([e[1] for e in ENVIRONMENTS]),
        method_names=np.asarray(methods),
        method_labels=np.asarray([LABELS.get(m, m) for m in methods]),
        mean_returns=mean_returns,
        std_returns=std_returns,
    )
    print(f"[saved] {out_png}")
    print(f"[saved] {out_npz}")


if __name__ == "__main__":
    main()
