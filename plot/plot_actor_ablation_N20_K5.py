"""
Collect and plot N=20, K=5 actor ablation results across environments.

Inputs:
  synthetic-stationary/actor_ablation_seed0_N20_K5..npz
  synthetic-drifting/actor_ablation_seed0_N20_K5..npz
  mimic-icu/actor_ablation_seed0_N20_K5.npz

Outputs:
  actor_ablation_combined_N20_K5.png
  actor_ablation_combined_N20_K5.npz
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
METHOD_ORDER = ["mlp_actor", "joint_N", "dpmd"]
LABELS = {
    "mlp_actor": "MLP Actor",
    "joint_N": "Joint-N",
    "dpmd": "BIRD",
}
COLORS = {
    "mlp_actor": "#e377c2",
    "joint_N": "#ff7f0e",
    "dpmd": "#2ca02c",
}
INPUT_CANDIDATES = [
    "actor_ablation_seed0_N20_K5.npz",
    "actor_ablation_seed0_N20_K5..npz",
]
ALIASES = {"dpmd_joint_N": "joint_N"}


def _load_result(env_dir: str) -> tuple[dict[str, np.ndarray], str]:
    for filename in INPUT_CANDIDATES:
        path = ROOT / env_dir / filename
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=True)
        if "method_names" in data:
            names = [str(x) for x in data["method_names"]]
        else:
            names = [str(x) for x in data.files]
        result = {
            ALIASES.get(name, name): np.asarray(data[name], dtype=np.float32)
            for name in names
            if name in data
        }
        if all(name in result for name in METHOD_ORDER):
            return result, filename
    raise FileNotFoundError(f"Missing actor ablation result file in {ROOT / env_dir}")


def main() -> None:
    loaded = [(env_dir, title, *_load_result(env_dir)) for env_dir, title in ENVIRONMENTS]
    env_results = [(env_dir, title, res) for env_dir, title, res, _ in loaded]
    source_files = [filename for _, _, _, filename in loaded]
    available = set().union(*(set(res) for _, _, res in env_results))
    methods = [m for m in METHOD_ORDER if m in available and all(m in res for _, _, res in env_results)]
    if not methods:
        raise RuntimeError("No common methods found across actor ablation files.")

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

    out_png = ROOT / "actor_ablation_combined_N20_K5.png"
    out_npz = ROOT / "actor_ablation_combined_N20_K5.npz"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    np.savez(
        out_npz,
        environment_dirs=np.asarray([e[0] for e in ENVIRONMENTS]),
        environment_labels=np.asarray([e[1] for e in ENVIRONMENTS]),
        source_files=np.asarray(source_files),
        method_names=np.asarray(methods),
        method_labels=np.asarray([LABELS.get(m, m) for m in methods]),
        mean_returns=mean_returns,
        std_returns=std_returns,
    )
    print(f"[saved] {out_png}")
    print(f"[saved] {out_npz}")


if __name__ == "__main__":
    main()
