"""Plot the eight main policies at 25% budgets in the paper's style."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs/figures"
ENVS = [("synthetic-stationary", "Stationary", "synthetic_stationary"),
        ("synthetic-drifting", "Drifting", "synthetic_drifting"),
        ("mimic-icu", "MIMIC-ICU", "mimic_icu")]
METHODS = ["random", "obs_greedy", "activation_greedy", "neurwin",
           "ppo", "learned_rollout", "dpmd", "oracle_lookahead"]
LABELS = ["Random", "Observation-greedy", "Activation-greedy", "NeurWIN",
          "PPO", "Learned rollout", "BIRD", "Oracle lookahead"]


def plot_size(n, k):
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.2), layout="constrained")
    for i, (ax, (folder, title, stem)) in enumerate(zip(axes, ENVS)):
        if n == 20:
            source = ROOT / f"experiment_outputs/main_comparison/{stem}_comparison.npz"
        else:
            group = "all_policies_transfer"
            source = ROOT / folder / f"experiment_outputs/{group}/comparison_N{n}_K{k}.npz"
        with np.load(source) as data:
            assert int(data["N"]) == n and int(data["K"]) == k
            samples = [data[m] for m in METHODS]
            assert all(a.size == 100 and np.isfinite(a).all() for a in samples)
            means = [a.mean() for a in samples]
            stds = [a.std() for a in samples]
        colors = ["#167d9a" if m == "dpmd" else "#aaaaaa" if m == "oracle_lookahead" else "#d4dfe7" for m in METHODS]
        ax.barh(range(len(METHODS)), means, xerr=stds, color=colors,
                error_kw={"elinewidth": .65, "capsize": 1.5})
        ax.set_yticks(range(len(METHODS)), LABELS if i == 0 else [""] * len(METHODS), fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Episode return")
        ax.set_xlim(left=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    if n != 20:
        fig.suptitle(f"N = {n}, K = {k}", fontsize=11)
    if n != 20:
        fig.supxlabel("Learned policies transfer from N=20, K=5 without retraining.", fontsize=7)
    for extension in ("png", "pdf"):
        path = OUT / f"main_comparison_N{n}_K{k}.{extension}"
        fig.savefig(path, dpi=300)
        print(path)
    if n == 20:
        fig.savefig(OUT / "expanded_main.pdf")
    plt.close(fig)


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    OUT.mkdir(parents=True, exist_ok=True)
    for n, k in [(20, 5), (40, 10), (100, 25)]:
        plot_size(n, k)
