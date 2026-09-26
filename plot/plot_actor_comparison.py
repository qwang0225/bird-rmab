"""Regenerate the four-actor N=20 comparison from saved episode returns."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input-dir", type=Path, default=REPO, help="Root directory of experiment results.")
parser.add_argument("--output-dir", type=Path, default=REPO / "outputs" / "figures",
                    help="Directory for generated paper figures and tables.")
args = parser.parse_args()
ROOT = args.input_dir.resolve()
OUT = args.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)

ENVS = [
    ("Stationary", "synthetic-stationary/experiment_outputs/actor_ablation/actor_ablation_N20_K5.npz", "synthetic_stationary"),
    ("Drifting", "synthetic-drifting/experiment_outputs/actor_ablation/actor_ablation_N20_K5.npz", "synthetic_drifting"),
    ("MIMIC-ICU", "mimic-icu/experiment_outputs/actor_ablation/actor_ablation_N20_K5.npz", "mimic_icu"),
]
METHODS = [("MLP Actor", "mlp_actor"), ("Joint-$N$", "joint_N"),
           ("Gaussian actor", "gaussian_actor"), ("BIRD", "dpmd")]
COLORS = ["#d4dfe7", "#c3d1dc", "#aebfce", "#167d9a"]


def main():
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7), layout="constrained")
    rows = []
    for j, (ax, (title, ablation_path, main_stem)) in enumerate(zip(axes, ENVS)):
        with np.load(ROOT / ablation_path) as data:
            ablation = {name: data[name].copy() for name in data.files}
        returns = [ablation[key] for _, key in METHODS]
        assert all(a.size == 100 and np.isfinite(a).all() for a in returns)
        means = [a.mean() for a in returns]
        stds = [a.std() for a in returns]
        rows.append([title] + [f"${m:.1f} \\pm {s:.1f}$" for m, s in zip(means, stds)])
        ax.barh(range(4), means, xerr=stds, color=COLORS,
                error_kw={"elinewidth": .65, "capsize": 1.5})
        ax.set_yticks(range(4), [m[0] for m in METHODS] if j == 0 else [""] * 4)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Episode return")
        ax.set_xlim(left=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"actor_ablation_combined_N20_K5.{extension}", dpi=300)
    plt.close(fig)
    tex = [r"\begin{table}[H]", r"\centering", r"\small",
           r"\caption{Actor comparison at $N=20,K=5$: mean return $\pm$ episode standard deviation over 100 episodes. All four actors use the actor-ablation evaluation. Bold indicates the highest mean return in each environment.}",
           r"\label{tab:actor_comparison}", r"\begin{tabular}{lcccc}", r"\toprule",
           r"Environment & MLP Actor & Joint-$N$ & Gaussian actor & BIRD \\", r"\midrule"]
    for row in rows:
        row[-1] = r"$\boldsymbol{" + row[-1].strip("$") + "}$"
        tex.append(" & ".join(row) + r" \\")
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (OUT / "actor_comparison_table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
