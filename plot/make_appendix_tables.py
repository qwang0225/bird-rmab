"""Generate appendix LaTeX tables from saved combined result files."""
from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "result_tables.tex"


def _escape(text: str) -> str:
    return text.replace("_", r"\_")


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.1f} $\\pm$ {std:.1f}"


def _fmt_method(method: str, mean: float, std: float) -> str:
    value = _fmt(mean, std)
    if method == "BIRD":
        return r"\textbf{" + value + "}"
    return value


def _load(path: str):
    data = np.load(ROOT / path, allow_pickle=True)
    return (
        [str(x) for x in data["environment_labels"]],
        [str(x) for x in data["method_labels"]],
        np.asarray(data["mean_returns"], dtype=float),
        np.asarray(data["std_returns"], dtype=float),
    )


def _table(title: str, label: str, npz_path: str) -> str:
    envs, methods, means, stds = _load(npz_path)
    cols = "l" + "c" * len(methods)
    lines = [
        r"\begin{table}[h]",
        r"  \centering",
        f"  \\caption{{{title}}}",
        f"  \\label{{{label}}}",
        r"  \small",
        f"  \\begin{{tabular}}{{{cols}}}",
        r"    \toprule",
        "    Environment & " + " & ".join(_escape(m) for m in methods) + r" \\",
        r"    \midrule",
    ]
    for i, env in enumerate(envs):
        vals = [_fmt_method(methods[j], means[i, j], stds[i, j]) for j in range(len(methods))]
        lines.append("    " + _escape(env) + " & " + " & ".join(vals) + r" \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def _combined_main_table() -> str:
    specs = [
        (20, 5, "main_comparison_N20_K5.npz"),
        (40, 8, "main_comparison_N40_K8.npz"),
        (100, 20, "main_comparison_N100_K20.npz"),
    ]
    loaded = [(n, k, *_load(path)) for n, k, path in specs]
    methods = loaded[0][3]
    if any(item[3] != methods for item in loaded):
        raise RuntimeError("Main comparison method labels differ across sizes.")

    cols = "ccl" + "c" * len(methods)
    lines = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \caption{Main comparison across problem sizes. Values are mean episode return \(\pm\) standard deviation over 100 evaluation episodes with different seeds.}",
        r"  \label{tab:main-comparison-all-sizes}",
        r"  \scriptsize",
        f"  \\begin{{tabular}}{{{cols}}}",
        r"    \toprule",
        r"    \(N\) & \(K\) & Environment & " + " & ".join(_escape(m) for m in methods) + r" \\",
        r"    \midrule",
    ]
    for block_idx, (n, k, envs, _, means, stds) in enumerate(loaded):
        if block_idx:
            lines.append(r"    \midrule")
        for i, env in enumerate(envs):
            vals = [_fmt_method(methods[j], means[i, j], stds[i, j]) for j in range(len(methods))]
            n_cell = str(n) if i == 0 else ""
            k_cell = str(k) if i == 0 else ""
            lines.append(
                "    "
                + n_cell
                + " & "
                + k_cell
                + " & "
                + _escape(env)
                + " & "
                + " & ".join(vals)
                + r" \\"
            )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    tables = [
        _combined_main_table(),
        _table(
            r"Actor ablation at \(N=20,K=5\). Values are mean episode return \(\pm\) standard deviation over 100 evaluation episodes with different seeds.",
            "tab:actor-ablation-n20",
            "actor_ablation_combined_N20_K5.npz",
        ),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(tables), encoding="utf-8")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
