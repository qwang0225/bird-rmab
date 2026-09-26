"""
Nearest-neighbor oracle-priority diagnostic for partial observability.

For each environment, this script collects simulator trajectories using only
observed arm histories for matching. It then uses privileged simulator state
only after matching to compute oracle marginal activation values for nearby
histories. A wide or multi-peaked neighbor-priority histogram means similar
observed histories can correspond to different hidden simulator states and
different activation priorities.

This is a diagnostic for ambiguity under partial observability. It is not a
claim that the true optimal policy distribution is known or multi-modal.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent


@dataclass
class NeighborResult:
    key: str
    label: str
    query_index: int
    query_priority: float
    query_type: int
    neighbor_indices: np.ndarray
    neighbor_priorities: np.ndarray
    neighbor_types: np.ndarray
    type_names: list[str]


def _clear_env_modules() -> None:
    for name in ["env", "baselines"]:
        if name in sys.modules:
            del sys.modules[name]


def _load_modules(env_dir: str):
    _clear_env_modules()
    path = str(ROOT / env_dir)
    sys.path.insert(0, path)
    try:
        env_mod = importlib.import_module("env")
        base_mod = importlib.import_module("baselines")
    finally:
        sys.path.remove(path)
    return env_mod, base_mod


def _random_action(rng: np.random.Generator, n_arms: int, budget: int) -> np.ndarray:
    action = np.zeros(n_arms, dtype=np.int32)
    action[rng.choice(n_arms, size=budget, replace=False)] = 1
    return action


def _collect_adapt(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    env_mod, base_mod = _load_modules("synthetic-drifting")
    cfg = env_mod.AdaptRMABConfig(N=args.n_arms, K=args.budget, T=args.horizon)
    oracle = base_mod.OracleLookaheadPolicy(cfg, H=args.oracle_horizon, gamma=args.gamma)
    rng = np.random.default_rng(args.seed + 101)

    features: list[np.ndarray] = []
    priorities: list[float] = []
    types: list[int] = []

    for ep in range(args.episodes):
        env = env_mod.AdaptRMABEnv(cfg, seed=args.seed + ep)
        obs, info = env.reset()
        obs_hist = np.zeros((cfg.N, args.history_len), dtype=np.float32)
        act_hist = np.zeros((cfg.N, args.history_len), dtype=np.float32)

        for t in range(cfg.T):
            obs_hist = np.roll(obs_hist, -1, axis=1)
            obs_hist[:, -1] = obs.astype(np.float32)

            if t >= args.history_len:
                for i in range(cfg.N):
                    theta = int(info["theta_true"][i])
                    priority = oracle._marginal(
                        float(info["x_true"][i]),
                        float(info["alpha_true"][i]),
                        float(info["beta_true"][i]),
                        env_mod.TYPE_ALPHA_MEAN[theta],
                        env_mod.TYPE_BETA_MEAN[theta],
                    )
                    features.append(np.concatenate([obs_hist[i], act_hist[i]]))
                    priorities.append(float(priority))
                    types.append(theta)

            action = _random_action(rng, cfg.N, cfg.K)
            obs, _, done, info = env.step(action)
            act_hist = np.roll(act_hist, -1, axis=1)
            act_hist[:, -1] = action.astype(np.float32)
            if done:
                break

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(priorities, dtype=np.float32),
        np.asarray(types, dtype=np.int32),
    )


def _collect_mimic(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    env_mod, base_mod = _load_modules("mimic-icu")
    cfg = env_mod.MIMICRMABConfig(N=args.n_arms, K=args.budget, T=args.horizon)
    oracle = base_mod.OracleLookahead(cfg, H=args.oracle_horizon, gamma=args.gamma)
    rng = np.random.default_rng(args.seed + 202)

    features: list[np.ndarray] = []
    priorities: list[float] = []
    types: list[int] = []

    for ep in range(args.episodes):
        env = env_mod.MIMICRMABEnv(cfg, seed=args.seed + 10_000 + ep)
        obs, info = env.reset()
        obs_hist = np.zeros((cfg.N, args.history_len, cfg.obs_dim), dtype=np.float32)
        act_hist = np.zeros((cfg.N, args.history_len), dtype=np.float32)

        for t in range(cfg.T):
            obs_hist = np.roll(obs_hist, -1, axis=1)
            obs_hist[:, -1, :] = obs.astype(np.float32)

            if t >= args.history_len:
                for i in range(cfg.N):
                    theta = int(info["theta_true"][i])
                    priority = oracle._marginal(
                        info["x_true"][i],
                        info["alpha_true"][i],
                        info["beta_true"][i],
                        theta,
                    )
                    features.append(np.concatenate([obs_hist[i].reshape(-1), act_hist[i]]))
                    priorities.append(float(priority))
                    types.append(theta)

            action = _random_action(rng, cfg.N, cfg.K)
            obs, _, done, info = env.step(action)
            act_hist = np.roll(act_hist, -1, axis=1)
            act_hist[:, -1] = action.astype(np.float32)
            if done:
                break

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(priorities, dtype=np.float32),
        np.asarray(types, dtype=np.int32),
    )


def _local_peak_count(values: np.ndarray, bins: int = 20) -> int:
    counts, _ = np.histogram(values, bins=bins)
    if counts.max() == 0:
        return 0
    threshold = max(2, int(0.08 * counts.max()))
    peaks = 0
    for i in range(1, len(counts) - 1):
        if counts[i] >= threshold and counts[i] > counts[i - 1] and counts[i] >= counts[i + 1]:
            peaks += 1
    return peaks


def _choose_query(
    features: np.ndarray,
    priorities: np.ndarray,
    types: np.ndarray,
    args: argparse.Namespace,
    seed_offset: int,
) -> tuple[int, np.ndarray]:
    rng = np.random.default_rng(args.seed + seed_offset)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-6
    features_norm = (features - mean) / std

    n = len(features)
    candidate_count = min(args.candidates, n)
    candidate_idx = rng.choice(n, size=candidate_count, replace=False)
    k = min(args.neighbors, n - 1)

    best_score = -np.inf
    best_query = int(candidate_idx[0])
    best_neighbors = np.array([], dtype=np.int64)

    for q in candidate_idx:
        diff = features_norm - features_norm[q]
        dist = np.einsum("ij,ij->i", diff, diff)
        nn = np.argpartition(dist, k + 1)[: k + 1]
        nn = nn[nn != q]
        nn = nn[np.argsort(dist[nn])[:k]]

        vals = priorities[nn]
        type_diversity = len(np.unique(types[nn]))
        spread = float(np.percentile(vals, 90) - np.percentile(vals, 10))
        peaks = _local_peak_count(vals)
        score = spread + 0.25 * float(type_diversity - 1) + 0.10 * float(peaks)
        if score > best_score:
            best_score = score
            best_query = int(q)
            best_neighbors = nn.astype(np.int64)

    return best_query, best_neighbors


def _build_result(
    key: str,
    label: str,
    type_names: list[str],
    features: np.ndarray,
    priorities: np.ndarray,
    types: np.ndarray,
    args: argparse.Namespace,
    seed_offset: int,
) -> NeighborResult:
    query, neighbors = _choose_query(features, priorities, types, args, seed_offset)
    return NeighborResult(
        key=key,
        label=label,
        query_index=query,
        query_priority=float(priorities[query]),
        query_type=int(types[query]),
        neighbor_indices=neighbors,
        neighbor_priorities=priorities[neighbors],
        neighbor_types=types[neighbors],
        type_names=type_names,
    )


def _plot_results(results: list[NeighborResult], out_png: Path) -> None:
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756"]
    fig, axes = plt.subplots(1, len(results), figsize=(10.0, 3.6), sharey=False)
    if len(results) == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        all_vals = res.neighbor_priorities
        bins = np.linspace(float(all_vals.min()), float(all_vals.max()), 22)
        if np.allclose(bins[0], bins[-1]):
            bins = 20

        for idx, theta in enumerate(sorted(np.unique(res.neighbor_types))):
            mask = res.neighbor_types == theta
            label = res.type_names[int(theta)] if int(theta) < len(res.type_names) else f"type {theta}"
            ax.hist(
                res.neighbor_priorities[mask],
                bins=bins,
                alpha=0.62,
                color=colors[idx % len(colors)],
                label=label,
                edgecolor="white",
                linewidth=0.4,
            )

        ax.axvline(
            res.query_priority,
            color="black",
            linestyle="--",
            linewidth=1.3,
            label="query",
        )
        ax.set_title(res.label)
        ax.set_xlabel("Oracle marginal activation value")
        ax.set_ylabel("Nearest-neighbor count")
        ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.8)
        ax.legend(fontsize=7, frameon=False)

        std = float(res.neighbor_priorities.std())
        spread = float(np.percentile(res.neighbor_priorities, 90) - np.percentile(res.neighbor_priorities, 10))
        ax.text(
            0.02,
            0.98,
            f"k={len(res.neighbor_indices)}\nstd={std:.2f}\np90-p10={spread:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2.5},
        )

    fig.suptitle("Similar observed histories can imply different privileged oracle priorities", y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _rank_percentiles(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(len(values), dtype=np.float32)
    denom = max(len(values) - 1, 1)
    return ranks / float(denom)


def _plot_rank_results(results: list[NeighborResult], out_png: Path) -> None:
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756"]
    fig, axes = plt.subplots(1, len(results), figsize=(10.0, 3.6), sharey=False)
    if len(results) == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        ranks = _rank_percentiles(res.neighbor_priorities)
        query_rank = float((res.neighbor_priorities < res.query_priority).mean())
        bins = np.linspace(0.0, 1.0, 21)

        for idx, theta in enumerate(sorted(np.unique(res.neighbor_types))):
            mask = res.neighbor_types == theta
            label = res.type_names[int(theta)] if int(theta) < len(res.type_names) else f"type {theta}"
            ax.hist(
                ranks[mask],
                bins=bins,
                alpha=0.62,
                color=colors[idx % len(colors)],
                label=label,
                edgecolor="white",
                linewidth=0.4,
            )

        ax.axvline(
            query_rank,
            color="black",
            linestyle="--",
            linewidth=1.3,
            label="query",
        )
        ax.set_title(res.label)
        ax.set_xlabel("Oracle priority rank percentile")
        ax.set_ylabel("Nearest-neighbor count")
        ax.set_xlim(0.0, 1.0)
        ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.8)
        ax.legend(fontsize=7, frameon=False)

        top_k_fraction = 0.25
        high_priority = float((ranks >= 1.0 - top_k_fraction).mean())
        ax.text(
            0.02,
            0.98,
            f"k={len(res.neighbor_indices)}\nquery pct={query_rank:.2f}\ntop quartile={high_priority:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2.5},
        )

    fig.suptitle("Observed-history neighbors induce a distribution over oracle priority ranks", y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_npz(results: list[NeighborResult], out_npz: Path) -> None:
    payload = {}
    for res in results:
        prefix = res.key
        neighbor_ranks = _rank_percentiles(res.neighbor_priorities)
        payload[f"{prefix}_query_index"] = np.asarray(res.query_index)
        payload[f"{prefix}_query_priority"] = np.asarray(res.query_priority, dtype=np.float32)
        payload[f"{prefix}_query_type"] = np.asarray(res.query_type, dtype=np.int32)
        payload[f"{prefix}_query_rank_percentile"] = np.asarray(
            float((res.neighbor_priorities < res.query_priority).mean()),
            dtype=np.float32,
        )
        payload[f"{prefix}_neighbor_indices"] = res.neighbor_indices
        payload[f"{prefix}_neighbor_priorities"] = res.neighbor_priorities
        payload[f"{prefix}_neighbor_rank_percentiles"] = neighbor_ranks
        payload[f"{prefix}_neighbor_types"] = res.neighbor_types
        payload[f"{prefix}_type_names"] = np.asarray(res.type_names)
    np.savez(out_npz, **payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--n_arms", type=int, default=20)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--history_len", type=int, default=40)
    parser.add_argument("--neighbors", type=int, default=300)
    parser.add_argument("--candidates", type=int, default=250)
    parser.add_argument("--oracle_horizon", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_png", type=Path, default=ROOT / "oracle_priority_neighbors.png")
    parser.add_argument("--out_rank_png", type=Path, default=ROOT / "oracle_rank_neighbors.png")
    parser.add_argument("--out_npz", type=Path, default=ROOT / "oracle_priority_neighbors.npz")
    args = parser.parse_args()

    print("[collect] synthetic-drifting")
    adapt_features, adapt_priorities, adapt_types = _collect_adapt(args)
    print(f"  samples={len(adapt_priorities)} priority_std={adapt_priorities.std():.3f}")

    print("[collect] mimic-icu")
    mimic_features, mimic_priorities, mimic_types = _collect_mimic(args)
    print(f"  samples={len(mimic_priorities)} priority_std={mimic_priorities.std():.3f}")

    adapt_env, _ = _load_modules("synthetic-drifting")
    mimic_env, _ = _load_modules("mimic-icu")
    results = [
        _build_result(
            "adapt_lr",
            "Synthetic drifting RMAB",
            list(adapt_env.TYPE_NAMES),
            adapt_features,
            adapt_priorities,
            adapt_types,
            args,
            seed_offset=303,
        ),
        _build_result(
            "mimic_icu",
            "MIMIC-ICU Simulator",
            list(mimic_env.TYPE_NAMES),
            mimic_features,
            mimic_priorities,
            mimic_types,
            args,
            seed_offset=404,
        ),
    ]

    _plot_results(results, args.out_png)
    _plot_rank_results(results, args.out_rank_png)
    _save_npz(results, args.out_npz)

    for res in results:
        counts = {
            res.type_names[int(t)] if int(t) < len(res.type_names) else f"type {int(t)}": int((res.neighbor_types == t).sum())
            for t in sorted(np.unique(res.neighbor_types))
        }
        vals = res.neighbor_priorities
        print(
            f"[{res.key}] query_priority={res.query_priority:.3f} "
            f"neighbor_mean={vals.mean():.3f} neighbor_std={vals.std():.3f} "
            f"p10={np.percentile(vals, 10):.3f} p90={np.percentile(vals, 90):.3f} "
            f"type_counts={counts}"
        )

    print(f"[saved] {args.out_png}")
    print(f"[saved] {args.out_rank_png}")
    print(f"[saved] {args.out_npz}")


if __name__ == "__main__":
    main()
