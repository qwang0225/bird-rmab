"""
Particle-filter posterior priority diagnostic.

This script visualizes why a stochastic priority-score actor is useful under
partial observability. For one fixed observed arm history, it runs a bootstrap
particle filter using the known simulator dynamics, obtains particles from the
latent posterior, and computes the privileged oracle marginal activation value
for each particle.

The resulting histogram estimates p(oracle priority | observed history). This
diagnostic is only for analysis and visualization; BIRD training does not use
particles, hidden states, or simulator parameters.
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
class QueryHistory:
    obs: np.ndarray
    actions: np.ndarray
    true_priority: float
    true_type: int


@dataclass
class PosteriorResult:
    key: str
    label: str
    priorities: np.ndarray
    types: np.ndarray
    true_priority: float
    true_type: int
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


def _normalize_log_weights(log_w: np.ndarray) -> np.ndarray:
    log_w = log_w - np.max(log_w)
    w = np.exp(log_w)
    total = float(w.sum())
    if total <= 0.0 or not np.isfinite(total):
        return np.full_like(w, 1.0 / len(w), dtype=np.float64)
    return w / total


def _resample(rng: np.random.Generator, weights: np.ndarray) -> np.ndarray:
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cdf = np.cumsum(weights)
    return np.searchsorted(cdf, positions, side="right")


def _collect_adapt_queries(args: argparse.Namespace) -> list[QueryHistory]:
    env_mod, base_mod = _load_modules("synthetic-drifting")
    cfg = env_mod.AdaptRMABConfig(N=args.n_arms, K=args.budget, T=args.horizon)
    oracle = base_mod.OracleLookaheadPolicy(cfg, H=args.oracle_horizon, gamma=args.gamma)
    rng = np.random.default_rng(args.seed + 11)
    queries: list[QueryHistory] = []

    for ep in range(args.query_episodes):
        env = env_mod.AdaptRMABEnv(cfg, seed=args.seed + ep)
        obs, info = env.reset()
        obs_traj = [obs.copy()]
        action_traj = []
        info_traj = [info]

        for _ in range(cfg.T):
            action = _random_action(rng, cfg.N, cfg.K)
            obs, _, done, info = env.step(action)
            action_traj.append(action.copy())
            obs_traj.append(obs.copy())
            info_traj.append(info)
            if done:
                break

        for _ in range(max(1, args.candidates // max(args.query_episodes, 1))):
            t = int(rng.integers(args.history_len, len(obs_traj)))
            arm = int(rng.integers(0, cfg.N))
            theta = int(info_traj[t]["theta_true"][arm])
            true_priority = oracle._marginal(
                float(info_traj[t]["x_true"][arm]),
                float(info_traj[t]["alpha_true"][arm]),
                float(info_traj[t]["beta_true"][arm]),
                env_mod.TYPE_ALPHA_MEAN[theta],
                env_mod.TYPE_BETA_MEAN[theta],
            )
            queries.append(
                QueryHistory(
                    obs=np.asarray([obs_traj[j][arm] for j in range(t - args.history_len, t + 1)], dtype=np.float32),
                    actions=np.asarray([action_traj[j][arm] for j in range(t - args.history_len, t)], dtype=np.float32),
                    true_priority=float(true_priority),
                    true_type=theta,
                )
            )

    return queries


def _collect_mimic_queries(args: argparse.Namespace) -> list[QueryHistory]:
    env_mod, base_mod = _load_modules("mimic-icu")
    cfg = env_mod.MIMICRMABConfig(N=args.n_arms, K=args.budget, T=args.horizon)
    oracle = base_mod.OracleLookahead(cfg, H=args.oracle_horizon, gamma=args.gamma)
    rng = np.random.default_rng(args.seed + 22)
    queries: list[QueryHistory] = []

    for ep in range(args.query_episodes):
        env = env_mod.MIMICRMABEnv(cfg, seed=args.seed + 10_000 + ep)
        obs, info = env.reset()
        obs_traj = [obs.copy()]
        action_traj = []
        info_traj = [info]

        for _ in range(cfg.T):
            action = _random_action(rng, cfg.N, cfg.K)
            obs, _, done, info = env.step(action)
            action_traj.append(action.copy())
            obs_traj.append(obs.copy())
            info_traj.append(info)
            if done:
                break

        for _ in range(max(1, args.candidates // max(args.query_episodes, 1))):
            t = int(rng.integers(args.history_len, len(obs_traj)))
            arm = int(rng.integers(0, cfg.N))
            theta = int(info_traj[t]["theta_true"][arm])
            true_priority = oracle._marginal(
                info_traj[t]["x_true"][arm],
                info_traj[t]["alpha_true"][arm],
                info_traj[t]["beta_true"][arm],
                theta,
            )
            queries.append(
                QueryHistory(
                    obs=np.asarray([obs_traj[j][arm] for j in range(t - args.history_len, t + 1)], dtype=np.float32),
                    actions=np.asarray([action_traj[j][arm] for j in range(t - args.history_len, t)], dtype=np.float32),
                    true_priority=float(true_priority),
                    true_type=theta,
                )
            )

    return queries


def _adapt_particle_priorities(query: QueryHistory, n_particles: int, seed: int, args: argparse.Namespace):
    env_mod, base_mod = _load_modules("synthetic-drifting")
    cfg = env_mod.AdaptRMABConfig(N=args.n_arms, K=args.budget, T=args.horizon)
    oracle = base_mod.OracleLookaheadPolicy(cfg, H=args.oracle_horizon, gamma=args.gamma)
    rng = np.random.default_rng(seed)

    theta = rng.integers(0, cfg.M, size=n_particles)
    alpha = np.asarray([env_mod.TYPE_ALPHA_MEAN[int(m)] for m in theta], dtype=np.float64)
    beta = np.asarray([env_mod.TYPE_BETA_MEAN[int(m)] for m in theta], dtype=np.float64)
    x = cfg.init_mean + cfg.init_std * rng.standard_normal(n_particles)

    log_w = -0.5 * ((float(query.obs[0]) - x) / cfg.sigma_v) ** 2
    weights = _normalize_log_weights(log_w)
    idx = _resample(rng, weights)
    theta, alpha, beta, x = theta[idx], alpha[idx], beta[idx], x[idx]

    for k, action in enumerate(query.actions):
        process_noise = cfg.sigma_w * rng.standard_normal(n_particles)
        x = alpha * x + beta * float(action) - cfg.drift + process_noise

        alpha_bar = np.asarray([env_mod.TYPE_ALPHA_MEAN[int(m)] for m in theta], dtype=np.float64)
        beta_bar = np.asarray([env_mod.TYPE_BETA_MEAN[int(m)] for m in theta], dtype=np.float64)
        alpha = np.clip(
            alpha_bar + cfg.ou_rho * (alpha - alpha_bar) + cfg.sigma_alpha * rng.standard_normal(n_particles),
            cfg.alpha_lo,
            cfg.alpha_hi,
        )
        beta = np.clip(
            beta_bar + cfg.ou_rho * (beta - beta_bar) + cfg.sigma_beta * rng.standard_normal(n_particles),
            cfg.beta_lo,
            cfg.beta_hi,
        )

        y = float(query.obs[k + 1])
        log_w = -0.5 * ((y - x) / cfg.sigma_v) ** 2
        weights = _normalize_log_weights(log_w)
        if 1.0 / np.sum(weights * weights) < 0.5 * n_particles:
            idx = _resample(rng, weights)
            theta, alpha, beta, x = theta[idx], alpha[idx], beta[idx], x[idx]

    priorities = np.asarray(
        [
            oracle._marginal(
                float(x[i]),
                float(alpha[i]),
                float(beta[i]),
                env_mod.TYPE_ALPHA_MEAN[int(theta[i])],
                env_mod.TYPE_BETA_MEAN[int(theta[i])],
            )
            for i in range(n_particles)
        ],
        dtype=np.float32,
    )
    return priorities, theta.astype(np.int32), list(env_mod.TYPE_NAMES)


def _mimic_particle_priorities(query: QueryHistory, n_particles: int, seed: int, args: argparse.Namespace):
    env_mod, base_mod = _load_modules("mimic-icu")
    cfg = env_mod.MIMICRMABConfig(N=args.n_arms, K=args.budget, T=args.horizon)
    oracle = base_mod.OracleLookahead(cfg, H=args.oracle_horizon, gamma=args.gamma)
    rng = np.random.default_rng(seed)

    theta = rng.integers(0, cfg.M, size=n_particles)
    alpha = np.asarray([[env_mod.TYPE_PARAMS[int(m)][0], env_mod.TYPE_PARAMS[int(m)][1]] for m in theta], dtype=np.float64)
    beta = np.asarray([[env_mod.TYPE_PARAMS[int(m)][2], env_mod.TYPE_PARAMS[int(m)][3]] for m in theta], dtype=np.float64)
    x = cfg.init_mean + cfg.init_std * rng.standard_normal((n_particles, 2))
    obs_sigma = env_mod.OBS_SIGMA.astype(np.float64)
    loading = env_mod.LOADING_MATRIX.astype(np.float64)

    pred = x @ loading.T
    log_w = -0.5 * np.sum(((query.obs[0] - pred) / obs_sigma) ** 2, axis=1)
    weights = _normalize_log_weights(log_w)
    idx = _resample(rng, weights)
    theta, alpha, beta, x = theta[idx], alpha[idx], beta[idx], x[idx]

    for k, action in enumerate(query.actions):
        drift = np.asarray([[env_mod.TYPE_PARAMS[int(m)][4], env_mod.TYPE_PARAMS[int(m)][5]] for m in theta], dtype=np.float64)
        sigma_w = np.asarray([[env_mod.TYPE_PARAMS[int(m)][6], env_mod.TYPE_PARAMS[int(m)][7]] for m in theta], dtype=np.float64)
        coupling = np.column_stack(
            [
                env_mod.COUPLING_C_RH * x[:, 1],
                env_mod.COUPLING_C_HR * x[:, 0],
            ]
        )
        x = alpha * x + coupling + beta * float(action) + drift + sigma_w * rng.standard_normal((n_particles, 2))

        alpha_bar = np.asarray([[env_mod.TYPE_PARAMS[int(m)][0], env_mod.TYPE_PARAMS[int(m)][1]] for m in theta], dtype=np.float64)
        beta_bar = np.asarray([[env_mod.TYPE_PARAMS[int(m)][2], env_mod.TYPE_PARAMS[int(m)][3]] for m in theta], dtype=np.float64)
        alpha = np.clip(
            alpha_bar + cfg.ou_rho * (alpha - alpha_bar) + cfg.sigma_alpha * rng.standard_normal((n_particles, 2)),
            cfg.alpha_lo,
            cfg.alpha_hi,
        )
        beta = np.clip(
            beta_bar + cfg.ou_rho * (beta - beta_bar) + cfg.sigma_beta * rng.standard_normal((n_particles, 2)),
            cfg.beta_lo,
            cfg.beta_hi,
        )

        pred = x @ loading.T
        log_w = -0.5 * np.sum(((query.obs[k + 1] - pred) / obs_sigma) ** 2, axis=1)
        weights = _normalize_log_weights(log_w)
        if 1.0 / np.sum(weights * weights) < 0.5 * n_particles:
            idx = _resample(rng, weights)
            theta, alpha, beta, x = theta[idx], alpha[idx], beta[idx], x[idx]

    priorities = np.asarray(
        [
            oracle._marginal(
                x[i].astype(np.float32),
                alpha[i].astype(np.float32),
                beta[i].astype(np.float32),
                int(theta[i]),
            )
            for i in range(n_particles)
        ],
        dtype=np.float32,
    )
    return priorities, theta.astype(np.int32), list(env_mod.TYPE_NAMES)


def _select_query(queries: list[QueryHistory], particle_fn, args: argparse.Namespace, seed_offset: int) -> tuple[QueryHistory, np.ndarray, np.ndarray, list[str]]:
    best_score = -np.inf
    best: tuple[QueryHistory, np.ndarray, np.ndarray, list[str]] | None = None
    for j, query in enumerate(queries[: args.candidates]):
        priorities, types, type_names = particle_fn(query, args.search_particles, args.seed + seed_offset + j, args)
        spread = float(np.percentile(priorities, 90) - np.percentile(priorities, 10))
        counts = np.bincount(types, minlength=max(int(types.max()) + 1, 1)).astype(np.float64)
        probs = counts[counts > 0] / max(float(counts.sum()), 1.0)
        entropy = float(-(probs * np.log(probs + 1e-12)).sum())
        diversity = float(len(probs))
        score = spread + 0.75 * entropy + 0.15 * diversity
        if score > best_score:
            best_score = score
            best = (query, priorities, types, type_names)
    assert best is not None
    return best


def _make_result(key: str, label: str, query: QueryHistory, priorities: np.ndarray, types: np.ndarray, type_names: list[str]) -> PosteriorResult:
    return PosteriorResult(
        key=key,
        label=label,
        priorities=priorities,
        types=types,
        true_priority=query.true_priority,
        true_type=query.true_type,
        type_names=type_names,
    )


def _plot(results: list[PosteriorResult], out_png: Path) -> None:
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756"]
    fig, axes = plt.subplots(1, len(results), figsize=(10.0, 3.6), sharey=False)
    if len(results) == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        vals = res.priorities
        bins = np.linspace(float(vals.min()), float(vals.max()), 28)
        if np.allclose(bins[0], bins[-1]):
            bins = 20

        for idx, theta in enumerate(sorted(np.unique(res.types))):
            mask = res.types == theta
            label = res.type_names[int(theta)] if int(theta) < len(res.type_names) else f"type {theta}"
            ax.hist(
                vals[mask],
                bins=bins,
                alpha=0.62,
                color=colors[idx % len(colors)],
                label=label,
                edgecolor="white",
                linewidth=0.4,
            )

        ax.axvline(
            res.true_priority,
            color="black",
            linestyle="--",
            linewidth=1.3,
            label="simulator truth",
        )
        ax.set_title(res.label)
        ax.set_xlabel("Oracle marginal activation value")
        ax.set_ylabel("Particle count")
        ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.8)
        ax.legend(fontsize=7, frameon=False)
        ax.text(
            0.02,
            0.98,
            f"particles={len(vals)}\nstd={vals.std():.2f}\np90-p10={np.percentile(vals, 90) - np.percentile(vals, 10):.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2.5},
        )

    fig.suptitle("One observed history can induce a posterior distribution over oracle priorities", y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_npz(results: list[PosteriorResult], out_npz: Path) -> None:
    payload = {}
    for res in results:
        payload[f"{res.key}_priorities"] = res.priorities
        payload[f"{res.key}_types"] = res.types
        payload[f"{res.key}_true_priority"] = np.asarray(res.true_priority, dtype=np.float32)
        payload[f"{res.key}_true_type"] = np.asarray(res.true_type, dtype=np.int32)
        payload[f"{res.key}_type_names"] = np.asarray(res.type_names)
    np.savez(out_npz, **payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_arms", type=int, default=20)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--history_len", type=int, default=20)
    parser.add_argument("--query_episodes", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=40)
    parser.add_argument("--search_particles", type=int, default=1000)
    parser.add_argument("--particles", type=int, default=8000)
    parser.add_argument("--oracle_horizon", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_png", type=Path, default=ROOT / "particle_priority_posterior.png")
    parser.add_argument("--out_npz", type=Path, default=ROOT / "particle_priority_posterior.npz")
    args = parser.parse_args()

    print("[queries] synthetic-drifting")
    adapt_queries = _collect_adapt_queries(args)
    adapt_query, _, _, _ = _select_query(adapt_queries, _adapt_particle_priorities, args, seed_offset=1000)
    adapt_priorities, adapt_types, adapt_type_names = _adapt_particle_priorities(
        adapt_query,
        args.particles,
        args.seed + 2000,
        args,
    )

    print("[queries] mimic-icu")
    mimic_queries = _collect_mimic_queries(args)
    mimic_query, _, _, _ = _select_query(mimic_queries, _mimic_particle_priorities, args, seed_offset=3000)
    mimic_priorities, mimic_types, mimic_type_names = _mimic_particle_priorities(
        mimic_query,
        args.particles,
        args.seed + 4000,
        args,
    )

    results = [
        _make_result("adapt_lr", "Synthetic drifting RMAB", adapt_query, adapt_priorities, adapt_types, adapt_type_names),
        _make_result("mimic_icu", "MIMIC-ICU Simulator", mimic_query, mimic_priorities, mimic_types, mimic_type_names),
    ]

    _plot(results, args.out_png)
    _save_npz(results, args.out_npz)

    for res in results:
        counts = {
            res.type_names[int(t)] if int(t) < len(res.type_names) else f"type {int(t)}": int((res.types == t).sum())
            for t in sorted(np.unique(res.types))
        }
        print(
            f"[{res.key}] true_priority={res.true_priority:.3f} "
            f"posterior_mean={res.priorities.mean():.3f} posterior_std={res.priorities.std():.3f} "
            f"p10={np.percentile(res.priorities, 10):.3f} p90={np.percentile(res.priorities, 90):.3f} "
            f"type_counts={counts}"
        )

    print(f"[saved] {args.out_png}")
    print(f"[saved] {args.out_npz}")


if __name__ == "__main__":
    main()
