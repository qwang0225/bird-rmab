"""
Particle-filter posterior rank diagnostic for adapt-lr.

For one fixed joint observed history in the drifting synthetic RMAB, this
script runs an independent bootstrap particle filter for each arm, computes
privileged oracle marginal activation values for posterior particles, and
converts those values into posterior ranks and Top-K membership probabilities.

This diagnostic is for visualization only. BIRD does not use particles, hidden
states, simulator parameters, or oracle values during training.
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


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class JointHistory:
    obs: np.ndarray
    actions: np.ndarray
    true_priorities: np.ndarray
    true_types: np.ndarray
    time_index: int


@dataclass
class RankResult:
    priorities: np.ndarray
    ranks: np.ndarray
    topk_probs: np.ndarray
    mean_priorities: np.ndarray
    query_arm: int
    query_arms: np.ndarray
    true_priorities: np.ndarray
    true_types: np.ndarray
    time_index: int


def _load_adapt_modules():
    for name in ["env", "baselines"]:
        if name in sys.modules:
            del sys.modules[name]
    path = str(ROOT / "synthetic-drifting")
    sys.path.insert(0, path)
    try:
        env_mod = importlib.import_module("env")
    finally:
        sys.path.remove(path)
    return env_mod


def _random_action(rng: np.random.Generator, n_arms: int, budget: int) -> np.ndarray:
    action = np.zeros(n_arms, dtype=np.int32)
    action[rng.choice(n_arms, size=budget, replace=False)] = 1
    return action


def _normalize_log_weights(log_w: np.ndarray) -> np.ndarray:
    log_w = log_w - np.max(log_w)
    weights = np.exp(log_w)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.full_like(weights, 1.0 / len(weights), dtype=np.float64)
    return weights / total


def _resample(rng: np.random.Generator, weights: np.ndarray) -> np.ndarray:
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cdf = np.cumsum(weights)
    return np.searchsorted(cdf, positions, side="right")


def _oracle_marginal_vector(env_mod, cfg, x0, alpha0, beta0, theta, horizon, gamma):
    theta = theta.astype(np.int32)
    alpha_bar = np.asarray([env_mod.TYPE_ALPHA_MEAN[int(m)] for m in theta], dtype=np.float64)
    beta_bar = np.asarray([env_mod.TYPE_BETA_MEAN[int(m)] for m in theta], dtype=np.float64)
    x_active = x0.astype(np.float64).copy()
    x_passive = x0.astype(np.float64).copy()
    alpha = alpha0.astype(np.float64).copy()
    beta = beta0.astype(np.float64).copy()
    values = np.zeros_like(x_active, dtype=np.float64)

    for h in range(horizon):
        alpha = np.clip(
            alpha_bar + cfg.ou_rho * (alpha - alpha_bar),
            cfg.alpha_lo,
            cfg.alpha_hi,
        )
        beta = np.clip(
            beta_bar + cfg.ou_rho * (beta - beta_bar),
            cfg.beta_lo,
            cfg.beta_hi,
        )
        x_active = alpha * x_active + beta * (1.0 if h == 0 else 0.0) - cfg.drift
        x_passive = alpha * x_passive - cfg.drift
        values += (gamma ** (h + 1)) * (np.maximum(x_active, 0.0) - np.maximum(x_passive, 0.0))

    return values.astype(np.float32)


def _particle_priorities_for_arm(env_mod, cfg, arm_obs, arm_actions, n_particles, seed, args):
    rng = np.random.default_rng(seed)
    theta = rng.integers(0, cfg.M, size=n_particles)
    alpha = np.asarray([env_mod.TYPE_ALPHA_MEAN[int(m)] for m in theta], dtype=np.float64)
    beta = np.asarray([env_mod.TYPE_BETA_MEAN[int(m)] for m in theta], dtype=np.float64)
    x = cfg.init_mean + cfg.init_std * rng.standard_normal(n_particles)

    log_w = -0.5 * ((float(arm_obs[0]) - x) / cfg.sigma_v) ** 2
    weights = _normalize_log_weights(log_w)
    idx = _resample(rng, weights)
    theta, alpha, beta, x = theta[idx], alpha[idx], beta[idx], x[idx]

    for k, action in enumerate(arm_actions):
        x = alpha * x + beta * float(action) - cfg.drift + cfg.sigma_w * rng.standard_normal(n_particles)
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

        log_w = -0.5 * ((float(arm_obs[k + 1]) - x) / cfg.sigma_v) ** 2
        weights = _normalize_log_weights(log_w)
        ess = 1.0 / np.sum(weights * weights)
        if ess < 0.5 * n_particles:
            idx = _resample(rng, weights)
            theta, alpha, beta, x = theta[idx], alpha[idx], beta[idx], x[idx]

    priorities = _oracle_marginal_vector(
        env_mod,
        cfg,
        x,
        alpha,
        beta,
        theta,
        args.oracle_horizon,
        args.gamma,
    )
    return priorities, theta.astype(np.int32)


def _compute_rank_result(env_mod, cfg, hist: JointHistory, n_particles: int, seed: int, args) -> RankResult:
    all_priorities = []
    for arm in range(cfg.N):
        priorities, _ = _particle_priorities_for_arm(
            env_mod,
            cfg,
            hist.obs[:, arm],
            hist.actions[:, arm],
            n_particles,
            seed + 17 * arm,
            args,
        )
        all_priorities.append(priorities)

    priorities = np.stack(all_priorities, axis=1)
    order = np.argsort(-priorities, axis=1)
    ranks = np.empty_like(order, dtype=np.int32)
    row = np.arange(n_particles)[:, None]
    ranks[row, order] = np.arange(1, cfg.N + 1, dtype=np.int32)
    topk_probs = (ranks <= cfg.K).mean(axis=0)
    mean_priorities = priorities.mean(axis=0)

    uncertainty = topk_probs * (1.0 - topk_probs)
    rank_spread = ranks.std(axis=0) / max(float(cfg.N), 1.0)
    arm_scores = uncertainty + 0.15 * rank_spread
    query_arms = np.argsort(-arm_scores)[:2].astype(np.int32)
    query_arm = int(query_arms[0])

    return RankResult(
        priorities=priorities,
        ranks=ranks,
        topk_probs=topk_probs,
        mean_priorities=mean_priorities,
        query_arm=query_arm,
        query_arms=query_arms,
        true_priorities=hist.true_priorities,
        true_types=hist.true_types,
        time_index=hist.time_index,
    )


def _collect_histories(env_mod, cfg, args) -> list[JointHistory]:
    rng = np.random.default_rng(args.seed + 50)
    histories = []
    per_episode = max(1, args.candidates // max(args.query_episodes, 1))

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

        for _ in range(per_episode):
            t = int(rng.integers(args.history_len, len(obs_traj)))
            obs_window = np.asarray(obs_traj[t - args.history_len : t + 1], dtype=np.float32)
            action_window = np.asarray(action_traj[t - args.history_len : t], dtype=np.float32)
            current = info_traj[t]
            true_priorities = _oracle_marginal_vector(
                env_mod,
                cfg,
                current["x_true"].astype(np.float64),
                current["alpha_true"].astype(np.float64),
                current["beta_true"].astype(np.float64),
                current["theta_true"].astype(np.int32),
                args.oracle_horizon,
                args.gamma,
            )
            histories.append(
                JointHistory(
                    obs=obs_window,
                    actions=action_window,
                    true_priorities=true_priorities,
                    true_types=current["theta_true"].astype(np.int32),
                    time_index=t,
                )
            )

    return histories


def _select_history(env_mod, cfg, histories: list[JointHistory], args) -> RankResult:
    best_score = -np.inf
    best_result = None
    for idx, hist in enumerate(histories[: args.candidates]):
        result = _compute_rank_result(env_mod, cfg, hist, args.search_particles, args.seed + 1000 + 31 * idx, args)
        probs = result.topk_probs
        uncertainty = float(np.max(probs * (1.0 - probs)))
        spread = float(np.max(result.ranks.std(axis=0)))
        score = uncertainty + 0.03 * spread
        if score > best_score:
            best_score = score
            best_result = result
            best_hist = hist

    assert best_result is not None
    return _compute_rank_result(env_mod, cfg, best_hist, args.particles, args.seed + 5000, args)


def _plot(result: RankResult, cfg, out_png: Path) -> None:
    query_arms = result.query_arms[:2]
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.15, 2.55), sharey=True)
    bins = np.arange(1, cfg.N + 2) - 0.5
    bar_colors = ["#8fb3da", "#8ed0c6"]
    line_colors = ["#204b83", "#16766f"]

    for panel, (ax, query) in enumerate(zip(axes, query_arms), start=1):
        ranks = result.ranks[:, int(query)]
        counts, _, patches = ax.hist(
            ranks,
            bins=bins,
            density=True,
            color=bar_colors[panel - 1],
            edgecolor="white",
            linewidth=0.65,
            alpha=0.58,
        )
        ax.axvspan(0.5, cfg.K + 0.5, color="#d95f5f", alpha=0.075, lw=0, zorder=0)
        ax.axvline(
            cfg.K + 0.5,
            color="#2b2b2b",
            linestyle="--",
            linewidth=1.05,
            label=f"Top-{cfg.K} cutoff",
        )
        kde_x = np.linspace(1.0, 11.0, 240)
        bandwidth = 0.55
        kde = np.exp(-0.5 * ((kde_x[:, None] - ranks[None, :]) / bandwidth) ** 2).mean(axis=1)
        kde = kde / (bandwidth * np.sqrt(2.0 * np.pi))
        ax.plot(kde_x, kde, color=line_colors[panel - 1], linewidth=2.0, alpha=0.98)

        ax.set_xlim(0.5, 11.5)
        ax.set_xticks([1, 3, cfg.K, 7, 9, 11])
        ax.set_xlabel(f"Oracle rank of arm {int(query)}")
        ax.set_title(f"Boundary arm {panel}")
        ax.grid(False)
        ax.legend(frameon=False, loc="upper right")
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#333333")
            ax.spines[spine].set_linewidth(0.8)
        ax.text(
            0.04,
            0.90,
            f"Pr(a=1)={result.topk_probs[int(query)]:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#d8d8d8", "linewidth": 0.45, "alpha": 0.92, "pad": 2.6},
        )

    axes[0].set_ylabel("Posterior density")
    fig.subplots_adjust(left=0.08, right=0.985, top=0.86, bottom=0.20, wspace=0.18)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_npz(result: RankResult, out_npz: Path) -> None:
    np.savez(
        out_npz,
        priorities=result.priorities,
        ranks=result.ranks,
        topk_probs=result.topk_probs,
        mean_priorities=result.mean_priorities,
        query_arm=np.asarray(result.query_arm, dtype=np.int32),
        query_arms=result.query_arms,
        true_priorities=result.true_priorities,
        true_types=result.true_types,
        time_index=np.asarray(result.time_index, dtype=np.int32),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_arms", type=int, default=20)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--history_len", type=int, default=20)
    parser.add_argument("--query_episodes", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=18)
    parser.add_argument("--search_particles", type=int, default=800)
    parser.add_argument("--particles", type=int, default=5000)
    parser.add_argument("--oracle_horizon", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_png", type=Path, default=ROOT / "adapt_lr_particle_rank_posterior.png")
    parser.add_argument("--out_npz", type=Path, default=ROOT / "adapt_lr_particle_rank_posterior.npz")
    args = parser.parse_args()

    env_mod = _load_adapt_modules()
    cfg = env_mod.AdaptRMABConfig(N=args.n_arms, K=args.budget, T=args.horizon)
    histories = _collect_histories(env_mod, cfg, args)
    result = _select_history(env_mod, cfg, histories, args)

    _plot(result, cfg, args.out_png)
    _save_npz(result, args.out_npz)

    query = result.query_arm
    print(
        f"[adapt-lr] time={result.time_index} selected_arm={query} "
        f"topk_prob={result.topk_probs[query]:.3f} "
        f"rank_mean={result.ranks[:, query].mean():.2f} "
        f"rank_std={result.ranks[:, query].std():.2f} "
        f"rank_p10={np.percentile(result.ranks[:, query], 10):.1f} "
        f"rank_p90={np.percentile(result.ranks[:, query], 90):.1f}"
    )
    for arm in result.query_arms[1:]:
        arm = int(arm)
        print(
            f"[adapt-lr] selected_arm={arm} "
            f"topk_prob={result.topk_probs[arm]:.3f} "
            f"rank_mean={result.ranks[:, arm].mean():.2f} "
            f"rank_std={result.ranks[:, arm].std():.2f} "
            f"rank_p10={np.percentile(result.ranks[:, arm], 10):.1f} "
            f"rank_p90={np.percentile(result.ranks[:, arm], 90):.1f}"
        )
    print(f"[saved] {args.out_png}")
    print(f"[saved] {args.out_npz}")


if __name__ == "__main__":
    main()
