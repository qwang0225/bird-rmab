"""
Boundary-state oracle value diagnostic for adapt-lr.

This diagnostic asks whether BIRD's stochastic score samples produce
higher-value Top-K activation sets than a deterministic MLP actor on histories
where the oracle Top-K boundary is ambiguous. The evaluator is privileged
simulator-oracle marginal value, not either method's learned critic, so the
comparison is not circular.

BIRD training does not use this oracle. This script is for analysis only.
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
import torch


ROOT = Path(__file__).resolve().parents[1]
ADAPT_DIR = ROOT / "synthetic-drifting"


@dataclass
class HistoryBatch:
    obs_hist: np.ndarray
    act_hist: np.ndarray
    oracle_priorities: np.ndarray
    oracle_gap: np.ndarray


def _load_adapt_modules():
    for name in [
        "env",
        "baselines",
        "diffusion_model",
        "diffusion_DPMD_train",
        "mlp_actor",
    ]:
        if name in sys.modules:
            del sys.modules[name]
    sys.path.insert(0, str(ADAPT_DIR))
    try:
        env_mod = importlib.import_module("env")
        dpmd_mod = importlib.import_module("diffusion_DPMD_train")
        mlp_mod = importlib.import_module("mlp_actor")
        diff_mod = importlib.import_module("diffusion_model")
    finally:
        sys.path.remove(str(ADAPT_DIR))
    return env_mod, dpmd_mod, mlp_mod, diff_mod


def _random_action(rng: np.random.Generator, n_arms: int, budget: int) -> np.ndarray:
    action = np.zeros(n_arms, dtype=np.int64)
    action[rng.choice(n_arms, size=budget, replace=False)] = 1
    return action


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


def _collect_histories(env_mod, cfg, args) -> HistoryBatch:
    rng = np.random.default_rng(args.seed + 37)
    obs_histories = []
    act_histories = []
    priorities = []
    gaps = []

    for ep in range(args.episodes):
        env = env_mod.AdaptRMABEnv(cfg, seed=args.seed + ep)
        obs, info = env.reset()
        obs_hist = np.zeros((cfg.N, args.history_len), dtype=np.float32)
        act_hist = np.zeros((cfg.N, args.history_len), dtype=np.float32)

        for t in range(cfg.T):
            obs_hist = np.roll(obs_hist, -1, axis=1)
            obs_hist[:, -1] = obs.astype(np.float32)

            if t >= args.history_len:
                oracle_p = _oracle_marginal_vector(
                    env_mod,
                    cfg,
                    info["x_true"],
                    info["alpha_true"],
                    info["beta_true"],
                    info["theta_true"],
                    args.oracle_horizon,
                    args.gamma,
                )
                sorted_p = np.sort(oracle_p)[::-1]
                gap = float(sorted_p[cfg.K - 1] - sorted_p[cfg.K])
                obs_histories.append(obs_hist.copy())
                act_histories.append(act_hist.copy())
                priorities.append(oracle_p)
                gaps.append(gap)

            action = _random_action(rng, cfg.N, cfg.K)
            obs, _, done, info = env.step(action)
            act_hist = np.roll(act_hist, -1, axis=1)
            act_hist[:, -1] = action.astype(np.float32)
            if done:
                break

    obs_arr = np.asarray(obs_histories, dtype=np.float32)
    act_arr = np.asarray(act_histories, dtype=np.float32)
    pr_arr = np.asarray(priorities, dtype=np.float32)
    gap_arr = np.asarray(gaps, dtype=np.float32)

    keep = np.argsort(gap_arr)[: min(args.num_histories, len(gap_arr))]
    return HistoryBatch(
        obs_hist=obs_arr[keep],
        act_hist=act_arr[keep],
        oracle_priorities=pr_arr[keep],
        oracle_gap=gap_arr[keep],
    )


def _topk_numpy(scores: np.ndarray, k: int) -> np.ndarray:
    action = np.zeros_like(scores, dtype=np.float32)
    idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(scores.shape[0])[:, None]
    action[rows, idx] = 1.0
    return action


@torch.no_grad()
def _evaluate_actions(env_mod, dpmd_mod, mlp_mod, diff_mod, batch: HistoryBatch, args):
    device = torch.device(args.device)
    dpmd_cfg = dpmd_mod.DPMDTrainConfig(device=args.device)
    mlp_cfg = mlp_mod.MLPActorConfig(device=args.device)
    dpmd_agent = dpmd_mod.DPMDAgent(N=args.n_arms, K=args.budget, cfg=dpmd_cfg)
    mlp_agent = mlp_mod.MLPActorAgent(N=args.n_arms, K=args.budget, cfg=mlp_cfg)
    dpmd_agent.load_checkpoint(ADAPT_DIR / args.bird_ckpt)
    mlp_agent.load_checkpoint(ADAPT_DIR / args.mlp_ckpt)
    dpmd_agent.encoder.eval()
    dpmd_agent.actor.eval()
    mlp_agent.encoder.eval()
    mlp_agent.actor.eval()

    oh = torch.from_numpy(batch.obs_hist).float().to(device)
    ah = torch.from_numpy(batch.act_hist).float().to(device)

    z_mlp = mlp_agent.encoder(oh, ah)
    mlp_scores = mlp_agent.actor(z_mlp).detach().cpu().numpy()
    mlp_action = _topk_numpy(mlp_scores, args.budget)
    mlp_value = np.sum(mlp_action * batch.oracle_priorities, axis=1)

    z_bird = dpmd_agent.encoder(oh, ah)
    n_hist = batch.obs_hist.shape[0]
    bird_values = np.zeros((n_hist, args.bird_samples), dtype=np.float32)

    for start in range(0, args.bird_samples, args.sample_chunk):
        chunk = min(args.sample_chunk, args.bird_samples - start)
        scores = dpmd_agent.actor.sample(z_bird, num_samples=chunk)
        scores_np = scores.detach().cpu().numpy()
        for j in range(chunk):
            action = _topk_numpy(scores_np[:, j, :], args.budget)
            bird_values[:, start + j] = np.sum(action * batch.oracle_priorities, axis=1)

    bird_mean = bird_values.mean(axis=1)
    bird_sample_diff = bird_values - mlp_value[:, None]
    bird_mean_diff = bird_mean - mlp_value
    return {
        "mlp_value": mlp_value,
        "bird_values": bird_values,
        "bird_mean": bird_mean,
        "bird_sample_diff": bird_sample_diff,
        "bird_mean_diff": bird_mean_diff,
    }


def _plot(results, batch: HistoryBatch, out_png: Path) -> None:
    diffs = results["bird_mean_diff"]
    sample_diffs = results["bird_sample_diff"].reshape(-1)
    mean_diff = float(diffs.mean())
    frac_positive = float((diffs > 0.0).mean())
    sample_frac_positive = float((sample_diffs > 0.0).mean())

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
    fig, ax = plt.subplots(figsize=(3.65, 2.55))
    bins = np.linspace(
        float(np.percentile(diffs, 2)),
        float(np.percentile(diffs, 98)),
        22,
    )
    if np.allclose(bins[0], bins[-1]):
        bins = 18

    ax.hist(
        diffs,
        bins=bins,
        density=True,
        color="#92b7df",
        edgecolor="white",
        linewidth=0.65,
        alpha=0.62,
    )
    kde_x = np.linspace(float(diffs.min()), float(diffs.max()), 240)
    bandwidth = max(float(diffs.std()) * 0.35, 1e-3)
    kde = np.exp(-0.5 * ((kde_x[:, None] - diffs[None, :]) / bandwidth) ** 2).mean(axis=1)
    kde = kde / (bandwidth * np.sqrt(2.0 * np.pi))
    ax.plot(kde_x, kde, color="#235894", linewidth=2.0)
    ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    ax.axvline(mean_diff, color="#d95f5f", linewidth=1.35)
    ax.set_xlabel(r"Oracle value difference: BIRD samples $-$ MLP")
    ax.set_ylabel("Density")
    ax.grid(False)
    ax.text(
        0.04,
        0.94,
        f"mean={mean_diff:.2f}\nPr(mean>0)={frac_positive:.2f}\nPr(sample>0)={sample_frac_positive:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#d8d8d8", "linewidth": 0.45, "alpha": 0.92, "pad": 2.6},
    )
    fig.subplots_adjust(left=0.16, right=0.98, top=0.96, bottom=0.22)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_npz(results, batch: HistoryBatch, out_npz: Path) -> None:
    np.savez(
        out_npz,
        oracle_gap=batch.oracle_gap,
        oracle_priorities=batch.oracle_priorities,
        mlp_value=results["mlp_value"],
        bird_values=results["bird_values"],
        bird_mean=results["bird_mean"],
        bird_sample_diff=results["bird_sample_diff"],
        bird_mean_diff=results["bird_mean_diff"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_arms", type=int, default=20)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--history_len", type=int, default=40)
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--num_histories", type=int, default=120)
    parser.add_argument("--bird_samples", type=int, default=64)
    parser.add_argument("--sample_chunk", type=int, default=16)
    parser.add_argument("--oracle_horizon", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bird_ckpt", type=str, default="checkpoints_dpmd/best.pth")
    parser.add_argument("--mlp_ckpt", type=str, default="checkpoints_mlp_actor/best.pth")
    parser.add_argument("--out_png", type=Path, default=ROOT / "adapt_lr_boundary_value_diagnostic.png")
    parser.add_argument("--out_npz", type=Path, default=ROOT / "adapt_lr_boundary_value_diagnostic.npz")
    args = parser.parse_args()

    env_mod, dpmd_mod, mlp_mod, diff_mod = _load_adapt_modules()
    cfg = env_mod.AdaptRMABConfig(N=args.n_arms, K=args.budget, T=args.horizon)
    batch = _collect_histories(env_mod, cfg, args)
    results = _evaluate_actions(env_mod, dpmd_mod, mlp_mod, diff_mod, batch, args)
    _plot(results, batch, args.out_png)
    _save_npz(results, batch, args.out_npz)

    diffs = results["bird_mean_diff"]
    sample_diffs = results["bird_sample_diff"].reshape(-1)
    print(
        f"[adapt-lr boundary value] histories={len(diffs)} bird_samples={args.bird_samples} "
        f"mean_diff={diffs.mean():.3f} median_diff={np.median(diffs):.3f} "
        f"Pr(mean_diff>0)={(diffs > 0).mean():.3f} "
        f"Pr(sample_diff>0)={(sample_diffs > 0).mean():.3f} "
        f"mean_oracle_gap={batch.oracle_gap.mean():.3f}"
    )
    print(f"[saved] {args.out_png}")
    print(f"[saved] {args.out_npz}")


if __name__ == "__main__":
    main()
