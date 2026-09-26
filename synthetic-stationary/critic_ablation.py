"""
Critic architecture ablation for BIRD/DPMD on synthetic-stationary.

Compares:
  per_arm_twin    default BIRD additive per-arm twin critic
  per_arm_single  additive per-arm single critic
  joint_twin      joint critic over all arms/actions with twin Q heads

Outputs:
  experiment_outputs/critic_ablation_N{N}_K{K}.npz
  experiment_outputs/critic_ablation_N{N}_K{K}.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import diffusion_DPMD_train as dpmd
from diffusion_model import ArmCriticNet
from env import AdaptRMABConfig, AdaptRMABEnv


OriginalCritic = dpmd.PerArmTwinCritic


class PerArmSingleCritic(nn.Module):
    def __init__(self, n_arms: int, z_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.n_arms = int(n_arms)
        self.q_net = ArmCriticNet(z_dim=z_dim, hidden_dim=hidden_dim)

    def forward(self, z: torch.Tensor, action: torch.Tensor):
        action = action.float()
        batch, n_arms, z_dim = z.shape
        q = self.q_net(
            z.reshape(batch * n_arms, z_dim),
            action.reshape(batch * n_arms),
        ).view(batch, n_arms)
        return q, q.detach()

    def min_q(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q, _ = self(z, action)
        return q


class JointCriticNet(nn.Module):
    def __init__(self, n_arms: int, z_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.n_arms = int(n_arms)
        self.z_dim = int(z_dim)
        self.net = nn.Sequential(
            nn.Linear(self.n_arms * (self.z_dim + 1), hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, self.n_arms),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        batch = z.size(0)
        x = torch.cat([z.reshape(batch, -1), action.float()], dim=-1)
        return self.net(x)


class JointTwinCritic(nn.Module):
    def __init__(self, n_arms: int, z_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.q1_net = JointCriticNet(n_arms=n_arms, z_dim=z_dim, hidden_dim=hidden_dim)
        self.q2_net = JointCriticNet(n_arms=n_arms, z_dim=z_dim, hidden_dim=hidden_dim)

    def forward(self, z: torch.Tensor, action: torch.Tensor):
        return self.q1_net(z, action), self.q2_net(z, action)

    def min_q(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(z, action)
        return torch.min(q1, q2)


CRITICS = {
    "per_arm_twin": OriginalCritic,
    "per_arm_single": PerArmSingleCritic,
    "joint_twin": JointTwinCritic,
}
LABELS = {
    "per_arm_twin": "Per-arm Twin",
    "per_arm_single": "Per-arm Single",
    "joint_twin": "Joint Twin",
}
COLORS = {
    "per_arm_twin": "#2ca02c",
    "per_arm_single": "#d62728",
    "joint_twin": "#1f77b4",
}


def evaluate_agent(agent, env_cfg: AdaptRMABConfig, n_episodes: int, seed: int) -> np.ndarray:
    returns = []
    for ep in range(n_episodes):
        env = AdaptRMABEnv(env_cfg, seed=seed + ep)
        obs, _ = env.reset()
        agent.reset_history()
        ep_return = 0.0
        for _ in range(env_cfg.T):
            action = agent.act_hard(obs)
            obs, reward_vec, done, _ = env.step(action)
            ep_return += float(reward_vec.sum())
            if done:
                break
        returns.append(ep_return)
    return np.asarray(returns, dtype=np.float32)


def plot_results(results: dict[str, np.ndarray], out_path: Path) -> None:
    names = list(results.keys())
    means = [results[n].mean() for n in names]
    stds = [results[n].std() for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    bars = ax.bar(
        x, means, yerr=stds, capsize=5,
        color=[COLORS.get(n, "#8c8c8c") for n in names],
        edgecolor="black", linewidth=0.8,
    )
    if "per_arm_twin" in names:
        bars[names.index("per_arm_twin")].set_edgecolor("red")
        bars[names.index("per_arm_twin")].set_linewidth(2.2)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(n, n) for n in names], rotation=15, ha="right")
    ax.set_ylabel("Episode return")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{m:.1f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close(fig)


def _train_or_load(args: argparse.Namespace, env_cfg: AdaptRMABConfig,
                   method: str, seed: int):
    save_dir = Path(args.save_dir_template.format(method=method, seed=seed))
    ckpt_path = save_dir / "best.pth"
    if ckpt_path.exists() and not args.force_retrain:
        cfg = dpmd.DPMDTrainConfig(N=args.N, K=args.K, T=args.T, seed=seed)
        agent = dpmd.DPMDAgent(N=args.N, K=args.K, cfg=cfg)
        agent.load_checkpoint(ckpt_path)
        return agent

    cfg = dpmd.DPMDTrainConfig(
        N=args.N, K=args.K, T=args.T, seed=seed, epochs=args.epochs,
        save_dir=str(save_dir),
    )
    env = AdaptRMABEnv(env_cfg, seed=seed)
    return dpmd.train(env, cfg=cfg, checkpoint_dir=cfg.save_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--train_seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--methods", nargs="+", default=list(CRITICS.keys()),
                        choices=list(CRITICS.keys()))
    parser.add_argument("--save_dir_template", default="checkpoints_dpmd_critic_ablation_{method}_seed{seed}")
    parser.add_argument("--out_dir", default="experiment_outputs")
    parser.add_argument("--force_retrain", action="store_true")
    args = parser.parse_args()

    env_cfg = AdaptRMABConfig(N=args.N, K=args.K, T=args.T)
    raw_returns = np.zeros(
        (len(args.methods), len(args.train_seeds), args.n_episodes), dtype=np.float32)
    results: dict[str, np.ndarray] = {}

    try:
        for method_idx, method in enumerate(args.methods):
            dpmd.PerArmTwinCritic = CRITICS[method]
            method_returns = []
            for seed_idx, seed in enumerate(args.train_seeds):
                print("=" * 72)
                print(f"Training critic ablation: method={method} seed={seed}")
                print("=" * 72)
                agent = _train_or_load(args, env_cfg, method, seed)
                returns = evaluate_agent(agent, env_cfg, args.n_episodes, seed=seed + 1000)
                raw_returns[method_idx, seed_idx] = returns
                method_returns.append(returns)
                print(f"[{method} seed={seed}] mean={returns.mean():.2f} std={returns.std():.2f}")
            results[method] = np.concatenate(method_returns)
    finally:
        dpmd.PerArmTwinCritic = OriginalCritic

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"critic_ablation_N{args.N}_K{args.K}"
    names = list(results.keys())
    np.savez(
        f"{stem}.npz",
        method_names=np.asarray(names),
        mean_returns=np.asarray([results[n].mean() for n in names], dtype=np.float32),
        std_returns=np.asarray([results[n].std() for n in names], dtype=np.float32),
        train_seeds=np.asarray(args.train_seeds, dtype=np.int32),
        raw_returns=raw_returns,
        env_name=np.asarray("synthetic-stationary"),
        N=np.asarray(args.N, dtype=np.int32),
        K=np.asarray(args.K, dtype=np.int32),
        T=np.asarray(args.T, dtype=np.int32),
        n_episodes=np.asarray(args.n_episodes, dtype=np.int32),
    )
    plot_results(results, Path(f"{stem}.png"))
    print(f"[saved] {stem}.npz")
    print(f"[saved] {stem}.png")


if __name__ == "__main__":
    main()
