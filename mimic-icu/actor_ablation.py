"""
Actor ablation for BIRD/DPMD on mimic-icu.

Compares only the deterministic MLP actor ablation against full BIRD.

Outputs:
  actor_ablation_N{N}_K{K}.npz
  actor_ablation_N{N}_K{K}.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from env import MIMICRMABConfig, MIMICRMABEnv


SEED_OFFSET = 2000
LABELS = {"mlp_actor": "MLP Actor", "joint_N": "BIRD Joint-N", "dpmd": "BIRD"}
COLORS = {"mlp_actor": "#e377c2", "joint_N": "#ff7f0e", "dpmd": "#2ca02c"}


def _load_dpmd(ckpt_path: str, env_cfg: MIMICRMABConfig):
    from diffusion_DPMD_train import DPMDAgent, DPMDTrainConfig
    ckpt = torch.load(ckpt_path, map_location="cpu")
    saved = ckpt.get("cfg", {})
    cfg = DPMDTrainConfig(N=env_cfg.N, K=env_cfg.K, T=env_cfg.T, obs_dim=env_cfg.obs_dim)
    for key in ("z_dim", "encoder_hidden", "encoder_heads", "encoder_layers", "L",
                "actor_hidden", "actor_t_dim", "T_diff", "score_clip",
                "critic_hidden", "action_candidates", "target_action_candidates"):
        if key in saved:
            setattr(cfg, key, saved[key])
    agent = DPMDAgent(N=cfg.N, K=cfg.K, cfg=cfg)
    agent.load_checkpoint(ckpt_path)
    return agent


def _load_joint_n(ckpt_path: str, env_cfg: MIMICRMABConfig):
    from diffusion_joint_N_ablation import JointDPMDAgent
    from diffusion_DPMD_train import DPMDTrainConfig
    ckpt = torch.load(ckpt_path, map_location="cpu")
    saved = ckpt.get("cfg", {})
    cfg = DPMDTrainConfig(N=env_cfg.N, K=env_cfg.K, T=env_cfg.T, obs_dim=env_cfg.obs_dim)
    for key in ("z_dim", "encoder_hidden", "encoder_heads", "encoder_layers", "L",
                "actor_hidden", "actor_t_dim", "T_diff", "score_clip",
                "critic_hidden", "action_candidates", "target_action_candidates"):
        if key in saved:
            setattr(cfg, key, saved[key])
    agent = JointDPMDAgent(N=cfg.N, K=cfg.K, cfg=cfg)
    agent.load_checkpoint(ckpt_path)
    return agent


def _load_mlp_actor(ckpt_path: str, env_cfg: MIMICRMABConfig):
    from mlp_actor import MLPActorAgent, MLPActorConfig
    ckpt = torch.load(ckpt_path, map_location="cpu")
    saved = ckpt.get("cfg", {})
    cfg = MLPActorConfig(N=env_cfg.N, K=env_cfg.K, T=env_cfg.T, obs_dim=env_cfg.obs_dim)
    for key in ("z_dim", "encoder_hidden", "encoder_heads", "encoder_layers",
                "L", "actor_hidden", "critic_hidden"):
        if key in saved:
            setattr(cfg, key, saved[key])
    agent = MLPActorAgent(N=cfg.N, K=cfg.K, cfg=cfg)
    agent.load_checkpoint(ckpt_path)
    return agent


def _eval_agent(agent, env_cfg: MIMICRMABConfig, n_episodes: int, seed: int) -> np.ndarray:
    returns = []
    for ep in range(n_episodes):
        env = MIMICRMABEnv(env_cfg, seed=seed + ep)
        obs, _ = env.reset()
        if hasattr(agent, "reset_history"):
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


def _save_npz(path: Path, results: dict[str, np.ndarray],
              env_name: str, args: argparse.Namespace) -> None:
    names = list(results.keys())
    np.savez(
        path,
        method_names=np.asarray(names),
        mean_returns=np.asarray([results[m].mean() for m in names], dtype=np.float32),
        std_returns=np.asarray([results[m].std() for m in names], dtype=np.float32),
        env_name=np.asarray(env_name),
        N=np.asarray(args.N, dtype=np.int32),
        K=np.asarray(args.K, dtype=np.int32),
        T=np.asarray(args.T, dtype=np.int32),
        seed=np.asarray(args.seed, dtype=np.int32),
        n_episodes=np.asarray(args.n_episodes, dtype=np.int32),
        **{m: results[m] for m in names},
    )


def _plot(results: dict[str, np.ndarray], out_path: str) -> None:
    names = list(results.keys())
    means = [results[n].mean() for n in names]
    stds = [results[n].std() for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    bars = ax.bar(
        x, means, yerr=stds, capsize=5,
        color=[COLORS.get(n, "#8c8c8c") for n in names],
        edgecolor="black", linewidth=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(n, n) for n in names])
    ax.set_ylabel("Episode return")
    ax.grid(True, axis="y", alpha=0.3)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{m:.1f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlp_actor_ckpt", default="checkpoints_mlp_actor/best.pth")
    parser.add_argument("--joint_n_ckpt", default="checkpoints_dpmd_joint_N/best.pth")
    parser.add_argument("--dpmd_ckpt", default="checkpoints_dpmd/best.pth")
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    env_cfg = MIMICRMABConfig(N=args.N, K=args.K, T=args.T)
    variants = [
        ("mlp_actor", args.mlp_actor_ckpt, _load_mlp_actor),
        ("joint_N", args.joint_n_ckpt, _load_joint_n),
        ("dpmd", args.dpmd_ckpt, _load_dpmd),
    ]

    results: dict[str, np.ndarray] = {}
    for name, ckpt_path, loader in variants:
        if not Path(ckpt_path).exists():
            print(f"  {name:12s} [checkpoint not found: {ckpt_path}]")
            continue
        agent = loader(ckpt_path, env_cfg)
        for attr in ("encoder", "actor", "critic"):
            if hasattr(agent, attr):
                getattr(agent, attr).eval()
        rets = _eval_agent(agent, env_cfg, args.n_episodes, SEED_OFFSET + args.seed)
        results[name] = rets
        print(f"  {name:12s} mean={rets.mean():.2f} std={rets.std():.2f}")

    if not results:
        print("No checkpoints found.")
        return

    stem = args.out or f"actor_ablation_N{args.N}_K{args.K}.png"
    out_png = Path(stem)
    out_npz = out_png.with_suffix(".npz")
    _save_npz(out_npz, results, "mimic-icu", args)
    _plot(results, str(out_png))
    print(f"[saved] {out_npz}")
    print(f"[saved] {out_png}")


if __name__ == "__main__":
    main()
