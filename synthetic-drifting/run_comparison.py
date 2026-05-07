"""
run_comparison.py  (adapt)

Compare all policies on AdaptRMAB (hidden types + drifting dynamics).

Policies:
  random           -- uniform random K arms
  oracle           -- knows true alpha/beta, activates top-K by active_SS
  neurwin          -- NeurWIN point-estimate  (checkpoints_neurwin/best.pth)
  dpmd             -- Diffusion DPMD          (checkpoints_dpmd/best.pth)
  ppo              -- PPO deep RL baseline    (checkpoints_ppo/best.pth)

Usage:
  python run_comparison.py
  python run_comparison.py --n_episodes 100 --N 20 --K 5
  python run_comparison.py --dpmd_ckpt checkpoints_dpmd/best.pth
  python run_comparison.py --ppo_ckpt checkpoints_ppo/best.pth
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

from env import AdaptRMABConfig, AdaptRMABEnv, TYPE_ALPHA_MEAN, TYPE_BETA_MEAN, TYPE_NAMES
from baselines import RandomPolicy, GreedyObsPolicy, OracleGreedyPolicy, OracleLookaheadPolicy, evaluate_policy


# ---------------------------------------------------------------------------
# Agent evaluation helpers
# ---------------------------------------------------------------------------

def _load_neurwin(ckpt_path: str, env_cfg: AdaptRMABConfig) -> object:
    from neurwin import Agent, TrainConfig
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    saved = ckpt.get("cfg", {})
    cfg   = TrainConfig()
    cfg.N = env_cfg.N; cfg.K = env_cfg.K; cfg.T = env_cfg.T
    for key in ("z_dim", "encoder_hidden", "encoder_heads", "encoder_layers",
                "L", "index_hidden", "sigmoid_scale"):
        if key in saved:
            setattr(cfg, key, saved[key])
    agent = Agent(N=cfg.N, K=cfg.K, cfg=cfg)
    agent.load_checkpoint(ckpt_path)
    return agent


def _load_dpmd(ckpt_path: str, env_cfg: AdaptRMABConfig) -> object:
    from diffusion_DPMD_train import DPMDAgent, DPMDTrainConfig
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    saved = ckpt.get("cfg", {})
    cfg   = DPMDTrainConfig()
    cfg.N = env_cfg.N; cfg.K = env_cfg.K; cfg.T = env_cfg.T
    for key in ("z_dim", "encoder_hidden", "encoder_heads", "encoder_layers", "L",
                "actor_hidden", "actor_t_dim", "T_diff", "score_clip",
                "critic_hidden", "action_candidates"):
        if key in saved:
            setattr(cfg, key, saved[key])
    agent = DPMDAgent(N=cfg.N, K=cfg.K, cfg=cfg)
    agent.load_checkpoint(ckpt_path)
    return agent


def _load_ppo(ckpt_path: str, env_cfg: AdaptRMABConfig) -> object:
    from ppo import PPOAgent, PPOConfig
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    saved = ckpt.get("cfg", {})
    cfg   = PPOConfig()
    cfg.N = env_cfg.N; cfg.K = env_cfg.K; cfg.T = env_cfg.T
    for key in ("L", "arm_enc_hidden", "z_dim"):
        if key in saved:
            setattr(cfg, key, saved[key])
    agent = PPOAgent(N=cfg.N, K=cfg.K, cfg=cfg)
    agent.load_checkpoint(ckpt_path)
    return agent


def _eval_agent(agent, env_cfg: AdaptRMABConfig,
                n_episodes: int, seed: int) -> np.ndarray:
    """Roll out a trained agent for n_episodes, return per-episode returns."""
    T = env_cfg.T
    returns = []
    for ep in range(n_episodes):
        env = AdaptRMABEnv(env_cfg, seed=seed + ep)
        obs, _ = env.reset()
        if hasattr(agent, "reset_history"):
            agent.reset_history()
        ep_return = 0.0
        for _ in range(T):
            action = agent.act_hard(obs)
            obs, reward_vec, done, _ = env.step(action)
            ep_return += float(reward_vec.sum())
            if done:
                break
        returns.append(ep_return)
    return np.array(returns, dtype=np.float32)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {
    "random":            "#d62728",
    "greedy":            "#ff9896",
    "oracle_greedy":     "#aec7e8",
    "oracle_lookahead":  "#1f77b4",
    "neurwin":           "#ff7f0e",
    "ppo":               "#8c564b",
    "dpmd":              "#2ca02c",
}
LABELS = {
    "random":           "Random",
    "greedy":           "Greedy",
    "neurwin":          "NeurWIN",
    "ppo":              "PPO",
    "dpmd":             "BIRD (ours)",
    "oracle_lookahead": "Oracle",
}


def plot_comparison(results: dict[str, np.ndarray],
                    env_cfg: AdaptRMABConfig,
                    save_path: str = "comparison.png"):
    names = list(results.keys())
    data  = [results[n] for n in names]
    colors = [COLORS.get(n, "#8c8c8c") for n in names]

    fig, ax = plt.subplots(figsize=(7, 5))

    means = [d.mean() for d in data]
    stds  = [d.std()  for d in data]
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85,
                  edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(n, n) for n in names], rotation=15, ha="right")
    ax.set_ylabel("Mean Episode Return ± Std")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{m:.1f}", ha="center", va="bottom", fontsize=8)
    if "dpmd" in names:
        bars[names.index("dpmd")].set_edgecolor("red")
        bars[names.index("dpmd")].set_linewidth(2.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {save_path}")


def save_results_npz(path: Path, results: dict[str, np.ndarray],
                     env_name: str, args: argparse.Namespace) -> None:
    names = list(results.keys())
    np.savez(
        path,
        method_names=np.asarray(names),
        mean_returns=np.asarray([results[n].mean() for n in names], dtype=np.float32),
        std_returns=np.asarray([results[n].std() for n in names], dtype=np.float32),
        env_name=np.asarray(env_name),
        N=np.asarray(args.N, dtype=np.int32),
        K=np.asarray(args.K, dtype=np.int32),
        T=np.asarray(args.T, dtype=np.int32),
        seed=np.asarray(args.seed, dtype=np.int32),
        n_episodes=np.asarray(args.n_episodes, dtype=np.int32),
        **{n: results[n] for n in names},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpmd_ckpt", type=str,
                        default="checkpoints_dpmd/best.pth")
    parser.add_argument("--nw_ckpt",   type=str,
                        default="checkpoints_neurwin/best.pth")
    parser.add_argument("--ppo_ckpt",   type=str,
                        default="checkpoints_ppo/best.pth")
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--N",          type=int, default=20)
    parser.add_argument("--K",          type=int, default=5)
    parser.add_argument("--T",          type=int, default=100)
    parser.add_argument("--out",        type=str, default="comparison.png")
    args = parser.parse_args()

    env_cfg = AdaptRMABConfig(N=args.N, K=args.K, T=args.T)
    print("=" * 60)
    print(f"AdaptRMAB  N={env_cfg.N}  K={env_cfg.K}  T={env_cfg.T}  "
          f"M={env_cfg.M}  episodes={args.n_episodes}")
    print(f"  sigma_v={env_cfg.sigma_v}  sigma_w={env_cfg.sigma_w}  "
          f"ou_rho={env_cfg.ou_rho}  sigma_beta={env_cfg.sigma_beta}")
    print("=" * 60)
    for m in range(env_cfg.M):
        ss_a = (TYPE_BETA_MEAN[m] - env_cfg.drift) / (1 - TYPE_ALPHA_MEAN[m])
        print(f"  type {m} ({TYPE_NAMES[m]:12s}): "
              f"alpha_bar={TYPE_ALPHA_MEAN[m]:.2f}  beta_bar={TYPE_BETA_MEAN[m]:.2f}  "
              f"active_SS={ss_a:.1f}")
    print()

    results: dict[str, np.ndarray] = {}

    # ── Baselines ────────────────────────────────────────────────────────
    for name, pname, policy in [
        ("random",  "random",  RandomPolicy(env_cfg, seed=args.seed)),
        ("greedy",  "greedy",  GreedyObsPolicy(env_cfg)),
    ]:
        rets = evaluate_policy(pname, policy, env_cfg,
                               args.n_episodes, seed_offset=args.seed)
        results[name] = rets
        print(f"  {name:18s}  mean={rets.mean():8.1f}  std={rets.std():.1f}")

    # ── Learned agents ───────────────────────────────────────────────────
    trained = [
        ("neurwin",          args.nw_ckpt,   _load_neurwin),
        ("ppo",          args.ppo_ckpt,   _load_ppo),
        ("dpmd",         args.dpmd_ckpt,  _load_dpmd),
    ]

    for name, ckpt_path, loader in trained:
        p = Path(ckpt_path)
        if not p.exists():
            print(f"  {name:18s}  [checkpoint not found: {ckpt_path}]")
            continue
        try:
            agent = loader(ckpt_path, env_cfg)
            if hasattr(agent, "encoder"):
                agent.encoder.eval()
            if hasattr(agent, "index_net"):
                agent.index_net.eval()
            if hasattr(agent, "actor"):
                agent.actor.eval()
            if hasattr(agent, "critic"):
                agent.critic.eval()
            rets = _eval_agent(agent, env_cfg,
                               n_episodes=args.n_episodes,
                               seed=args.seed + 1000)
            results[name] = rets
            print(f"  {name:18s}  mean={rets.mean():8.1f}  std={rets.std():.1f}")
        except Exception as e:
            print(f"  {name:18s}  [ERROR: {e}]")

    # ── Oracle last (rightmost bar) ───────────────────────────────────────
    rets = evaluate_policy("oracle_lookahead", OracleLookaheadPolicy(env_cfg),
                           env_cfg, args.n_episodes, seed_offset=args.seed)
    results["oracle_lookahead"] = rets
    print(f"  {'oracle_lookahead':18s}  mean={rets.mean():8.1f}  std={rets.std():.1f}")

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    rand_m = results["random"].mean()
    orac_m = results["oracle_lookahead"].mean()
    print("% of oracle_lookahead gap above random:")
    for name, rets in results.items():
        m   = rets.mean()
        pct = 100.0 * (m - rand_m) / (orac_m - rand_m + 1e-8)
        print(f"  {name:18s}  {m:8.1f}  ({pct:+.1f}%)")

    print()
    save_results_npz(Path(args.out).with_suffix(".npz"), results, "synthetic-drifting", args)
    plot_comparison(results, env_cfg, save_path=args.out)


if __name__ == "__main__":
    main()
