"""
run_ablation.py  (mimic-icu)

Actor architecture ablation: compare per-arm DPMD variants on MIMIC-ICU RMAB.

Models compared:
  mlp_actor    -- Direct per-arm MLP (no diffusion)
  dpmd_joint_N -- N-dim joint diffusion over all arms
  dpmd         -- Per-arm diffusion (PerArmDiffusionActor) — the full method

All three share the same BeliefEncoder (5D obs) and PerArmTwinCritic.
The only axis of variation is the actor architecture.

Usage:
  python run_ablation.py
  python run_ablation.py --n_episodes 100
  python run_ablation.py --dpmd_ckpt checkpoints_dpmd/best.pth
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

from env import MIMICRMABConfig, MIMICRMABEnv, OBS_DIM

SEED_OFFSET = 2000

COLORS = {
    "mlp_actor":    "#e377c2",
    "dpmd_joint_N": "#17becf",
    "dpmd":         "#2ca02c",
}
LABELS = {
    "mlp_actor":    "MLP Actor",
    "dpmd_joint_N": "DPMD Joint-N",
    "dpmd":         "BIRD (ours)",
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_dpmd(ckpt_path: str, env_cfg: MIMICRMABConfig):
    from diffusion_DPMD_train import DPMDAgent, DPMDTrainConfig
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    saved = ckpt.get("cfg", {})
    cfg   = DPMDTrainConfig(N=env_cfg.N, K=env_cfg.K, T=env_cfg.T, obs_dim=env_cfg.obs_dim)
    for key in ("z_dim", "encoder_hidden", "encoder_heads", "encoder_layers", "L",
                "actor_hidden", "actor_t_dim", "T_diff", "score_clip", "critic_hidden"):
        if key in saved:
            setattr(cfg, key, saved[key])
    agent = DPMDAgent(N=cfg.N, K=cfg.K, cfg=cfg)
    agent.load_checkpoint(ckpt_path)
    return agent


def _load_mlp_actor(ckpt_path: str, env_cfg: MIMICRMABConfig):
    from mlp_actor import MLPActorAgent, MLPActorConfig
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    saved = ckpt.get("cfg", {})
    cfg   = MLPActorConfig(N=env_cfg.N, K=env_cfg.K, T=env_cfg.T, obs_dim=env_cfg.obs_dim)
    for key in ("z_dim", "encoder_hidden", "encoder_heads", "encoder_layers",
                "L", "actor_hidden", "critic_hidden"):
        if key in saved:
            setattr(cfg, key, saved[key])
    agent = MLPActorAgent(N=cfg.N, K=cfg.K, cfg=cfg)
    agent.load_checkpoint(ckpt_path)
    return agent


def _load_joint_dpmd(ckpt_path: str, env_cfg: MIMICRMABConfig):
    from diffusion_joint_N_ablation import JointDPMDAgent
    from diffusion_DPMD_train import DPMDTrainConfig
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    saved = ckpt.get("cfg", {})
    cfg   = DPMDTrainConfig(N=env_cfg.N, K=env_cfg.K, T=env_cfg.T, obs_dim=env_cfg.obs_dim)
    for key in ("z_dim", "encoder_hidden", "encoder_heads", "encoder_layers", "L",
                "actor_hidden", "actor_t_dim", "T_diff", "score_clip",
                "critic_hidden", "action_candidates"):
        if key in saved:
            setattr(cfg, key, saved[key])
    agent = JointDPMDAgent(N=cfg.N, K=cfg.K, cfg=cfg)
    agent.load_checkpoint(ckpt_path)
    return agent


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _run_agent(env: MIMICRMABEnv, agent, n_episodes: int, seed_offset: int) -> np.ndarray:
    returns = []
    for ep in range(n_episodes):
        if hasattr(agent, "reset_history"):
            agent.reset_history()
        obs, _ = env.reset(seed=seed_offset + ep)
        ep_ret = 0.0
        for _ in range(env.cfg.T):
            action = agent.act_hard(obs)
            obs, reward_vec, done, _ = env.step(action)
            ep_ret += float(reward_vec.sum())
            if done:
                break
        returns.append(ep_ret)
    return np.array(returns, dtype=np.float32)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ablation(results: dict[str, np.ndarray],
                  env_cfg: MIMICRMABConfig,
                  save_path: str = "ablation.png",
                  pvalues: dict | None = None):
    names  = list(results.keys())
    means  = [results[n].mean() for n in names]
    stds   = [results[n].std()  for n in names]
    colors = [COLORS.get(n, "#8c8c8c") for n in names]
    labels = [LABELS.get(n, n) for n in names]

    fig, ax = plt.subplots(figsize=(6, 5))
    x    = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors, alpha=0.85,
                  edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylabel("Mean Episode Return ± Std")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{m:.1f}", ha="center", va="bottom", fontsize=9)
    if "dpmd" in names:
        bars[names.index("dpmd")].set_edgecolor("red")
        bars[names.index("dpmd")].set_linewidth(2.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpmd_ckpt",      default="checkpoints_dpmd/best.pth")
    parser.add_argument("--joint_ckpt",     default="checkpoints_dpmd_joint_N/best.pth")
    parser.add_argument("--mlp_actor_ckpt", default="checkpoints_mlp_actor/best.pth")
    parser.add_argument("--N",              type=int, default=20)
    parser.add_argument("--K",              type=int, default=5)
    parser.add_argument("--T",              type=int, default=100)
    parser.add_argument("--n_episodes",     type=int, default=100)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--out",            default="ablation.png")
    args = parser.parse_args()

    env_cfg = MIMICRMABConfig(N=args.N, K=args.K, T=args.T)
    env     = MIMICRMABEnv(env_cfg, seed=0)

    print("=" * 60)
    print(f"Actor Ablation  MIMIC-ICU  N={args.N}  K={args.K}  T={args.T}  "
          f"episodes={args.n_episodes}")
    print("=" * 60)

    variants = [
        ("mlp_actor",    args.mlp_actor_ckpt, _load_mlp_actor),
        ("dpmd_joint_N", args.joint_ckpt,     _load_joint_dpmd),
        ("dpmd",         args.dpmd_ckpt,      _load_dpmd),
    ]

    results: dict[str, np.ndarray] = {}

    for name, ckpt_path, loader in variants:
        p = Path(ckpt_path)
        if not p.exists():
            print(f"  {name:14s}  [checkpoint not found: {ckpt_path}]")
            continue
        try:
            print(f"  Evaluating {name} ...", end=" ", flush=True)
            agent = loader(ckpt_path, env_cfg)
            for attr in ("encoder", "actor", "critic"):
                if hasattr(agent, attr):
                    getattr(agent, attr).eval()
            rets = _run_agent(env, agent, args.n_episodes, SEED_OFFSET + args.seed)
            results[name] = rets
            print(f"mean={rets.mean():.1f}  std={rets.std():.1f}")
        except Exception as e:
            print(f"  {name:14s}  [ERROR: {e}]")

    if not results:
        print("No checkpoints found — train the models first.")
        return

    # ── Summary table ────────────────────────────────────────────────────
    print()
    print(f"{'Variant':<16} {'Mean':>9} {'Std':>7}")
    print("-" * 36)
    for name, rets in results.items():
        print(f"  {name:<14} {rets.mean():>9.1f} {rets.std():>7.1f}")

    pvalues: dict[str, float] = {}
    if "dpmd" in results and len(results) > 1:
        try:
            from scipy.stats import ttest_rel
            print()
            print(f"{'Comparison':<30} {'t':>7} {'p':>8} {'Cohen d':>9}  sig")
            print("-" * 60)
            dpmd = results["dpmd"]
            for name, rets in results.items():
                if name == "dpmd":
                    continue
                n = min(len(rets), len(dpmd))
                t, p = ttest_rel(dpmd[:n], rets[:n])
                pooled_std = float(np.sqrt((dpmd[:n].std() ** 2 + rets[:n].std() ** 2) / 2))
                d = float((dpmd[:n].mean() - rets[:n].mean()) / (pooled_std + 1e-8))
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                print(f"  dpmd vs {name:<20} {t:>+7.3f} {p:>8.4f} {d:>+9.3f}  {sig}")
                pvalues[name] = p
        except ImportError:
            pass

    print()
    np.savez(Path(args.out).with_suffix(".npz"),
             **{k: v for k, v in results.items()})
    plot_ablation(results, env_cfg, save_path=args.out, pvalues=pvalues)


if __name__ == "__main__":
    main()
