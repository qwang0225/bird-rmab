"""
run_ablation.py  (synthetic-stationary)

Actor architecture ablation: compare per-arm DPMD variants.

Models compared:
  mlp_actor    -- Direct per-arm MLP (no diffusion)
  dpmd_joint_N -- Diffusion actor operating jointly over all N arms
  dpmd         -- Diffusion actor (PerArmDiffusionActor) — the full method

All three share the same BeliefEncoder and PerArmTwinCritic.
The only axis of variation is the actor architecture.

Usage:
  python run_ablation.py
  python run_ablation.py --n_episodes 100 --N 20 --K 5
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

sys.path.insert(0, str(Path(__file__).parent))

from env import AdaptRMABConfig
from agent_utils import load_dpmd, load_mlp_actor, eval_agent


# ---------------------------------------------------------------------------
# Joint-DPMD loader (ablation-only)
# ---------------------------------------------------------------------------

def _load_joint_dpmd(ckpt_path: str, env_cfg: AdaptRMABConfig) -> object:
    import torch
    from diffusion_joint_N_ablation import JointDPMDAgent
    from diffusion_DPMD_train import DPMDTrainConfig
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    saved = ckpt.get("cfg", {})
    cfg   = DPMDTrainConfig()
    cfg.N = env_cfg.N; cfg.K = env_cfg.K; cfg.T = env_cfg.T
    for key in ("z_dim", "encoder_hidden", "encoder_heads", "encoder_layers", "L",
                "actor_hidden", "actor_t_dim", "T_diff", "score_clip",
                "critic_hidden", "action_candidates"):
        if key in saved:
            setattr(cfg, key, saved[key])
    agent = JointDPMDAgent(N=cfg.N, K=cfg.K, cfg=cfg)
    agent.load_checkpoint(ckpt_path)
    return agent


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

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


def plot_ablation(results: dict[str, np.ndarray],
                  env_cfg: AdaptRMABConfig,
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
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
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
    parser.add_argument("--dpmd_ckpt",      type=str, default="checkpoints_dpmd/best.pth")
    parser.add_argument("--joint_ckpt",     type=str, default="checkpoints_dpmd_joint_N/best.pth")
    parser.add_argument("--mlp_actor_ckpt", type=str, default="checkpoints_mlp_actor/best.pth")
    parser.add_argument("--n_episodes",     type=int, default=100)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--N",              type=int, default=20)
    parser.add_argument("--K",              type=int, default=5)
    parser.add_argument("--T",              type=int, default=100)
    parser.add_argument("--out",            type=str, default="ablation.png")
    args = parser.parse_args()

    env_cfg = AdaptRMABConfig(N=args.N, K=args.K, T=args.T)
    print("=" * 60)
    print(f"Actor Ablation  N={env_cfg.N}  K={env_cfg.K}  T={env_cfg.T}  "
          f"episodes={args.n_episodes}")
    print("=" * 60)

    variants = [
        ("mlp_actor",    args.mlp_actor_ckpt, load_mlp_actor),
        ("dpmd_joint_N", args.joint_ckpt,     _load_joint_dpmd),
        ("dpmd",         args.dpmd_ckpt,      load_dpmd),
    ]

    results: dict[str, np.ndarray] = {}

    for name, ckpt_path, loader in variants:
        p = Path(ckpt_path)
        if not p.exists():
            print(f"  {name:14s}  [checkpoint not found: {ckpt_path}]")
            continue
        try:
            agent = loader(ckpt_path, env_cfg)
            for attr in ("encoder", "actor", "critic"):
                if hasattr(agent, attr):
                    getattr(agent, attr).eval()
            rets = eval_agent(agent, env_cfg,
                              n_episodes=args.n_episodes,
                              seed=args.seed + 1000)
            results[name] = rets
            print(f"  {name:14s}  mean={rets.mean():8.1f}  std={rets.std():.1f}")
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

    # ── Statistical significance (paired t-test + Cohen's d vs DPMD) ─────
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
