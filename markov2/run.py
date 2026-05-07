"""
run.py  —  2-state Markov RMAB benchmark.

Loads pre-trained DPMD checkpoint, then evaluates all policies.
This script never trains — run the training scripts separately:

  Train DPMD:    python diffusion_DPMD_train.py
  Train NeurWIN: python neurwin.py

Policies compared:
  random       — uniform random K activations
  greedy       — rescue K arms in state 0 first
  true_whittle — exact Whittle indices from true parameters (ceiling)
  neurwin      — NeurWIN oracle (loaded from checkpoint, skipped if missing)
  dpmd         — Diffusion DPMD (loaded from checkpoint, skipped if missing)

Usage:
  python run.py
  python run.py --N 50 --K 10
  python run.py --ckpt_dir checkpoints_dpmd --neurwin_ckpt_dir checkpoints_neurwin
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from env import MarkovRMABConfig, MarkovRMABEnv
from whittle import whittle_batch, OracleWhittlePolicy
from baselines import RandomPolicy, GreedyPolicy, WIQLOraclePolicy, evaluate_policy
from neurwin import load_neurwin
from diffusion_DPMD_train import DPMDAgent, DPMDTrainConfig
from ppo import PPOAgent, PPOConfig


# ---------------------------------------------------------------------------
# DPMD evaluation wrapper (matches evaluate_policy interface)
# ---------------------------------------------------------------------------

class DPMDEvalPolicy:
    """Wraps DPMDAgent for use with evaluate_policy()."""
    def __init__(self, agent: DPMDAgent):
        self.agent = agent

    def reset(self, **_):
        self.agent.reset_history()

    def act(self, state: np.ndarray, **_) -> np.ndarray:
        return self.agent.act_hard(state.astype(np.float32))


# ---------------------------------------------------------------------------
# Score extraction utilities
# ---------------------------------------------------------------------------

def _collect_scores_dpmd(agent: DPMDAgent, env: MarkovRMABEnv,
                          env_cfg: MarkovRMABConfig,
                          n_warmup: int = 10) -> np.ndarray:
    """Query DPMD scores for s=0 and s=1.  Shape: (N, 2)."""
    scores = np.zeros((env_cfg.N, 2))

    if agent.cfg.use_oracle_encoder:
        for s in range(2):
            state = np.full(env_cfg.N, float(s), dtype=np.float32)
            _, clean = agent.select_action(state, explore=False)
            scores[:, s] = clean
    else:
        warmup_pol = DPMDEvalPolicy(agent)
        evaluate_policy(warmup_pol, env_cfg, n_episodes=n_warmup, seed_offset=999)
        for s in range(2):
            state = np.full(env_cfg.N, float(s), dtype=np.float32)
            oh = np.roll(agent._obs_hist, -1, axis=1).copy()
            oh[:, -1] = state
            ah = agent._act_hist.copy()
            _, clean = agent.select_action(state, oh, ah, explore=False)
            scores[:, s] = clean

    return scores


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {
    "random":       "#d62728",
    "greedy":       "#ff9896",
    "wiql":         "#8c564b",
    "neurwin":      "#ff7f0e",
    "ppo":          "#9467bd",
    "dpmd":         "#2ca02c",
    "true_whittle": "#1f77b4",
}
LABELS = {
    "random":       "Random",
    "greedy":       "Greedy",
    # "wiql":         "WIQL",
    "neurwin":      "NeurWIN",
    "ppo":          "PPO",
    "dpmd":         "DPMD",
    "true_whittle": "True Whittle",
}


def plot_results(results, cfg=None, save_path="comparison.png"):
    names = list(results.keys())
    fig, ax = plt.subplots(figsize=(7, 5))

    means = [results[n].mean() for n in names]
    stds  = [results[n].std()  for n in names]
    x     = np.arange(len(names))
    bars  = ax.bar(x, means, yerr=stds, capsize=5,
                   color=[COLORS.get(n, "gray") for n in names],
                   alpha=0.85, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(n, n) for n in names], rotation=15, ha="right")
    ax.set_ylabel("Mean Episode Return ± Std")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{m:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot] {save_path}")


def plot_score_correlation(W_true, scores_dict, cfg,
                            save_path="score_correlation.png"):
    """Scatter: DPMD learned score vs true Whittle index. N*2 points total."""
    from scipy.stats import spearmanr

    n_methods = len(scores_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4))
    if n_methods == 1:
        axes = [axes]

    w_flat = W_true.flatten()

    for ax, (name, scores) in zip(axes, scores_dict.items()):
        s_flat     = scores.flatten()
        pearson_r  = float(np.corrcoef(w_flat, s_flat)[0, 1])
        spearman_r = float(spearmanr(w_flat, s_flat).statistic)

        for si, color, marker in [(0, "#1f77b4", "o"), (1, "#ff7f0e", "s")]:
            idx = slice(0, cfg.N) if si == 0 else slice(cfg.N, 2 * cfg.N)
            ax.scatter(w_flat[idx], s_flat[idx],
                       alpha=0.55, s=20, color=color,
                       marker=marker, label=f"s={si}")

        coef = np.polyfit(w_flat, s_flat, 1)
        xr   = np.linspace(w_flat.min(), w_flat.max(), 100)
        ax.plot(xr, np.polyval(coef, xr), "k--", linewidth=1.2, alpha=0.6)

        ax.set_xlabel("True Whittle Index")
        ax.set_ylabel("DPMD Score")
        ax.set_title(f"{LABELS.get(name, name)}\n"
                     f"Pearson r={pearson_r:.3f}  Spearman ρ={spearman_r:.3f}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"DPMD Score vs True Whittle  N={cfg.N}  K={cfg.K}", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot] {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N",             type=int,   default=50)
    parser.add_argument("--K",             type=int,   default=10)
    parser.add_argument("--T",             type=int,   default=100)
    parser.add_argument("--n_eval",        type=int,   default=50,
                        help="episodes for final policy comparison")
    parser.add_argument("--gamma",         type=float, default=0.99)
    parser.add_argument("--seed",          type=int,   default=0)
    parser.add_argument("--ckpt_dir",          type=str, default="checkpoints_dpmd",
                        help="directory containing DPMD best.pth checkpoint")
    parser.add_argument("--neurwin_ckpt_dir",  type=str, default="checkpoints_neurwin",
                        help="directory containing NeurWIN best.pth checkpoint")
    parser.add_argument("--ppo_ckpt_dir",      type=str, default="checkpoints_ppo",
                        help="directory containing PPO best.pth checkpoint")
    parser.add_argument("--out_dir",       type=str,   default=".")
    parser.add_argument("--score_warmup",  type=int,   default=10,
                        help="warmup episodes for DPMD score collection (belief encoder only)")
    parser.add_argument("--no_score_plot", action="store_true",
                        help="skip score correlation plot")
    args = parser.parse_args()

    env_cfg = MarkovRMABConfig(N=args.N, K=args.K, T=args.T, seed_params=args.seed)
    env     = MarkovRMABEnv(env_cfg, seed=args.seed)

    print("=" * 60)
    print(f"2-State Markov RMAB  N={args.N}  K={args.K}  T={args.T}  seed={args.seed}")
    print("=" * 60)
    env.arm_summary(); print()

    # ── Oracle Whittle indices (ground truth) ────────────────────────────
    t0 = time.time()
    W = whittle_batch(env.q0, env.q1, env.p0, env.p1, gamma=args.gamma, verbose=True)
    print(f"  Whittle done: {time.time()-t0:.1f}s")
    print(f"  W(s=0) mean={W[:,0].mean():.3f}  W(s=1) mean={W[:,1].mean():.3f}\n")

    transitions = np.stack([env.q0, env.q1, env.p0, env.p1], axis=1).astype(np.float32)

    # ── Load DPMD checkpoint (never retrain from run.py) ─────────────────
    best_ckpt = Path(args.ckpt_dir) / "best.pth"
    agent = None
    if best_ckpt.exists():
        print(f"[run] loading DPMD checkpoint: {best_ckpt}")
        cfg_dpmd = DPMDTrainConfig(N=args.N, K=args.K, T=args.T,
                                   save_dir=args.ckpt_dir, seed=args.seed)
        agent = DPMDAgent(args.N, args.K, cfg_dpmd,
                          transitions=transitions if cfg_dpmd.use_oracle_encoder else None)
        agent.load_checkpoint(str(best_ckpt), load_optimizers=False)
        agent.encoder.eval(); agent.actor.eval(); agent.critic.eval()
    else:
        print(f"[run] no DPMD checkpoint at {best_ckpt} — skipping DPMD. "
              f"Run diffusion_DPMD_train.py first to train.")

    # ── Load NeurWIN checkpoint (never retrain from run.py) ──────────────
    neurwin_ckpt = Path(args.neurwin_ckpt_dir) / "best.pth"
    neurwin_pol  = None
    if neurwin_ckpt.exists():
        print(f"[run] loading NeurWIN checkpoint: {neurwin_ckpt}")
        neurwin_pol = load_neurwin(str(neurwin_ckpt), env)
    else:
        print(f"[run] no NeurWIN checkpoint at {neurwin_ckpt} — skipping NeurWIN. "
              f"Run neurwin.py first to train.")

    # ── Load PPO checkpoint (never retrain from run.py) ──────────────────
    ppo_ckpt = Path(args.ppo_ckpt_dir) / "best.pth"
    ppo_agent = None
    if ppo_ckpt.exists():
        print(f"[run] loading PPO checkpoint: {ppo_ckpt}")
        cfg_ppo = PPOConfig(N=args.N, K=args.K, T=args.T, seed=args.seed,
                            save_dir=args.ppo_ckpt_dir)
        ppo_agent = PPOAgent(N=args.N, K=args.K, cfg=cfg_ppo,
                             transitions=transitions)
        ppo_agent.load_checkpoint(str(ppo_ckpt))
        ppo_agent.encoder.eval(); ppo_agent.actor.eval(); ppo_agent.critic.eval()
    else:
        print(f"[run] no PPO checkpoint at {ppo_ckpt} — skipping PPO. "
              f"Run ppo.py first to train.")

    # ── Build all eval policies ───────────────────────────────────────────
    oracle_pol   = OracleWhittlePolicy.__new__(OracleWhittlePolicy)
    oracle_pol.K = args.K
    oracle_pol.W = W

    policies = {
        "random": RandomPolicy(args.N, args.K, seed=args.seed + 99),
        "greedy": GreedyPolicy(args.N, args.K),
        "wiql":   WIQLOraclePolicy(args.N, args.K, env.q0, env.q1, env.p0, env.p1, gamma=args.gamma),
    }
    if neurwin_pol is not None:
        policies["neurwin"] = neurwin_pol
    if ppo_agent is not None:
        policies["ppo"] = DPMDEvalPolicy(ppo_agent)
    if agent is not None:
        policies["dpmd"] = DPMDEvalPolicy(agent)
    policies["true_whittle"] = oracle_pol

    # ── Evaluate ─────────────────────────────────────────────────────────
    print(f"\nEvaluating {args.n_eval} episodes each …")
    results = {}
    for name, pol in policies.items():
        t0   = time.time()
        rets = evaluate_policy(pol, env_cfg, n_episodes=args.n_eval,
                               seed_offset=args.seed * 1000 + 500)
        print(f"  {name:13s}  mean={rets.mean():7.2f}  std={rets.std():5.2f}"
              f"  ({time.time()-t0:.1f}s)")
        results[name] = rets

    print()
    rand_c = results["random"].mean()
    best_c = max(rets.mean() for rets in results.values())
    print("% of best gap above random:")
    for name, rets in results.items():
        m   = rets.mean()
        pct = 100.0 * (m - rand_c) / (best_c - rand_c + 1e-8)
        print(f"  {name:13s}  {m:.2f}  ({pct:.1f}%)")

    # # ── Score correlation plot (DPMD vs True Whittle only) ───────────────
    # if not args.no_score_plot and agent is not None:
    #     print("\nCollecting scores for correlation analysis …")
    #     dpmd_scores = _collect_scores_dpmd(agent, env, env_cfg,
    #                                        n_warmup=args.score_warmup)
    #     scores_dict = {"dpmd": dpmd_scores}

    #     out = Path(args.out_dir)
    #     plot_score_correlation(W, scores_dict, env_cfg,
    #                            save_path=str(out / f"score_corr_{args.N}_{args.K}.png"))

    #     from scipy.stats import spearmanr
    #     w_flat = W.flatten()
    #     pr = float(np.corrcoef(w_flat, dpmd_scores.flatten())[0, 1])
    #     sr = float(spearmanr(w_flat, dpmd_scores.flatten()).statistic)
    #     print(f"\nScore correlation (DPMD vs True Whittle, all arms × states):")
    #     print(f"  Pearson r={pr:.4f}  Spearman ρ={sr:.4f}")

    # ── Save + plot ───────────────────────────────────────────────────────
    out = Path(args.out_dir)
    np.savez(out / f"results_{args.N}_{args.K}.npz",
             **{k: v for k, v in results.items()})
    plot_results(results, env_cfg,
                 save_path=str(out / f"comparison_{args.N}_{args.K}.png"))


if __name__ == "__main__":
    main()
