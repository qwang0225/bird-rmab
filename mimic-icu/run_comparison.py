"""
run_comparison.py  (mimic-icu v2)

Compare all policies on MIMIC-ICU v3 POMDP RMAB (2D latent state, 5D obs).

Policies:
  random            -- uniform random K arms
  greedy            -- treat K most-sick-looking arms by mean of first 5 vitals
  oracle_lookahead  -- knows x_true, alpha, beta, theta; H-step lookahead
  neurwin           -- BeliefEncoder + BeliefIndexNet (checkpoints_neurwin/best.pth)
  ppo               -- PPO deep RL baseline (checkpoints_ppo/best.pth)
  dpmd              -- Diffusion DPMD (ours) (checkpoints_dpmd/best.pth)

Usage:
  python run_comparison.py
  python run_comparison.py --n_episodes 100
  python run_comparison.py --skip_missing     # skip missing checkpoints silently
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from env import MIMICRMABConfig, MIMICRMABEnv, OBS_DIM
from baselines import (
    RandomPolicy, GreedyWorstObs, ActivationGreedy,
    OracleLookahead,
    NeurWINEncoderPolicy, NeurWINEncoderConfig,
    PPOPolicy,
)

SEED_OFFSET = 1000

COLORS = {
    "random":           "#d62728",
    "obs_greedy":       "#ff9896",
    "activation_greedy": "#bcbd22",
    "learned_rollout":  "#7f7f7f",
    "mlp_actor":        "#e377c2",
    "gaussian_actor":   "#9467bd",
    "ppo":              "#8c564b",
    "neurwin":          "#ff7f0e",
    "dpmd":             "#2ca02c",
    "oracle_lookahead": "#1f77b4",
}
LABELS = {
    "random":           "Random",
    "obs_greedy":       "Obs-Greedy",
    "activation_greedy": "Activation-Greedy",
    "learned_rollout":  "Learned Rollout",
    "mlp_actor":        "MLP Actor",
    "gaussian_actor":   "Gaussian Actor",
    "ppo":              "PPO",
    "neurwin":          "NeurWIN",
    "dpmd":             "BIRD (ours)",
    "oracle_lookahead": "Oracle",
}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _run_policy(env, policy, n_episodes, seed_offset, needs_info=False,
                return_timing=False):
    returns = []
    action_time = 0.0
    action_count = 0
    for ep in range(n_episodes):
        if hasattr(policy, "reset"):
            policy.reset()
        obs, info = env.reset(seed=seed_offset + ep)
        ep_ret = 0.0
        for _ in range(env.cfg.T):
            t0 = time.perf_counter()
            action = policy.act(obs, info) if needs_info else policy.act(obs)
            action_time += time.perf_counter() - t0
            action_count += 1
            obs, reward_vec, done, info = env.step(action)
            ep_ret += float(reward_vec.sum())
            if done:
                break
        returns.append(ep_ret)
    returns = np.array(returns, dtype=np.float32)
    if return_timing:
        return returns, action_time / max(action_count, 1)
    return returns


def _run_agent(env, agent, n_episodes, seed_offset, return_timing=False):
    """For agents with act_hard + reset_history interface."""
    returns = []
    action_time = 0.0
    action_count = 0
    for ep in range(n_episodes):
        if hasattr(agent, "reset_history"):
            agent.reset_history()
        obs, _ = env.reset(seed=seed_offset + ep)
        ep_ret = 0.0
        for _ in range(env.cfg.T):
            t0 = time.perf_counter()
            action = agent.act_hard(obs)
            action_time += time.perf_counter() - t0
            action_count += 1
            obs, reward_vec, done, _ = env.step(action)
            ep_ret += float(reward_vec.sum())
            if done:
                break
        returns.append(ep_ret)
    returns = np.array(returns, dtype=np.float32)
    if return_timing:
        return returns, action_time / max(action_count, 1)
    return returns


def _load_dpmd(ckpt_path, env_cfg, device):
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
    agent.encoder.eval(); agent.actor.eval(); agent.critic.eval()
    return agent


def _load_dpmd_with_critic(ckpt_path, env_cfg, device, critic_cls):
    import diffusion_DPMD_train as dpmd

    original = dpmd.PerArmTwinCritic
    try:
        dpmd.PerArmTwinCritic = critic_cls
        return _load_dpmd(ckpt_path, env_cfg, device)
    finally:
        dpmd.PerArmTwinCritic = original


def _extra_dpmd_loader(label, env_cfg, device):
    if label in {"per_arm_single", "single", "critic_single"}:
        from critic_ablation import PerArmSingleCritic
        return lambda p: _load_dpmd_with_critic(p, env_cfg, device, PerArmSingleCritic)
    if label in {"joint_twin", "joint", "critic_joint"}:
        from critic_ablation import JointTwinCritic
        return lambda p: _load_dpmd_with_critic(p, env_cfg, device, JointTwinCritic)
    return lambda p: _load_dpmd(p, env_cfg, device)


def _load_neurwin(ckpt_path, env_cfg):
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    saved = ckpt.get("cfg", {})
    cfg   = NeurWINEncoderConfig(obs_dim=env_cfg.obs_dim)
    for key in ("z_dim", "encoder_hidden", "encoder_heads", "encoder_layers",
                "L", "index_hidden"):
        if key in saved:
            setattr(cfg, key, saved[key])
    agent = NeurWINEncoderPolicy(env_cfg.N, env_cfg.K, cfg)
    agent.load_checkpoint(ckpt_path)
    agent.encoder.eval(); agent.index_net.eval()
    return agent


def _load_mlp_actor(ckpt_path, env_cfg):
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
    agent.encoder.eval(); agent.actor.eval(); agent.critic.eval()
    return agent


def _load_gaussian_actor(ckpt_path, env_cfg):
    from gaussian_actor import GaussianActorAgent, GaussianActorConfig
    ckpt = torch.load(ckpt_path, map_location="cpu")
    saved = ckpt.get("cfg", {})
    cfg = GaussianActorConfig(N=env_cfg.N, K=env_cfg.K, T=env_cfg.T, obs_dim=env_cfg.obs_dim)
    for key in ("z_dim", "encoder_hidden", "encoder_heads", "encoder_layers",
                "L", "actor_hidden", "critic_hidden", "sigma_min", "sigma_max"):
        if key in saved:
            setattr(cfg, key, saved[key])
    agent = GaussianActorAgent(N=cfg.N, K=cfg.K, cfg=cfg)
    agent.load_checkpoint(ckpt_path)
    agent.encoder.eval(); agent.actor.eval(); agent.critic.eval()
    return agent


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_comparison(results, env_cfg, save_path="comparison.png"):
    names  = list(results.keys())
    data   = [results[n] for n in names]
    colors = [COLORS.get(n, "#8c8c8c") for n in names]

    fig, ax = plt.subplots(figsize=(7, 5))

    means = [d.mean() for d in data]
    stds  = [d.std()  for d in data]
    x     = np.arange(len(names))
    bars  = ax.bar(x, means, yerr=stds, capsize=5,
                   color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(n, n) for n in names], rotation=20, ha="right")
    ax.set_ylabel("Mean Episode Return +/- Std")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5, f"{m:.1f}",
                ha="center", va="bottom", fontsize=8)
    if "dpmd" in names:
        bars[names.index("dpmd")].set_edgecolor("red")
        bars[names.index("dpmd")].set_linewidth(2.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {save_path}")


def save_results_npz(path: Path, results: dict[str, np.ndarray],
                     env_name: str, args: argparse.Namespace,
                     latency_ms: dict[str, float] | None = None) -> None:
    names = list(results.keys())
    latency_ms = latency_ms or {}
    np.savez(
        path,
        method_names=np.asarray(names),
        mean_returns=np.asarray([results[n].mean() for n in names], dtype=np.float32),
        std_returns=np.asarray([results[n].std() for n in names], dtype=np.float32),
        latency_ms_per_decision=np.asarray(
            [latency_ms.get(n, np.nan) for n in names], dtype=np.float32),
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
    parser.add_argument("--N",           type=int,  default=20)
    parser.add_argument("--K",           type=int,  default=5)
    parser.add_argument("--T",           type=int,  default=100)
    parser.add_argument("--n_episodes",  type=int,  default=100)
    parser.add_argument("--seed",        type=int,  default=42)
    parser.add_argument("--skip_missing",action="store_true") 
    parser.add_argument("--out",         default="comparison.png")
    parser.add_argument("--dpmd_ckpt",      default="checkpoints_dpmd/best.pth")
    parser.add_argument("--extra_dpmd_ckpts", nargs="*", default=[],
                        help="Additional BIRD checkpoints as label=path entries.")
    parser.add_argument("--nw_ckpt",        default="checkpoints_neurwin/best.pth")
    parser.add_argument("--ppo_ckpt",       default="checkpoints_ppo/best.pth")
    parser.add_argument("--mlp_actor_ckpt", default="checkpoints_mlp_actor/best.pth")
    parser.add_argument("--gaussian_actor_ckpt", default="checkpoints_gaussian_actor/best.pth")
    parser.add_argument("--learned_rollout_ckpt", default="checkpoints_learned_rollout/best.pth")
    parser.add_argument("--learned_rollout_H", type=int, default=None,
                        help="Override learned-rollout planning horizon at evaluation time.")
    args = parser.parse_args()

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    env_cfg = MIMICRMABConfig(N=args.N, K=args.K, T=args.T)
    env     = MIMICRMABEnv(env_cfg, seed=0)

    print("=" * 60)
    print(f"MIMIC-ICU RMAB  N={args.N}  K={args.K}  T={args.T}  "
          f"episodes={args.n_episodes}")
    print("=" * 60)

    results = {}
    latency_ms: dict[str, float] = {}

    # Non-trained baselines
    for name, policy, needs_info in [
        ("random",  RandomPolicy(args.N, args.K, seed=args.seed), False),
        ("obs_greedy", GreedyWorstObs(args.N, args.K), False),
        ("activation_greedy", ActivationGreedy(env_cfg), False),
    ]:
        print(f"  Evaluating {name} ...", end=" ", flush=True)
        rets, sec_per_decision = _run_policy(
            env, policy, args.n_episodes, SEED_OFFSET, needs_info,
            return_timing=True)
        results[name] = rets
        latency_ms[name] = 1000.0 * sec_per_decision
        print(f"mean={rets.mean():.1f}  std={rets.std():.1f}"
              f"  latency={latency_ms[name]:.3f} ms/decision")

    # Trained agents
    trained = [
        ("neurwin",          args.nw_ckpt,        lambda p: _load_neurwin(p, env_cfg)),
        ("ppo",              args.ppo_ckpt,        lambda p: PPOPolicy.load(
                                                       p, N=args.N, K=args.K,
                                                       obs_dim=OBS_DIM, device=device)),
        ("mlp_actor",        args.mlp_actor_ckpt,  lambda p: _load_mlp_actor(p, env_cfg)),
        ("gaussian_actor",   args.gaussian_actor_ckpt, lambda p: _load_gaussian_actor(p, env_cfg)),
        ("learned_rollout",  args.learned_rollout_ckpt, None),
        ("dpmd",             args.dpmd_ckpt,        lambda p: _load_dpmd(p, env_cfg, device)),
    ]
    for item in args.extra_dpmd_ckpts:
        if "=" not in item:
            raise ValueError(f"--extra_dpmd_ckpts entries must be label=path, got: {item}")
        label, ckpt_path = item.split("=", 1)
        trained.append((label, ckpt_path, _extra_dpmd_loader(label, env_cfg, device)))

    for name, ckpt_path, loader in trained:
        p = Path(ckpt_path)
        if not p.exists():
            msg = f"[checkpoint not found: {ckpt_path}]"
            if not args.skip_missing:
                print(f"  {name:20s}  {msg}")
            else:
                print(f"  [SKIP] {name} -- {ckpt_path} not found")
            continue
        try:
            print(f"  Evaluating {name} ...", end=" ", flush=True)
            if name == "learned_rollout":
                from learned_rollout import load_learned_rollout
                agent = load_learned_rollout(ckpt_path, env_cfg)
                if args.learned_rollout_H is not None:
                    agent.cfg.H = int(args.learned_rollout_H)
            else:
                agent = loader(ckpt_path)
            rets, sec_per_decision = _run_agent(
                env, agent, args.n_episodes, SEED_OFFSET, return_timing=True)
            results[name] = rets
            latency_ms[name] = 1000.0 * sec_per_decision
            print(f"mean={rets.mean():.1f}  std={rets.std():.1f}"
                  f"  latency={latency_ms[name]:.3f} ms/decision")
        except Exception as e:
            print(f"  {name:20s}  [ERROR: {e}]")

    # Oracle baseline
    for name, policy, needs_info in [
        ("oracle_lookahead", OracleLookahead(env_cfg), True),
    ]:
        print(f"  Evaluating {name} ...", end=" ", flush=True)
        rets, sec_per_decision = _run_policy(
            env, policy, args.n_episodes, SEED_OFFSET, needs_info,
            return_timing=True)
        results[name] = rets
        latency_ms[name] = 1000.0 * sec_per_decision
        print(f"mean={rets.mean():.1f}  std={rets.std():.1f}"
              f"  latency={latency_ms[name]:.3f} ms/decision")

    # Summary table
    print()
    print("=" * 60)
    rand_m = float(results.get("random", np.array([0])).mean())
    orac_m = float(results.get("oracle_lookahead",
                   np.array([rand_m + 1e-8])).mean())
    print(f"{'Policy':<22} {'Mean':>9} {'Std':>7}  {'% of oracle gap':>16}")
    print("-" * 60)
    for name, rets in results.items():
        m   = float(rets.mean())
        s   = float(rets.std())
        pct = 100.0 * (m - rand_m) / (orac_m - rand_m + 1e-8)
        print(f"  {name:<20} {m:>9.1f} {s:>7.1f}  {pct:>+15.1f}%"
              f"  {latency_ms.get(name, np.nan):>8.3f} ms/decision")
    print("=" * 60)

    if results:
        save_results_npz(Path(args.out).with_suffix(".npz"), results,
                         "mimic-icu", args, latency_ms)
        plot_comparison(results, env_cfg, save_path=args.out)


if __name__ == "__main__":
    main()
