"""
Auxiliary prediction loss ablation for BIRD/DPMD on synthetic-drifting.

Compares matched retrains:
  with_aux     BIRD trained with next-observation auxiliary loss
  without_aux  BIRD trained with aux_coef=0

Outputs:
  aux_ablation_N{N}_K{K}.npz
  aux_ablation_N{N}_K{K}.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import diffusion_DPMD_train as dpmd
from env import AdaptRMABConfig, AdaptRMABEnv


METHODS = ("with_aux", "without_aux")
ALIASES = {"no_aux": "without_aux"}
LABELS = {"with_aux": "BIRD", "without_aux": "No Aux Loss"}
COLORS = {"with_aux": "#2ca02c", "without_aux": "#d62728"}


def _load_agent(ckpt_path: str, env_cfg: AdaptRMABConfig):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    saved = ckpt.get("cfg", {})
    cfg = dpmd.DPMDTrainConfig()
    cfg.N = env_cfg.N; cfg.K = env_cfg.K; cfg.T = env_cfg.T
    for key in ("z_dim", "encoder_hidden", "encoder_heads", "encoder_layers", "L",
                "actor_hidden", "actor_t_dim", "T_diff", "score_clip",
                "critic_hidden", "action_candidates", "target_action_candidates",
                "aux_coef"):
        if key in saved:
            setattr(cfg, key, saved[key])
    agent = dpmd.DPMDAgent(N=cfg.N, K=cfg.K, cfg=cfg)
    agent.load_checkpoint(ckpt_path)
    return agent


def _normalize_methods(methods: list[str]) -> list[str]:
    normalized = [ALIASES.get(m, m) for m in methods]
    unknown = sorted(set(normalized) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}. Valid methods: {list(METHODS)}")
    return normalized


def _get_trained_agent(args: argparse.Namespace, env_cfg: AdaptRMABConfig,
                       method: str, seed: int):
    aux_coef = args.with_aux_coef if method == "with_aux" else 0.0
    save_dir = Path(args.save_dir_template.format(method=method, seed=seed))
    ckpt_path = save_dir / "best.pth"
    if ckpt_path.exists() and not args.force_retrain:
        print(f"Loading {method} checkpoint: {ckpt_path}")
        return _load_agent(str(ckpt_path), env_cfg)

    print(f"Training {method} checkpoint: {ckpt_path} (aux_coef={aux_coef})")
    cfg = dpmd.DPMDTrainConfig(
        N=args.N, K=args.K, T=args.T, seed=seed, epochs=args.epochs,
        aux_coef=aux_coef, save_dir=str(save_dir),
    )
    env = AdaptRMABEnv(env_cfg, seed=seed)
    return dpmd.train(env, cfg=cfg, checkpoint_dir=cfg.save_dir)


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


def plot_results(results: dict[str, np.ndarray], out_path: str) -> None:
    names = list(results.keys())
    means = [results[n].mean() for n in names]
    stds = [results[n].std() for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=[COLORS.get(n, "#8c8c8c") for n in names],
                  edgecolor="black", linewidth=0.8)
    if "with_aux" in names:
        bars[names.index("with_aux")].set_edgecolor("red")
        bars[names.index("with_aux")].set_linewidth(2.2)
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


def save_npz(path: str, results: dict[str, np.ndarray], raw_returns: np.ndarray,
             args: argparse.Namespace) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    names = list(results.keys())
    np.savez(
        path,
        method_names=np.asarray(names),
        mean_returns=np.asarray([results[n].mean() for n in names], dtype=np.float32),
        std_returns=np.asarray([results[n].std() for n in names], dtype=np.float32),
        env_name=np.asarray("synthetic-drifting"),
        N=np.asarray(args.N, dtype=np.int32),
        K=np.asarray(args.K, dtype=np.int32),
        T=np.asarray(args.T, dtype=np.int32),
        train_seeds=np.asarray(args.train_seeds, dtype=np.int32),
        n_episodes=np.asarray(args.n_episodes, dtype=np.int32),
        raw_returns=raw_returns,
        **{n: results[n] for n in names},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--train_seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--with_aux_coef", type=float, default=0.5)
    parser.add_argument("--save_dir_template", default="checkpoints_dpmd_aux_ablation_{method}_seed{seed}")
    parser.add_argument("--force_retrain", action="store_true")
    parser.add_argument("--reuse_existing_with_aux", action="store_true")
    parser.add_argument("--with_aux_ckpt", default="checkpoints_dpmd/best.pth")
    parser.add_argument("--out_dir", default="experiment_outputs")
    args = parser.parse_args()
    args.methods = _normalize_methods(args.methods)

    env_cfg = AdaptRMABConfig(N=args.N, K=args.K, T=args.T)
    results: dict[str, np.ndarray] = {}
    raw_returns = np.zeros(
        (len(args.methods), len(args.train_seeds), args.n_episodes), dtype=np.float32)

    for method_idx, method in enumerate(args.methods):
        method_returns = []
        for seed_idx, seed in enumerate(args.train_seeds):
            print("=" * 72)
            print(f"Preparing aux ablation: {method} seed={seed}")
            print("=" * 72)
            if method == "with_aux" and args.reuse_existing_with_aux:
                ckpt_path = Path(args.with_aux_ckpt)
                if not ckpt_path.exists():
                    raise FileNotFoundError(f"with_aux checkpoint not found: {ckpt_path}")
                agent = _load_agent(str(ckpt_path), env_cfg)
            else:
                agent = _get_trained_agent(args, env_cfg, method, seed)
            returns = evaluate_agent(agent, env_cfg, args.n_episodes, seed=seed + 1000)
            raw_returns[method_idx, seed_idx] = returns
            method_returns.append(returns)
            print(f"[{method} seed={seed}] mean={returns.mean():.2f} std={returns.std():.2f}")
        results[method] = np.concatenate(method_returns)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"aux_ablation_N{args.N}_K{args.K}"
    save_npz(f"{stem}.npz", results, raw_returns, args)
    plot_results(results, f"{stem}.png")
    print(f"[saved] {stem}.npz")
    print(f"[saved] {stem}.png")


if __name__ == "__main__":
    main()
