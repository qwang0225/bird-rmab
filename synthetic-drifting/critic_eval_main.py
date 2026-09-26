"""Evaluation-only critic ablation using the main-comparison rollout setting."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import diffusion_DPMD_train as dpmd
from critic_ablation import CRITICS, LABELS, plot_results
from env import AdaptRMABConfig
from run_comparison import _eval_agent, _load_dpmd


OriginalCritic = dpmd.PerArmTwinCritic


def _load_variant(method: str, ckpt_path: Path, env_cfg: AdaptRMABConfig):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing checkpoint for {method}: {ckpt_path}")
    dpmd.PerArmTwinCritic = CRITICS[method]
    agent = _load_dpmd(str(ckpt_path), env_cfg)
    agent.encoder.eval()
    agent.actor.eval()
    agent.critic.eval()
    return agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", nargs="+", default=list(CRITICS.keys()),
                        choices=list(CRITICS.keys()))
    parser.add_argument("--ckpt_template",
                        default="checkpoints_dpmd_critic_ablation_{method}_seed0/best.pth")
    parser.add_argument("--out_dir", default="experiment_outputs/critic_eval_main")
    args = parser.parse_args()

    env_cfg = AdaptRMABConfig(N=args.N, K=args.K, T=args.T)
    results: dict[str, np.ndarray] = {}
    latency_ms: dict[str, float] = {}

    try:
        for method in args.methods:
            ckpt_path = Path(args.ckpt_template.format(method=method))
            print(f"Evaluating {method} from {ckpt_path}")
            agent = _load_variant(method, ckpt_path, env_cfg)

            t_action = 0.0
            n_action = 0
            original_act = agent.act_hard

            def timed_act(obs, **kwargs):
                nonlocal t_action, n_action
                t0 = time.perf_counter()
                action = original_act(obs, **kwargs)
                t_action += time.perf_counter() - t0
                n_action += 1
                return action

            agent.act_hard = timed_act
            returns = _eval_agent(
                agent, env_cfg, n_episodes=args.n_episodes,
                seed=args.seed + 1000)
            results[method] = returns
            latency_ms[method] = 1000.0 * t_action / max(n_action, 1)
            print(f"  {method:16s} mean={returns.mean():.1f} std={returns.std():.1f} "
                  f"latency={latency_ms[method]:.3f} ms/decision")
    finally:
        dpmd.PerArmTwinCritic = OriginalCritic

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"critic_eval_main_N{args.N}_K{args.K}"
    names = list(results)
    np.savez(
        f"{stem}.npz",
        method_names=np.asarray(names),
        method_labels=np.asarray([LABELS.get(n, n) for n in names]),
        mean_returns=np.asarray([results[n].mean() for n in names], dtype=np.float32),
        std_returns=np.asarray([results[n].std() for n in names], dtype=np.float32),
        latency_ms_per_decision=np.asarray([latency_ms[n] for n in names], dtype=np.float32),
        env_name=np.asarray("synthetic-drifting"),
        N=np.asarray(args.N, dtype=np.int32),
        K=np.asarray(args.K, dtype=np.int32),
        T=np.asarray(args.T, dtype=np.int32),
        seed=np.asarray(args.seed, dtype=np.int32),
        n_episodes=np.asarray(args.n_episodes, dtype=np.int32),
        **{n: results[n] for n in names},
    )
    plot_results(results, Path(f"{stem}.png"))
    print(f"[saved] {stem}.npz")
    print(f"[saved] {stem}.png")


if __name__ == "__main__":
    main()
