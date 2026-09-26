"""Evaluation-only critic ablation using the main-comparison rollout setting."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import diffusion_DPMD_train as dpmd
from critic_ablation import CRITICS, LABELS, plot_results
from env import MIMICRMABConfig, MIMICRMABEnv
from run_comparison import SEED_OFFSET, _load_dpmd, _run_agent


OriginalCritic = dpmd.PerArmTwinCritic


def _load_variant(method: str, ckpt_path: Path, env_cfg: MIMICRMABConfig, device: str):
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing checkpoint for {method}: {ckpt_path}")
    dpmd.PerArmTwinCritic = CRITICS[method]
    agent = _load_dpmd(str(ckpt_path), env_cfg, device)
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
    parser.add_argument("--methods", nargs="+", default=list(CRITICS.keys()),
                        choices=list(CRITICS.keys()))
    parser.add_argument("--ckpt_template",
                        default="checkpoints_dpmd_critic_ablation_{method}_seed0/best.pth")
    parser.add_argument("--per_arm_twin_ckpt", default=None)
    parser.add_argument("--per_arm_single_ckpt", default=None)
    parser.add_argument("--joint_twin_ckpt", default=None)
    parser.add_argument("--out_dir", default="experiment_outputs/critic_eval_main")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_cfg = MIMICRMABConfig(N=args.N, K=args.K, T=args.T)
    env = MIMICRMABEnv(env_cfg, seed=0)
    results: dict[str, np.ndarray] = {}
    latency_ms: dict[str, float] = {}

    try:
        for method in args.methods:
            explicit_ckpts = {
                "per_arm_twin": args.per_arm_twin_ckpt,
                "per_arm_single": args.per_arm_single_ckpt,
                "joint_twin": args.joint_twin_ckpt,
            }
            ckpt_path = Path(explicit_ckpts[method] or args.ckpt_template.format(method=method))
            print(f"Evaluating {method} from {ckpt_path}")
            agent = _load_variant(method, ckpt_path, env_cfg, device)
            returns, sec_per_decision = _run_agent(
                env=env, agent=agent, n_episodes=args.n_episodes,
                seed_offset=SEED_OFFSET, return_timing=True)
            results[method] = returns
            latency_ms[method] = 1000.0 * sec_per_decision
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
        env_name=np.asarray("mimic-icu"),
        N=np.asarray(args.N, dtype=np.int32),
        K=np.asarray(args.K, dtype=np.int32),
        T=np.asarray(args.T, dtype=np.int32),
        seed=np.asarray(42, dtype=np.int32),
        n_episodes=np.asarray(args.n_episodes, dtype=np.int32),
        **{n: results[n] for n in names},
    )
    plot_results(results, Path(f"{stem}.png"))
    print(f"[saved] {stem}.npz")
    print(f"[saved] {stem}.png")


if __name__ == "__main__":
    main()
