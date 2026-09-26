"""Evaluation-only DPMD timing helper for MIMIC-ICU."""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from env import MIMICRMABConfig, MIMICRMABEnv
from run_comparison import _load_dpmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--n_episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = MIMICRMABConfig(N=args.N, K=args.K, T=args.T)
    agent = _load_dpmd(args.ckpt, cfg, device)
    agent.encoder.eval()
    agent.actor.eval()
    agent.critic.eval()

    if args.warmup_steps > 0:
        env = MIMICRMABEnv(cfg, seed=args.seed - 1)
        obs, _ = env.reset()
        agent.reset_history()
        for _ in range(args.warmup_steps):
            action = agent.act_hard(obs)
            obs, _, done, _ = env.step(action)
            if done:
                break

    returns = []
    action_time = 0.0
    action_count = 0
    for ep in range(args.n_episodes):
        env = MIMICRMABEnv(cfg, seed=args.seed + ep)
        obs, _ = env.reset()
        agent.reset_history()
        ep_return = 0.0
        for _ in range(cfg.T):
            t0 = time.perf_counter()
            action = agent.act_hard(obs)
            action_time += time.perf_counter() - t0
            action_count += 1
            obs, reward_vec, done, _ = env.step(action)
            ep_return += float(reward_vec.sum())
            if done:
                break
        returns.append(ep_return)

    returns_np = np.asarray(returns, dtype=np.float32)
    print(f"episodes={args.n_episodes}")
    print(f"mean={returns_np.mean():.2f}")
    print(f"std={returns_np.std():.2f}")
    print(f"latency_ms_per_decision={1000.0 * action_time / max(action_count, 1):.3f}")


if __name__ == "__main__":
    main()
