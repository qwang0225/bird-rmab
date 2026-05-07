"""
train_ppo.py  (mimic-icu)

Train PPO on MIMIC-ICU POMDP RMAB.
Flattened obs (N * obs_dim) -> shared MLP -> actor + critic heads.
Budget enforced by keeping top-K arms from Bernoulli actor.
Saves checkpoint to checkpoints_ppo/best.pth.

Usage:
    python train_ppo.py
    python train_ppo.py --epochs 200 --N 20 --K 5
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from env import MIMICRMABConfig, MIMICRMABEnv, OBS_DIM
from baselines import PPOConfig, _PPONet, PPOPolicy, _topk_action


def train(cfg: PPOConfig | None = None):
    cfg = cfg or PPOConfig()
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device  = torch.device(cfg.device)
    env_cfg = MIMICRMABConfig(N=cfg.N, K=cfg.K, T=cfg.T)
    env     = MIMICRMABEnv(env_cfg, seed=cfg.seed)

    net = _PPONet(cfg.N, cfg.obs_dim, cfg.hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    os.makedirs(cfg.save_dir, exist_ok=True)

    returns = []; best_return = -float("inf")
    print(f"PPO  N={cfg.N}  K={cfg.K}  T={cfg.T}  obs_dim={cfg.obs_dim}  "
          f"epochs={cfg.epochs}  device={cfg.device}")

    for epoch in range(cfg.epochs):
        obs, _ = env.reset()
        states:  List[torch.Tensor] = []
        actions: List[torch.Tensor] = []
        logps:   List[torch.Tensor] = []
        rewards: List[float]        = []
        values:  List[float]        = []
        ep_ret = 0.0

        for _ in range(cfg.T):
            s = torch.from_numpy(obs.reshape(1, -1)).float().to(device)
            with torch.no_grad():
                logits, val = net(s)
                probs = torch.sigmoid(logits)[0]
                dist  = torch.distributions.Bernoulli(probs)
                act_t = dist.sample()
                scores = act_t.cpu().numpy()
                action = _topk_action(scores, cfg.K)
                act_t  = torch.from_numpy(action.astype(np.float32)).to(device)
                logp   = dist.log_prob(act_t).sum()

            obs_n, rew, done, _ = env.step(action)
            ep_ret += float(rew.sum())
            states.append(s); actions.append(act_t)
            logps.append(logp); rewards.append(float(rew.sum()))
            values.append(float(val.item()))
            obs = obs_n
            if done: break

        returns.append(ep_ret)
        T_ = len(rewards)

        # Discounted returns
        G, ret = [], 0.0
        for r in reversed(rewards):
            ret = r + cfg.gamma * ret; G.insert(0, ret)
        G_t   = torch.tensor(G, device=device)
        V_t   = torch.tensor(values, device=device)
        adv   = (G_t - V_t).detach()
        adv   = (adv - adv.mean()) / (adv.std() + 1e-8)
        old_logps = torch.stack(logps).detach()
        states_t  = torch.cat(states)
        acts_t    = torch.stack(actions)

        for _ in range(cfg.ppo_epochs):
            idx = np.random.permutation(T_)
            for start in range(0, T_, cfg.batch_size):
                b = idx[start:start + cfg.batch_size]
                logits_b, val_b = net(states_t[b])
                probs_b  = torch.sigmoid(logits_b)
                dist_b   = torch.distributions.Bernoulli(probs_b)
                new_logp = dist_b.log_prob(acts_t[b]).sum(dim=1)
                ratio    = torch.exp(new_logp - old_logps[b])
                adv_b    = adv[b]
                loss_pi  = -torch.min(
                    ratio * adv_b,
                    torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_b
                ).mean()
                loss_v   = F.mse_loss(val_b.squeeze(), G_t[b])
                loss     = loss_pi + 0.5 * loss_v
                opt.zero_grad(); loss.backward(); opt.step()

        avg10 = float(np.mean(returns[-10:]))
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:04d} | avg10 {avg10:.1f}")

        if avg10 > best_return:
            best_return = avg10
            torch.save({"net": net.state_dict(), "cfg": cfg.__dict__},
                       os.path.join(cfg.save_dir, "best.pth"))

    print(f"Done. Best avg10={best_return:.1f}  -> {cfg.save_dir}/best.pth")
    return PPOPolicy(net, cfg.N, cfg.K, cfg.obs_dim, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N",          type=int,   default=20)
    parser.add_argument("--K",          type=int,   default=5)
    parser.add_argument("--T",          type=int,   default=100)
    parser.add_argument("--epochs",     type=int,   default=200)
    parser.add_argument("--hidden",     type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--ppo_epochs", type=int,   default=4)
    parser.add_argument("--clip_eps",   type=float, default=0.2)
    parser.add_argument("--seed",       type=int,   default=0)
    parser.add_argument("--save_dir",   default="checkpoints_ppo")
    args = parser.parse_args()
    cfg = PPOConfig(N=args.N, K=args.K, T=args.T, epochs=args.epochs,
                    hidden=args.hidden, lr=args.lr, ppo_epochs=args.ppo_epochs,
                    clip_eps=args.clip_eps, seed=args.seed, save_dir=args.save_dir)
    train(cfg)


if __name__ == "__main__":
    main()
