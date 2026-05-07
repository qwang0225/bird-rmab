"""
ppo.py  (markov2)

PPO baseline for 2-state Markov RMAB with access to true transition dynamics.

Architecture:
  OracleEncoder : per-arm MLP over (s_i, q0_i, q1_i, p0_i, p1_i) -> z_i
  ActorHead     : z_i -> score_i,  top-K hard selection at eval
  CriticHead    : mean(z_i) -> V(obs)

No history needed — arm identity is fully encoded by known dynamics + current state.

Usage:
  python ppo.py
  python ppo.py --N 50 --K 10 --epochs 300
  python ppo.py --ckpt_dir checkpoints_ppo
"""
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env import MarkovRMABConfig, MarkovRMABEnv


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    N: int   = 50
    K: int   = 10
    T: int   = 100
    seed: int = 0

    z_dim:          int   = 64
    encoder_hidden: int   = 64

    epochs:              int   = 300
    episodes_per_epoch:  int   = 4
    ppo_epochs:          int   = 4
    batch_size:          int   = 512

    lr:              float = 3e-4
    gamma:           float = 0.99
    gae_lambda:      float = 0.95
    clip_eps:        float = 0.2
    entropy_coef:    float = 0.05
    value_coef:      float = 0.5
    grad_clip:       float = 0.5
    normalize_returns: bool = True

    save_dir: str = "checkpoints_ppo"
    device:   str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class OracleEncoder(nn.Module):
    """Per-arm MLP: (s_i, q0_i, q1_i, p0_i, p1_i) -> z_i."""

    def __init__(self, z_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, obs: torch.Tensor, transitions: torch.Tensor) -> torch.Tensor:
        """
        obs:         (batch, N)   binary state
        transitions: (N, 4)       [q0, q1, p0, p1] per arm (fixed)
        -> z:        (batch, N, z_dim)
        """
        batch, N = obs.shape
        t = transitions.unsqueeze(0).expand(batch, -1, -1)   # (batch, N, 4)
        x = torch.cat([obs.unsqueeze(-1), t], dim=-1)         # (batch, N, 5)
        return self.net(x.reshape(batch * N, 5)).reshape(batch, N, -1)


class ActorHead(nn.Module):
    def __init__(self, z_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (batch, N, z_dim) -> scores: (batch, N)"""
        batch, N, zd = z.shape
        return self.net(z.reshape(batch * N, zd)).view(batch, N)


class CriticHead(nn.Module):
    def __init__(self, z_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (batch, N, z_dim) -> V: (batch,)"""
        return self.net(z.mean(dim=1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Action sampling
# ---------------------------------------------------------------------------

def _gumbel_topk(log_probs: torch.Tensor, k: int) -> torch.Tensor:
    gumbel = -torch.empty_like(log_probs).exponential_().log()
    idx = torch.topk(log_probs + gumbel, k=k).indices
    a = torch.zeros_like(log_probs, dtype=torch.long)
    a.scatter_(0, idx, 1)
    return a


def _subset_log_prob(log_probs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """log_probs: (batch, N),  action: (batch, N) -> (batch,)"""
    return (log_probs * action.float()).sum(dim=-1)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class PPOAgent:
    def __init__(self, N: int, K: int, cfg: PPOConfig,
                 transitions: np.ndarray):
        """transitions: (N, 4) float32 array [q0, q1, p0, p1]."""
        self.N   = N
        self.K   = K
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.encoder = OracleEncoder(cfg.z_dim, cfg.encoder_hidden).to(self.device)
        self.actor   = ActorHead(cfg.z_dim).to(self.device)
        self.critic  = CriticHead(cfg.z_dim).to(self.device)

        self._trans = torch.tensor(transitions, dtype=torch.float32,
                                   device=self.device)  # (N, 4)

        params = (list(self.encoder.parameters()) +
                  list(self.actor.parameters()) +
                  list(self.critic.parameters()))
        self.opt = torch.optim.Adam(params, lr=cfg.lr)

    def reset_history(self):
        pass  # no history needed

    @torch.no_grad()
    def act_hard(self, obs: np.ndarray, **_) -> np.ndarray:
        o = torch.tensor(obs[None], dtype=torch.float32, device=self.device)
        z = self.encoder(o, self._trans)
        scores = self.actor(z)[0].cpu().numpy()
        a = np.zeros(self.N, dtype=np.int32)
        a[np.argsort(-scores)[:self.K]] = 1
        return a

    @torch.no_grad()
    def act_stochastic(self, obs: np.ndarray) -> tuple[np.ndarray, float]:
        o = torch.tensor(obs[None], dtype=torch.float32, device=self.device)
        z = self.encoder(o, self._trans)
        scores = self.actor(z)[0]
        log_p  = F.log_softmax(scores, dim=0)
        action = _gumbel_topk(log_p, self.K)
        lp = _subset_log_prob(log_p.unsqueeze(0), action.unsqueeze(0)).item()
        return action.cpu().numpy().astype(np.int32), lp

    @torch.no_grad()
    def value(self, obs: np.ndarray) -> float:
        o = torch.tensor(obs[None], dtype=torch.float32, device=self.device)
        z = self.encoder(o, self._trans)
        return self.critic(z).item()

    def save_checkpoint(self, path: str):
        os.makedirs(str(Path(path).parent), exist_ok=True)
        torch.save({
            "encoder": self.encoder.state_dict(),
            "actor":   self.actor.state_dict(),
            "critic":  self.critic.state_dict(),
            "cfg":     asdict(self.cfg),
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    def __init__(self):
        self.obs:       List[np.ndarray] = []
        self.actions:   List[np.ndarray] = []
        self.log_probs: List[float]      = []
        self.rewards:   List[float]      = []
        self.values:    List[float]      = []
        self.dones:     List[bool]       = []

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs.copy())
        self.actions.append(action.copy())
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns_advantages(self, gamma, gae_lambda, last_value):
        T = len(self.rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        values = self.values + [last_value]
        for t in reversed(range(T)):
            delta = (self.rewards[t]
                     + gamma * values[t + 1] * (1 - float(self.dones[t]))
                     - values[t])
            gae = delta + gamma * gae_lambda * (1 - float(self.dones[t])) * gae
            advantages[t] = gae
        returns = advantages + np.array(self.values, dtype=np.float32)
        return returns, advantages


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(agent: PPOAgent, buffer: RolloutBuffer,
               cfg: PPOConfig, device: torch.device) -> dict:
    T = len(buffer.rewards)
    last_val = agent.value(buffer.obs[-1])
    returns, advantages = buffer.compute_returns_advantages(
        cfg.gamma, cfg.gae_lambda, last_val)

    adv_t   = torch.tensor(advantages, dtype=torch.float32, device=device)
    adv_t   = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    ret_t   = torch.tensor(returns,    dtype=torch.float32, device=device)
    if cfg.normalize_returns:
        ret_t = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-8)
    old_lp_t  = torch.tensor(buffer.log_probs, dtype=torch.float32, device=device)
    actions_t = torch.tensor(np.stack(buffer.actions), dtype=torch.float32, device=device)
    obs_t     = torch.tensor(np.stack(buffer.obs),     dtype=torch.float32, device=device)

    stats = {"loss_total": [], "loss_actor": [], "loss_critic": [], "entropy": []}

    for _ in range(cfg.ppo_epochs):
        perm = torch.randperm(T, device=device)
        for start in range(0, T, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            if len(idx) < 2:
                continue

            o_b      = obs_t[idx]
            a_b      = actions_t[idx]
            old_lp_b = old_lp_t[idx]
            ret_b    = ret_t[idx]
            adv_b    = adv_t[idx]

            z      = agent.encoder(o_b, agent._trans)
            scores = agent.actor(z)
            log_p  = F.log_softmax(scores, dim=-1)
            new_lp = _subset_log_prob(log_p, a_b)

            ratio      = torch.exp(new_lp - old_lp_b)
            clip_ratio = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
            loss_actor = -torch.min(ratio * adv_b, clip_ratio * adv_b).mean()

            probs   = F.softmax(scores, dim=-1)
            entropy = -(probs * log_p).sum(dim=-1).mean()
            loss_actor = loss_actor - cfg.entropy_coef * entropy

            v_pred      = agent.critic(z)
            loss_critic = F.mse_loss(v_pred, ret_b)

            loss = loss_actor + cfg.value_coef * loss_critic
            agent.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(agent.encoder.parameters()) +
                list(agent.actor.parameters()) +
                list(agent.critic.parameters()),
                cfg.grad_clip)
            agent.opt.step()

            stats["loss_total"].append(loss.item())
            stats["loss_actor"].append(loss_actor.item())
            stats["loss_critic"].append(loss_critic.item())
            stats["entropy"].append(entropy.item())

    return {k: float(np.mean(v)) for k, v in stats.items()}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def collect_rollout(agent: PPOAgent, env: MarkovRMABEnv,
                    cfg: PPOConfig, seed: int) -> tuple[RolloutBuffer, float]:
    obs, _ = env.reset(seed=seed)
    buffer = RolloutBuffer()
    ep_return = 0.0

    for _ in range(cfg.T):
        action, log_prob = agent.act_stochastic(obs.astype(np.float32))
        value            = agent.value(obs.astype(np.float32))
        obs, reward_vec, done, _ = env.step(action)
        reward = float(reward_vec.sum())
        ep_return += reward
        buffer.add(obs.astype(np.float32), action, log_prob, reward, value, done)
        if done:
            break

    return buffer, ep_return


def train(cfg: PPOConfig):
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env_cfg = MarkovRMABConfig(N=cfg.N, K=cfg.K, T=cfg.T, seed_params=cfg.seed)
    env     = MarkovRMABEnv(env_cfg, seed=cfg.seed)
    transitions = np.stack([env.q0, env.q1, env.p0, env.p1], axis=1).astype(np.float32)

    agent  = PPOAgent(N=cfg.N, K=cfg.K, cfg=cfg, transitions=transitions)
    device = agent.device

    print("=" * 60)
    print(f"PPO (oracle dynamics)  N={cfg.N}  K={cfg.K}  T={cfg.T}  device={device}")
    print(f"epochs={cfg.epochs}  eps/epoch={cfg.episodes_per_epoch}")
    print("=" * 60)

    csv_path = str(Path(cfg.save_dir) / "train_log.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "mean_return", "loss_actor", "loss_critic", "entropy"])

    best_return = -np.inf
    return_hist = []

    for epoch in range(1, cfg.epochs + 1):
        agent.encoder.train(); agent.actor.train(); agent.critic.train()

        buffers = []
        ep_rets = []
        for ep in range(cfg.episodes_per_epoch):
            buf, ret = collect_rollout(
                agent, env, cfg,
                seed=cfg.seed * 10000 + epoch * cfg.episodes_per_epoch + ep)
            buffers.append(buf)
            ep_rets.append(ret)

        merged = RolloutBuffer()
        for buf in buffers:
            for i in range(len(buf.rewards)):
                merged.add(buf.obs[i], buf.actions[i], buf.log_probs[i],
                           buf.rewards[i], buf.values[i], buf.dones[i])

        stats    = ppo_update(agent, merged, cfg, device)
        mean_ret = float(np.mean(ep_rets))
        return_hist.append(mean_ret)

        if epoch % 20 == 0 or epoch == 1:
            print(f"epoch {epoch:4d}/{cfg.epochs}  ret={mean_ret:8.2f}  "
                  f"L_actor={stats['loss_actor']:+.4f}  "
                  f"L_critic={stats['loss_critic']:.4f}  "
                  f"H={stats['entropy']:.3f}")

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, mean_ret,
                stats["loss_actor"], stats["loss_critic"], stats["entropy"]])

        if mean_ret > best_return:
            best_return = mean_ret
            agent.save_checkpoint(str(Path(cfg.save_dir) / "best.pth"))

    agent.save_checkpoint(str(Path(cfg.save_dir) / "final.pth"))
    _plot_curve(return_hist, cfg)
    print(f"\nBest mean return: {best_return:.2f}")
    print(f"Checkpoints saved to: {cfg.save_dir}/")


def _plot_curve(return_hist, cfg):
    fig, ax = plt.subplots(figsize=(8, 4))
    r  = np.array(return_hist)
    w  = min(20, len(r))
    sm = np.convolve(r, np.ones(w) / w, mode="valid")
    ax.plot(r, alpha=0.3, color="#8c564b", label="raw")
    ax.plot(np.arange(w - 1, len(r)), sm, color="#8c564b",
            linewidth=2, label=f"smooth(w={w})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean Episode Return")
    ax.set_title(f"PPO (oracle dynamics)  N={cfg.N}  K={cfg.K}  T={cfg.T}")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = str(Path(cfg.save_dir) / "ppo_curve.png")
    plt.savefig(out, dpi=150); plt.close(fig)
    print(f"[plot] {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N",                   type=int,   default=50)
    parser.add_argument("--K",                   type=int,   default=10)
    parser.add_argument("--T",                   type=int,   default=100)
    parser.add_argument("--seed",                type=int,   default=0)
    parser.add_argument("--epochs",              type=int,   default=300)
    parser.add_argument("--episodes_per_epoch",  type=int,   default=4)
    parser.add_argument("--ppo_epochs",          type=int,   default=4)
    parser.add_argument("--batch_size",          type=int,   default=512)
    parser.add_argument("--z_dim",               type=int,   default=64)
    parser.add_argument("--lr",                  type=float, default=3e-4)
    parser.add_argument("--clip_eps",            type=float, default=0.2)
    parser.add_argument("--entropy_coef",        type=float, default=0.05)
    parser.add_argument("--ckpt_dir",            type=str,   default="checkpoints_ppo")
    args = parser.parse_args()

    cfg = PPOConfig(
        N=args.N, K=args.K, T=args.T, seed=args.seed,
        epochs=args.epochs,
        episodes_per_epoch=args.episodes_per_epoch,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        z_dim=args.z_dim,
        lr=args.lr, clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        save_dir=args.ckpt_dir,
    )
    train(cfg)


if __name__ == "__main__":
    main()
