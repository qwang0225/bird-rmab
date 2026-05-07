"""
ppo.py  (synthetic-drifting)

PPO baseline for AdaptRMAB with hidden types + drifting dynamics.

Architecture:
  PerArmEncoder : per-arm MLP over L-step (obs, action) history -> z_i
  ActorHead     : z_i per arm -> score_i,  top-K selection (hard eval)
  CriticHead    : mean(z_i) -> V(obs)

Action sampling for budget K:
  - Softmax over arm scores gives arm selection probabilities
  - Sample K arms without replacement (Gumbel-top-K trick)
  - log π(a|s) = sum of log-probs of selected arms under softmax
  - Evaluation: deterministic top-K

Algorithm: PPO with GAE-λ advantage estimation.

Usage:
  python ppo.py                        # train with defaults
  python ppo.py --N 20 --K 5 --T 100
  python ppo.py --epochs 300 --ckpt_dir checkpoints_ppo

Interface (compatible with run_comparison.py _eval_agent):
  agent.reset_history()       # call before each eval episode
  agent.act_hard(obs)         # returns binary np.ndarray of shape (N,)
  agent.load_checkpoint(path) # restore weights
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
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

sys.path.insert(0, str(Path(__file__).parent))
from env import AdaptRMABConfig, AdaptRMABEnv


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    # Environment
    N: int   = 20
    K: int   = 5
    T: int   = 100
    seed: int = 42

    # History encoder
    L: int   = 20          # history length per arm
    arm_enc_hidden: int = 64  # MLP hidden units for per-arm encoder
    z_dim: int = 32           # arm embedding dim

    # Training
    epochs: int = 400
    episodes_per_epoch: int = 4     # rollouts collected before each PPO update
    ppo_epochs: int = 4             # gradient steps per rollout
    batch_size: int = 512           # mini-batch size for PPO updates

    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.05   # higher to prevent entropy collapse
    value_coef: float = 0.5
    grad_clip: float = 0.5
    normalize_returns: bool = True   # normalize returns to prevent critic divergence

    # Checkpoint
    save_dir: str = "checkpoints_ppo"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class PerArmEncoder(nn.Module):
    """Shared MLP: [obs_L, act_L] per arm -> z_i."""

    def __init__(self, L: int, hidden_dim: int, z_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * L, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, obs_hist: torch.Tensor,
                act_hist: torch.Tensor) -> torch.Tensor:
        """
        obs_hist: (batch, N, L)
        act_hist: (batch, N, L)
        -> z: (batch, N, z_dim)
        """
        batch, N, L = obs_hist.shape
        x = torch.cat([obs_hist, act_hist], dim=-1)       # (batch, N, 2L)
        z = self.net(x.reshape(batch * N, 2 * L))          # (batch*N, z_dim)
        return z.view(batch, N, -1)


class ActorHead(nn.Module):
    """z_i -> scalar score_i."""

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
    """mean(z_i) -> scalar value."""

    def __init__(self, z_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (batch, N, z_dim) -> V: (batch,)"""
        z_mean = z.mean(dim=1)   # (batch, z_dim) — pool across arms
        return self.net(z_mean).squeeze(-1)   # (batch,)


# ---------------------------------------------------------------------------
# Budget-constrained action sampling
# ---------------------------------------------------------------------------

def _gumbel_topk_sample(log_probs: torch.Tensor, k: int) -> torch.Tensor:
    """
    Sample K arms without replacement proportional to exp(log_probs).
    Uses the Gumbel-top-K trick.

    log_probs: (N,)  — pre-normalised log probabilities (log softmax outputs)
    Returns: (N,) long tensor with K ones.
    """
    N = log_probs.size(0)
    k = min(int(k), N)
    gumbel = -torch.empty_like(log_probs).exponential_().log()  # Gumbel noise
    perturbed = log_probs + gumbel
    idx = torch.topk(perturbed, k=k).indices
    a = torch.zeros(N, dtype=torch.long, device=log_probs.device)
    a.scatter_(0, idx, 1)
    return a


def _subset_log_prob(log_probs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """
    Approximate log π(subset | scores) ≈ Σ_{i ∈ selected} log_probs[i]
    (valid approximation for independent arm scores; commonly used in RMAB RL).

    log_probs: (batch, N)
    action:    (batch, N)  binary long/float
    Returns: (batch,)
    """
    return (log_probs * action.float()).sum(dim=-1)


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent:
    def __init__(self, N: int, K: int, cfg: PPOConfig):
        self.N   = int(N)
        self.K   = int(K)
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.encoder = PerArmEncoder(
            L=cfg.L, hidden_dim=cfg.arm_enc_hidden, z_dim=cfg.z_dim,
        ).to(self.device)
        self.actor   = ActorHead(z_dim=cfg.z_dim).to(self.device)
        self.critic  = CriticHead(z_dim=cfg.z_dim).to(self.device)

        params = (list(self.encoder.parameters()) +
                  list(self.actor.parameters()) +
                  list(self.critic.parameters()))
        self.opt = torch.optim.Adam(params, lr=cfg.lr)

        # Internal history buffers used by act_hard()
        self._obs_hist: np.ndarray | None = None
        self._act_hist: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Inference interface (compatible with _eval_agent in run_comparison)
    # ------------------------------------------------------------------

    def reset_history(self):
        self._obs_hist = np.zeros((self.N, self.cfg.L), dtype=np.float32)
        self._act_hist = np.zeros((self.N, self.cfg.L), dtype=np.float32)

    @torch.no_grad()
    def act_hard(self, obs: np.ndarray, **_) -> np.ndarray:
        """Greedy (top-K) action. Maintains internal history."""
        if self._obs_hist is None:
            self.reset_history()
        self._obs_hist = np.roll(self._obs_hist, -1, axis=1)
        self._obs_hist[:, -1] = obs.astype(np.float32)

        oh = torch.from_numpy(self._obs_hist[None]).to(self.device)  # (1,N,L)
        ah = torch.from_numpy(self._act_hist[None]).to(self.device)  # (1,N,L)
        z  = self.encoder(oh, ah)                                     # (1,N,z)
        scores = self.actor(z)[0].cpu().numpy()                       # (N,)
        idx = np.argsort(-scores)[:self.K]
        action = np.zeros(self.N, dtype=np.int32)
        action[idx] = 1

        self._act_hist = np.roll(self._act_hist, -1, axis=1)
        self._act_hist[:, -1] = action.astype(np.float32)
        return action

    # ------------------------------------------------------------------
    # Stochastic action (used during training)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act_stochastic(self, obs_hist_np: np.ndarray,
                        act_hist_np: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Sample action with Gumbel-top-K.
        Returns (binary action np array, log_prob float).
        """
        oh = torch.from_numpy(obs_hist_np[None]).to(self.device)
        ah = torch.from_numpy(act_hist_np[None]).to(self.device)
        z  = self.encoder(oh, ah)
        scores = self.actor(z)[0]              # (N,)
        log_p  = F.log_softmax(scores, dim=0)  # (N,) — log probs
        action = _gumbel_topk_sample(log_p, self.K)  # (N,)
        lp     = _subset_log_prob(log_p.unsqueeze(0),
                                   action.unsqueeze(0)).item()
        return action.cpu().numpy().astype(np.int32), lp

    @torch.no_grad()
    def value(self, obs_hist_np: np.ndarray,
               act_hist_np: np.ndarray) -> float:
        oh = torch.from_numpy(obs_hist_np[None]).to(self.device)
        ah = torch.from_numpy(act_hist_np[None]).to(self.device)
        z  = self.encoder(oh, ah)
        return self.critic(z).item()

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

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
    """Stores one epoch's worth of transitions for PPO updates."""

    def __init__(self):
        self.obs_hists: List[np.ndarray] = []  # (N, L)
        self.act_hists: List[np.ndarray] = []  # (N, L)
        self.actions:   List[np.ndarray] = []  # (N,)
        self.log_probs: List[float]      = []
        self.rewards:   List[float]      = []
        self.values:    List[float]      = []
        self.dones:     List[bool]       = []

    def add(self, obs_hist, act_hist, action, log_prob, reward, value, done):
        self.obs_hists.append(obs_hist.copy())
        self.act_hists.append(act_hist.copy())
        self.actions.append(action.copy())
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns_advantages(self, gamma: float, gae_lambda: float,
                                    last_value: float) -> tuple:
        """GAE advantage estimation. Returns arrays shaped (T,)."""
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
    last_val = agent.value(buffer.obs_hists[-1], buffer.act_hists[-1])
    returns, advantages = buffer.compute_returns_advantages(
        cfg.gamma, cfg.gae_lambda, last_val)

    # Normalise advantages
    adv = torch.tensor(advantages, dtype=torch.float32, device=device)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    # Normalise returns to prevent critic divergence
    ret_t = torch.tensor(returns, dtype=torch.float32, device=device)
    if cfg.normalize_returns:
        ret_t = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-8)
    old_lp_t   = torch.tensor(buffer.log_probs,  dtype=torch.float32, device=device)
    actions_t  = torch.tensor(np.stack(buffer.actions), dtype=torch.float32, device=device)
    oh_t       = torch.tensor(np.stack(buffer.obs_hists), dtype=torch.float32, device=device)
    ah_t       = torch.tensor(np.stack(buffer.act_hists), dtype=torch.float32, device=device)

    stats = {"loss_total": [], "loss_actor": [], "loss_critic": [], "entropy": []}

    for _ in range(cfg.ppo_epochs):
        # mini-batch shuffle
        perm = torch.randperm(T, device=device)
        for start in range(0, T, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            if len(idx) < 2:
                continue

            oh_b  = oh_t[idx]    # (B, N, L)
            ah_b  = ah_t[idx]
            a_b   = actions_t[idx]   # (B, N)
            old_lp_b = old_lp_t[idx]
            ret_b    = ret_t[idx]
            adv_b    = adv[idx]

            z = agent.encoder(oh_b, ah_b)                # (B, N, z_dim)
            scores = agent.actor(z)                       # (B, N)
            log_p  = F.log_softmax(scores, dim=-1)        # (B, N)
            new_lp = _subset_log_prob(log_p, a_b)         # (B,)

            ratio     = torch.exp(new_lp - old_lp_b)
            clip_ratio = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
            loss_actor = -torch.min(ratio * adv_b, clip_ratio * adv_b).mean()

            # Entropy bonus: H = -sum(p * log_p) averaged over arms
            probs   = F.softmax(scores, dim=-1)           # (B, N)
            entropy = -(probs * log_p).sum(dim=-1).mean() # scalar
            loss_actor = loss_actor - cfg.entropy_coef * entropy

            v_pred    = agent.critic(z)                   # (B,)
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

def collect_rollout(agent: PPOAgent, env_cfg: AdaptRMABConfig,
                    cfg: PPOConfig, seed: int) -> tuple[RolloutBuffer, float]:
    """Collect one episode of transitions."""
    env = AdaptRMABEnv(env_cfg, seed=seed)
    obs, _ = env.reset()

    obs_hist = np.zeros((cfg.N, cfg.L), dtype=np.float32)
    act_hist = np.zeros((cfg.N, cfg.L), dtype=np.float32)

    buffer = RolloutBuffer()
    ep_return = 0.0

    for _ in range(cfg.T):
        obs_hist = np.roll(obs_hist, -1, axis=1)
        obs_hist[:, -1] = obs.astype(np.float32)

        action, log_prob = agent.act_stochastic(obs_hist, act_hist)
        value            = agent.value(obs_hist, act_hist)

        obs, reward_vec, done, _ = env.step(action)
        reward = float(reward_vec.sum())
        ep_return += reward

        buffer.add(obs_hist.copy(), act_hist.copy(),
                   action, log_prob, reward, value, done)

        act_hist = np.roll(act_hist, -1, axis=1)
        act_hist[:, -1] = action.astype(np.float32)

        if done:
            break

    return buffer, ep_return


def train(cfg: PPOConfig):
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env_cfg = AdaptRMABConfig(N=cfg.N, K=cfg.K, T=cfg.T)
    agent   = PPOAgent(N=cfg.N, K=cfg.K, cfg=cfg)
    device  = agent.device

    print("=" * 60)
    print(f"PPO  N={cfg.N}  K={cfg.K}  T={cfg.T}  device={device}")
    print(f"epochs={cfg.epochs}  eps/epoch={cfg.episodes_per_epoch}"
          f"  L={cfg.L}  z_dim={cfg.z_dim}")
    print("=" * 60)

    csv_path = str(Path(cfg.save_dir) / "train_log.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "mean_return", "loss_actor", "loss_critic", "entropy"])

    best_return  = -np.inf
    return_hist  = []

    for epoch in range(1, cfg.epochs + 1):
        # ── Collect rollouts ─────────────────────────────────────────────
        agent.encoder.train()
        agent.actor.train()
        agent.critic.train()

        buffers  = []
        ep_rets  = []
        for ep in range(cfg.episodes_per_epoch):
            buf, ret = collect_rollout(
                agent, env_cfg, cfg,
                seed=cfg.seed * 10000 + epoch * cfg.episodes_per_epoch + ep)
            buffers.append(buf)
            ep_rets.append(ret)

        # Merge buffers
        merged = RolloutBuffer()
        for buf in buffers:
            for i in range(len(buf.rewards)):
                merged.add(buf.obs_hists[i], buf.act_hists[i],
                           buf.actions[i],   buf.log_probs[i],
                           buf.rewards[i],   buf.values[i],
                           buf.dones[i])

        # ── PPO update ───────────────────────────────────────────────────
        stats = ppo_update(agent, merged, cfg, device)

        mean_ret = float(np.mean(ep_rets))
        return_hist.append(mean_ret)

        # ── Logging ──────────────────────────────────────────────────────
        if epoch % 20 == 0 or epoch == 1:
            print(f"epoch {epoch:4d}/{cfg.epochs}  "
                  f"ret={mean_ret:8.2f}  "
                  f"L_actor={stats['loss_actor']:+.4f}  "
                  f"L_critic={stats['loss_critic']:.4f}  "
                  f"H={stats['entropy']:.3f}")

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, mean_ret,
                stats["loss_actor"], stats["loss_critic"], stats["entropy"]])

        # ── Checkpoint ───────────────────────────────────────────────────
        if mean_ret > best_return:
            best_return = mean_ret
            agent.save_checkpoint(str(Path(cfg.save_dir) / "best.pth"))

        if epoch % 100 == 0:
            agent.save_checkpoint(
                str(Path(cfg.save_dir) / f"epoch_{epoch:04d}.pth"))

    # ── Final save + learning curve ──────────────────────────────────────
    agent.save_checkpoint(str(Path(cfg.save_dir) / "final.pth"))
    _plot_curve(return_hist, cfg)
    print(f"\nBest mean return: {best_return:.2f}")
    print(f"Checkpoints saved to: {cfg.save_dir}/")


def _plot_curve(return_hist: list, cfg: PPOConfig):
    fig, ax = plt.subplots(figsize=(8, 4))
    window = min(20, len(return_hist))
    r = np.array(return_hist)
    sm = np.convolve(r, np.ones(window) / window, mode="valid")
    ax.plot(r, alpha=0.3, color="#2ca02c", label="raw")
    ax.plot(np.arange(window - 1, len(r)), sm,
            color="#2ca02c", linewidth=2, label=f"smooth(w={window})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Episode Return")
    ax.set_title(f"PPO  N={cfg.N}  K={cfg.K}  T={cfg.T}")
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
    parser.add_argument("--N",            type=int,   default=20)
    parser.add_argument("--K",            type=int,   default=5)
    parser.add_argument("--T",            type=int,   default=100)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--epochs",       type=int,   default=400)
    parser.add_argument("--episodes_per_epoch", type=int, default=4)
    parser.add_argument("--ppo_epochs",   type=int,   default=4)
    parser.add_argument("--batch_size",   type=int,   default=512)
    parser.add_argument("--L",            type=int,   default=20)
    parser.add_argument("--z_dim",        type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--gamma",        type=float, default=0.99)
    parser.add_argument("--clip_eps",     type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--ckpt_dir",     type=str,   default="checkpoints_ppo")
    args = parser.parse_args()

    cfg = PPOConfig(
        N=args.N, K=args.K, T=args.T, seed=args.seed,
        epochs=args.epochs,
        episodes_per_epoch=args.episodes_per_epoch,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        L=args.L, z_dim=args.z_dim,
        lr=args.lr, gamma=args.gamma,
        clip_eps=args.clip_eps, entropy_coef=args.entropy_coef,
        save_dir=args.ckpt_dir,
    )
    train(cfg)


if __name__ == "__main__":
    main()
