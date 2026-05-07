"""
mlp_actor.py  (mimic-icu)

Ablation: replace diffusion actor in DPMD with a direct per-arm MLP.
Identical architecture to DPMD except the actor head — isolates the
contribution of the diffusion model on the MIMIC-ICU environment.

Architecture:
  BeliefEncoder   : identical to DPMD (Transformer over L-step 5D-obs history)
  MLPActorNet     : z_i -> score_i  (per-arm MLP, no diffusion)
  PerArmTwinCritic: identical to DPMD

Actor loss:
  Q-importance-weighted MSE — same weights as DPMD RSM loss but regresses
  scores directly toward high-Q historical scores from the replay buffer:
      actor_loss = E_w[ ||MLP(z) - a0_old||^2 ]
  where w = exp(Q(z, topK(a0_old)) / lambda)

Usage:
  python mlp_actor.py
  python mlp_actor.py --N 20 --K 5 --epochs 200
  python mlp_actor.py --ckpt_dir checkpoints_mlp_actor
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from diffusion_model import PerArmTwinCritic, soft_update_, topk_action
from diffusion_DPMD_train import (
    BeliefEncoder, ObsPredHead,
    HistoryReplayBuffer, _collect_episode, _random_action, set_seed,
)
from env import MIMICRMABConfig, MIMICRMABEnv, OBS_DIM


# ---------------------------------------------------------------------------
# MLP Actor Network: z_i -> score_i
# ---------------------------------------------------------------------------

class MLPActorNet(nn.Module):
    """Per-arm MLP: belief embedding z_i -> scalar score_i."""

    def __init__(self, z_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (batch, N, z_dim) -> scores: (batch, N)"""
        batch, N, zd = z.shape
        return self.net(z.reshape(batch * N, zd)).reshape(batch, N)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MLPActorConfig:
    seed:   int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    N: int = 20
    K: int = 5
    T: int = 100

    obs_dim: int = OBS_DIM   # 5 vitals

    # Belief encoder — identical to DPMD
    L:               int   = 80
    z_dim:           int   = 64
    encoder_hidden:  int   = 128
    encoder_heads:   int   = 4
    encoder_layers:  int   = 3

    # Auxiliary next-obs prediction loss
    aux_coef: float = 0.5

    # MLP actor
    actor_hidden: int = 64

    # Critic — identical to DPMD
    critic_hidden: int   = 64
    gamma:         float = 0.99
    tau:           float = 0.005

    # Training — matches DPMD defaults
    batch_size:           int   = 128
    replay_size:          int   = 100_000
    epochs:               int   = 200
    updates_per_epoch:    int   = 100
    start_updates_after:  int   = 2_000
    warmup_episodes:      int   = 20
    lr_actor:             float = 3e-4
    lr_critic:            float = 1e-4
    lr_critic_min:        float = 3e-5
    grad_clip:            float = 1.0

    # Q-weighting — identical to DPMD
    lambda_init:   float = 2.0
    lambda_target: float = 1.0
    lambda_beta:   float = 0.005
    lambda_min:    float = 0.5
    lambda_max:    float = 20.0
    ema_xi:        float = 0.05
    clip_exp:      float = 3.0

    policy_score_noise: float = 0.15

    save_dir:    str = "checkpoints_mlp_actor"
    save_every:  int = 25
    resume_path: str = ""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class MLPActorAgent:
    def __init__(self, N: int, K: int, cfg: MLPActorConfig):
        self.N = int(N); self.K = int(K); self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.encoder = BeliefEncoder(
            obs_dim=cfg.obs_dim, z_dim=cfg.z_dim, hidden_dim=cfg.encoder_hidden,
            n_heads=cfg.encoder_heads, n_layers=cfg.encoder_layers, L=cfg.L,
        ).to(self.device)

        self.obs_pred_head = ObsPredHead(
            z_dim=cfg.z_dim, obs_dim=cfg.obs_dim, hidden_dim=cfg.encoder_hidden,
            pred_dim=cfg.obs_dim,
        ).to(self.device)

        self.actor = MLPActorNet(
            z_dim=cfg.z_dim, hidden_dim=cfg.actor_hidden,
        ).to(self.device)

        self.critic = PerArmTwinCritic(
            n_arms=self.N, z_dim=cfg.z_dim, hidden_dim=cfg.critic_hidden,
        ).to(self.device)
        self.critic_t = PerArmTwinCritic(
            n_arms=self.N, z_dim=cfg.z_dim, hidden_dim=cfg.critic_hidden,
        ).to(self.device)
        self.critic_t.load_state_dict(self.critic.state_dict())

        self.opt_actor = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.actor.parameters()),
            lr=cfg.lr_actor)
        self.opt_critic = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.obs_pred_head.parameters())
            + list(self.critic.parameters()),
            lr=cfg.lr_critic)

        self.sched_critic = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_critic, T_max=cfg.epochs, eta_min=cfg.lr_critic_min)

        self.lambda_md = float(cfg.lambda_init)
        self.mu_q = 0.0; self.sig_q = 1.0; self.update_step = 0
        self.mu_r = 0.0; self.sig_r = 1.0

        self._obs_hist: np.ndarray | None = None
        self._act_hist: np.ndarray | None = None

    def reset_history(self):
        self._obs_hist = np.zeros((self.N, self.cfg.L, self.cfg.obs_dim), dtype=np.float32)
        self._act_hist = np.zeros((self.N, self.cfg.L), dtype=np.float32)

    def _encode(self, obs_hist: torch.Tensor, act_hist: torch.Tensor) -> torch.Tensor:
        """obs_hist (batch,N,L,obs_dim), act_hist (batch,N,L) -> z (batch,N,z_dim)"""
        return self.encoder(obs_hist, act_hist)

    @torch.no_grad()
    def select_action(self, obs_hist: np.ndarray, act_hist: np.ndarray,
                      explore: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """obs_hist: (N, L, obs_dim)  act_hist: (N, L)"""
        oh = torch.from_numpy(obs_hist[None]).float().to(self.device)
        ah = torch.from_numpy(act_hist[None]).float().to(self.device)
        z  = self._encode(oh, ah)
        scores = self.actor(z)[0]                          # (N,)
        clean  = scores.cpu().numpy().astype(np.float32)
        if explore and self.cfg.policy_score_noise > 0.0:
            scores = scores + self.cfg.policy_score_noise * torch.randn_like(scores)
        action = topk_action(scores.unsqueeze(0), self.K)[0].cpu().numpy().astype(np.int64)
        return action, clean

    @torch.no_grad()
    def act_hard(self, obs: np.ndarray, **_) -> np.ndarray:
        """obs: (N, obs_dim). Call reset_history() first."""
        if self._obs_hist is None:
            self.reset_history()
        self._obs_hist = np.roll(self._obs_hist, -1, axis=1)
        self._obs_hist[:, -1, :] = obs                     # 5D obs per arm
        action, _ = self.select_action(self._obs_hist, self._act_hist, explore=False)
        self._act_hist = np.roll(self._act_hist, -1, axis=1)
        self._act_hist[:, -1] = action.astype(np.float32)
        return action

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs_hist      = batch["obs_hist"].float()
        act_hist      = batch["act_hist"].float()
        obs_hist_next = batch["obs_hist_next"].float()
        act_hist_next = batch["act_hist_next"].float()
        action        = batch["action"].float()
        reward        = batch["reward"].float()
        done          = batch["done"].float()

        # Reward normalization
        with torch.no_grad():
            r_mean = float(reward.mean().item())
            r_std  = float(reward.std().item() + 1e-6)
            self.mu_r  = (1 - self.cfg.ema_xi) * self.mu_r  + self.cfg.ema_xi * r_mean
            self.sig_r = (1 - self.cfg.ema_xi) * self.sig_r + self.cfg.ema_xi * r_std
            reward_norm = (reward - self.mu_r) / (self.sig_r + 1e-8)

        # ── Critic update ──────────────────────────────────────────────────
        z = self._encode(obs_hist, act_hist)
        with torch.no_grad():
            z_next      = self._encode(obs_hist_next, act_hist_next)
            next_scores = self.actor(z_next)
            next_action = topk_action(next_scores, self.K).float()
            td_target   = (reward_norm
                           + self.cfg.gamma * (1.0 - done).unsqueeze(1)
                           * self.critic_t.min_q(z_next, next_action))

        q1, q2  = self.critic(z.detach(), action)
        loss_q  = F.smooth_l1_loss(q1, td_target) + F.smooth_l1_loss(q2, td_target)

        obs_next      = batch["obs_next"].float()
        pred_obs_next = self.obs_pred_head(z, action)
        loss_aux      = F.mse_loss(pred_obs_next, obs_next[..., :self.obs_pred_head.pred_dim])
        loss_critic_total = loss_q + self.cfg.aux_coef * loss_aux

        self.opt_critic.zero_grad(set_to_none=True)
        loss_critic_total.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters())
            + list(self.obs_pred_head.parameters())
            + list(self.critic.parameters()),
            self.cfg.grad_clip)
        self.opt_critic.step()

        # ── MLP actor update ───────────────────────────────────────────────
        z_actor = self._encode(obs_hist, act_hist)
        a0_old  = batch["indices"].float()
        with torch.no_grad():
            total_q = self.critic.min_q(z_actor.detach(),
                                        topk_action(a0_old, self.K).float()).sum(dim=1)
            mean_q  = float(total_q.mean().item())
            self.mu_q  = (1 - self.cfg.ema_xi) * self.mu_q  + self.cfg.ema_xi * mean_q
            self.sig_q = (1 - self.cfg.ema_xi) * self.sig_q + \
                         self.cfg.ema_xi * float(total_q.std().item() + 1e-6)
            norm_q  = (total_q - self.mu_q) / (self.sig_q + 1e-6)
            logits  = torch.clamp(norm_q / max(self.lambda_md, 1e-6),
                                  -self.cfg.clip_exp, self.cfg.clip_exp)
            weights = (torch.exp(logits) / (torch.exp(logits).mean() + 1e-8)).detach()

        current_scores = self.actor(z_actor)
        actor_loss = torch.mean(weights.unsqueeze(1) * (current_scores - a0_old) ** 2)

        self.opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.actor.parameters()),
            self.cfg.grad_clip)
        self.opt_actor.step()

        soft_update_(self.critic_t, self.critic, self.cfg.tau)
        self.lambda_md = float(np.clip(
            self.lambda_md + self.cfg.lambda_beta * (self.cfg.lambda_target - self.lambda_md),
            self.cfg.lambda_min, self.cfg.lambda_max))
        self.update_step += 1

        return {"loss_q":    float(loss_q.item()),
                "loss_actor": float(actor_loss.item()),
                "q_mean":     mean_q}

    def checkpoint_dict(self, epoch: int = 0, best_return: float = -float("inf")) -> Dict:
        return {
            "epoch": int(epoch), "best_return": float(best_return),
            "cfg":          asdict(self.cfg),
            "encoder":      self.encoder.state_dict(),
            "actor":        self.actor.state_dict(),
            "critic":       self.critic.state_dict(),
            "critic_t":     self.critic_t.state_dict(),
            "opt_actor":    self.opt_actor.state_dict(),
            "opt_critic":   self.opt_critic.state_dict(),
            "sched_critic": self.sched_critic.state_dict(),
        }

    def load_checkpoint(self, path: str | Path) -> Tuple[int, float]:
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"],   strict=False)
        self.actor.load_state_dict(ckpt["actor"],       strict=False)
        self.critic.load_state_dict(ckpt["critic"],     strict=False)
        self.critic_t.load_state_dict(ckpt["critic_t"], strict=False)
        return int(ckpt.get("epoch", 0)), float(ckpt.get("best_return", -float("inf")))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: MLPActorConfig):
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    env_cfg = MIMICRMABConfig(N=cfg.N, K=cfg.K, T=cfg.T)
    env     = MIMICRMABEnv(env_cfg, seed=cfg.seed)
    agent   = MLPActorAgent(N=cfg.N, K=cfg.K, cfg=cfg)
    device  = agent.device

    replay = HistoryReplayBuffer(
        max_size=cfg.replay_size, n_arms=cfg.N, L=cfg.L,
        obs_dim=cfg.obs_dim, device=device)

    print("=" * 60)
    print(f"MLP Actor (MIMIC-ICU)  N={cfg.N}  K={cfg.K}  T={cfg.T}  device={device}")
    print(f"epochs={cfg.epochs}  L={cfg.L}  z_dim={cfg.z_dim}  obs_dim={cfg.obs_dim}")
    print("=" * 60)

    start_epoch = 0
    best_return = -float("inf")
    if cfg.resume_path:
        start_epoch, best_return = agent.load_checkpoint(cfg.resume_path)
        print(f"[resume] epoch={start_epoch}  best={best_return:.2f}")

    csv_path = str(Path(cfg.save_dir) / "train_log.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "ep_return", "avg10_return", "loss_q", "loss_actor", "q_mean"])

    print(f"Warmup: {cfg.warmup_episodes} random episodes …")
    for _ in range(cfg.warmup_episodes):
        _collect_episode(env, agent, replay, random_policy=True)

    returns: List[float] = []
    plot_epochs: List[int]   = []
    plot_ep:     List[float] = []
    plot_avg10:  List[float] = []
    plot_lq:     List[float] = []
    plot_la:     List[float] = []

    for epoch in range(start_epoch, cfg.epochs):
        agent.encoder.train(); agent.actor.train()
        agent.obs_pred_head.train(); agent.critic.train()

        ep_return = _collect_episode(env, agent, replay, random_policy=False)

        loss_q_ep = []; loss_a_ep = []; qm_ep = []
        if replay.size >= cfg.start_updates_after:
            for _ in range(cfg.updates_per_epoch):
                batch = replay.sample(cfg.batch_size)
                stats = agent.update(batch)
                loss_q_ep.append(stats["loss_q"])
                loss_a_ep.append(stats["loss_actor"])
                qm_ep.append(stats["q_mean"])
            agent.sched_critic.step()

        returns.append(ep_return)
        avg10 = float(np.mean(returns[-10:]))
        lq = float(np.mean(loss_q_ep)) if loss_q_ep else 0.0
        la = float(np.mean(loss_a_ep)) if loss_a_ep else 0.0
        qm = float(np.mean(qm_ep))     if qm_ep     else 0.0

        plot_epochs.append(epoch + 1); plot_ep.append(ep_return)
        plot_avg10.append(avg10); plot_lq.append(lq); plot_la.append(la)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoch {epoch+1:4d}/{cfg.epochs}  "
                  f"ret={ep_return:8.2f}  avg10={avg10:8.2f}  "
                  f"Lq={lq:.4f}  La={la:.4f}  Qmean={qm:.2f}")

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, ep_return, avg10, lq, la, qm])

        if avg10 > best_return:
            best_return = avg10
            torch.save(agent.checkpoint_dict(epoch + 1, best_return),
                       str(Path(cfg.save_dir) / "best.pth"))

        if (epoch + 1) % cfg.save_every == 0:
            torch.save(agent.checkpoint_dict(epoch + 1, best_return),
                       str(Path(cfg.save_dir) / f"epoch_{epoch+1:04d}.pth"))

    torch.save(agent.checkpoint_dict(cfg.epochs, best_return),
               str(Path(cfg.save_dir) / "final.pth"))
    _plot_curve(plot_epochs, plot_ep, plot_avg10, plot_lq, plot_la, cfg)
    print(f"\nBest avg10 return: {best_return:.2f}")
    print(f"Checkpoints saved to: {cfg.save_dir}/")


def _plot_curve(epochs, ep_returns, avg10, loss_q, loss_actor, cfg):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].plot(epochs, ep_returns, alpha=0.3, label="ep_return")
    axes[0].plot(epochs, avg10, linewidth=2, label="avg10")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Return")
    axes[0].set_title(f"MLP Actor (MIMIC-ICU)  N={cfg.N}  K={cfg.K}")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, loss_actor, color="tab:orange", label="actor")
    axes[1].plot(epochs, loss_q,     color="tab:blue",   label="critic")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    out = str(Path(cfg.save_dir) / "mlp_actor_curve.png")
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
    parser.add_argument("--seed",         type=int,   default=0)
    parser.add_argument("--epochs",       type=int,   default=200)
    parser.add_argument("--L",            type=int,   default=80)
    parser.add_argument("--z_dim",        type=int,   default=64)
    parser.add_argument("--actor_hidden", type=int,   default=64)
    parser.add_argument("--lr_actor",     type=float, default=3e-4)
    parser.add_argument("--lr_critic",    type=float, default=1e-4)
    parser.add_argument("--ckpt_dir",     type=str,   default="checkpoints_mlp_actor")
    parser.add_argument("--resume_path",  type=str,   default="")
    args = parser.parse_args()

    cfg = MLPActorConfig(
        N=args.N, K=args.K, T=args.T, seed=args.seed,
        epochs=args.epochs, L=args.L, z_dim=args.z_dim,
        actor_hidden=args.actor_hidden,
        lr_actor=args.lr_actor, lr_critic=args.lr_critic,
        save_dir=args.ckpt_dir, resume_path=args.resume_path,
    )
    train(cfg)


if __name__ == "__main__":
    main()
