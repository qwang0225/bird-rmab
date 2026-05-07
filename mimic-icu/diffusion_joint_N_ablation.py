"""
diffusion_joint_N_ablation.py  (mimic-icu)

Ablation: N-dimensional JOINT diffusion actor for MIMIC-ICU POMDP RMAB.

Per-arm DPMD (baseline):  one 1D diffusion per arm, shared score_net.
Joint-N DPMD (this file): one N-dim diffusion over ALL arms jointly.

Key architectural difference:
  PerArmDiffusionActor : score_net input = (w_i scalar, z_i, t_emb)  [per-arm, shared weights]
  JointDiffusionActor  : score_net input = (w_1..w_N, z_1..z_N flat, t_emb)  [global, no sharing]

Expected result: worse than per-arm because
  1. Input dim = N + N*z_dim + t_dim  (e.g. N=20: 20+1280+32=1332) — hard to train
  2. No permutation equivariance — arm order matters
  3. Single diffusion in N-dim space vs N independent 1D diffusions

Usage:
    python diffusion_joint_N_ablation.py
    python diffusion_joint_N_ablation.py --N 20 --K 5 --epochs 400
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from diffusion_model import (
    cosine_beta_schedule, timestep_embedding,
    PerArmTwinCritic, soft_update_, topk_action,
)
from diffusion_DPMD_train import (
    BeliefEncoder, ObsPredHead, DPMDTrainConfig,
    HistoryReplayBuffer, _collect_episode,
    set_seed,
)
from env import MIMICRMABConfig, MIMICRMABEnv, OBS_DIM


# ---------------------------------------------------------------------------
# Joint N-dim score network
# ---------------------------------------------------------------------------

class JointScoreNet(nn.Module):
    """
    Score net for N-dim joint diffusion.

    Input:  [noisy_w (N), z_flat (N*z_dim), t_emb (t_dim)]  -> eps (N)

    Unlike per-arm, this sees all arms simultaneously — no weight sharing,
    no permutation equivariance.
    """

    def __init__(self, N: int, z_dim: int, t_dim: int, hidden_dim: int):
        super().__init__()
        in_dim = N + N * z_dim + t_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, N),
        )

    def forward(self, noisy_w: torch.Tensor,
                z_flat: torch.Tensor,
                t_emb: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([noisy_w, z_flat, t_emb], dim=-1))


class JointDiffusionActor(nn.Module):
    """One N-dim diffusion process over all arm scores jointly."""

    def __init__(self, N: int, z_dim: int, hidden_dim: int, t_dim: int,
                 diffusion_steps: int, device: torch.device, score_clip: float = 6.0):
        super().__init__()
        self.N = int(N); self.z_dim = int(z_dim)
        self.t_dim = int(t_dim); self.diffusion_steps = int(diffusion_steps)
        self.device = device; self.score_clip = float(score_clip)

        self.score_net = JointScoreNet(N=N, z_dim=z_dim, t_dim=t_dim, hidden_dim=hidden_dim)

        betas     = cosine_beta_schedule(self.diffusion_steps).to(device)
        alphas    = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas",     betas)
        self.register_buffer("alphas",    alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    def _z_flat(self, z: torch.Tensor) -> torch.Tensor:
        """z: (batch, N, z_dim) -> (batch, N*z_dim)"""
        return z.reshape(z.size(0), self.N * self.z_dim)

    def eps_pred(self, noisy_w: torch.Tensor,
                 z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """noisy_w: (B,N)  z: (B,N,z_dim)  t: (B,) -> eps: (B,N)"""
        t_emb = timestep_embedding(t, self.t_dim)   # (B, t_dim)
        return self.score_net(noisy_w, self._z_flat(z), t_emb)

    def q_sample(self, clean_w: torch.Tensor,
                 t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        ab = self.alpha_bar[t].unsqueeze(1)
        return torch.sqrt(ab) * clean_w + torch.sqrt(1.0 - ab) * eps

    @torch.no_grad()
    def p_sample(self, noisy_w: torch.Tensor, z: torch.Tensor, step: int) -> torch.Tensor:
        batch = noisy_w.size(0)
        t     = torch.full((batch,), step, dtype=torch.long, device=noisy_w.device)
        eps   = self.eps_pred(noisy_w, z, t)

        beta_t      = self.betas[step]
        alpha_t     = self.alphas[step]
        alpha_bar_t = self.alpha_bar[step]
        alpha_bar_p = self.alpha_bar[step - 1] if step > 0 else torch.tensor(1.0, device=noisy_w.device)

        clean_pred = (noisy_w - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t + 1e-8)
        clean_pred = torch.clamp(clean_pred, -self.score_clip, self.score_clip)

        coef1 = torch.sqrt(alpha_bar_p) * beta_t / (1.0 - alpha_bar_t + 1e-8)
        coef2 = torch.sqrt(alpha_t) * (1.0 - alpha_bar_p) / (1.0 - alpha_bar_t + 1e-8)
        mean  = coef1 * clean_pred + coef2 * noisy_w

        if step == 0:
            return mean
        var = beta_t * (1.0 - alpha_bar_p) / (1.0 - alpha_bar_t + 1e-8)
        return mean + torch.sqrt(var + 1e-8) * torch.randn_like(noisy_w)

    @torch.no_grad()
    def _sample_once(self, z: torch.Tensor) -> torch.Tensor:
        w = torch.randn((z.size(0), self.N), device=z.device)
        for step in reversed(range(self.diffusion_steps)):
            w = self.p_sample(w, z, step)
        return torch.clamp(w, -self.score_clip, self.score_clip)

    @torch.no_grad()
    def sample(self, z: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        if num_samples <= 1:
            return self._sample_once(z)
        return torch.stack([self._sample_once(z) for _ in range(num_samples)], dim=1)


# ---------------------------------------------------------------------------
# Agent (same interface as DPMDAgent but with 5D obs)
# ---------------------------------------------------------------------------

class JointDPMDAgent:
    """
    Drop-in replacement for DPMDAgent with JointDiffusionActor.
    Handles 5D observations (MIMIC-ICU): obs_hist is (N, L, obs_dim).
    """

    def __init__(self, N: int, K: int, cfg: DPMDTrainConfig):
        self.N = int(N); self.K = int(K); self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.rng = np.random.default_rng(cfg.seed)

        self.encoder = BeliefEncoder(
            obs_dim=cfg.obs_dim, z_dim=cfg.z_dim, hidden_dim=cfg.encoder_hidden,
            n_heads=cfg.encoder_heads, n_layers=cfg.encoder_layers, L=cfg.L,
        ).to(self.device)

        self.obs_pred_head = ObsPredHead(
            z_dim=cfg.z_dim, obs_dim=cfg.obs_dim, hidden_dim=cfg.encoder_hidden,
            pred_dim=cfg.obs_dim,
        ).to(self.device)

        self.actor = JointDiffusionActor(
            N=self.N, z_dim=cfg.z_dim, hidden_dim=cfg.actor_hidden,
            t_dim=cfg.actor_t_dim, diffusion_steps=cfg.T_diff,
            device=self.device, score_clip=cfg.score_clip,
        ).to(self.device)
        self.actor_t = JointDiffusionActor(
            N=self.N, z_dim=cfg.z_dim, hidden_dim=cfg.actor_hidden,
            t_dim=cfg.actor_t_dim, diffusion_steps=cfg.T_diff,
            device=self.device, score_clip=cfg.score_clip,
        ).to(self.device)
        self.actor_t.load_state_dict(self.actor.state_dict())

        self.critic = PerArmTwinCritic(
            n_arms=self.N, z_dim=cfg.z_dim, hidden_dim=cfg.critic_hidden).to(self.device)
        self.critic_t = PerArmTwinCritic(
            n_arms=self.N, z_dim=cfg.z_dim, hidden_dim=cfg.critic_hidden).to(self.device)
        self.critic_t.load_state_dict(self.critic.state_dict())

        self.opt_actor = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.actor.parameters()), lr=cfg.lr_actor)
        self.opt_critic = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.obs_pred_head.parameters())
            + list(self.critic.parameters()), lr=cfg.lr_critic)
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
        return self.encoder(obs_hist, act_hist)

    def _score_objective(self, critic, z, scores):
        return critic.min_q(z, topk_action(scores, self.K).float()).sum(dim=1)

    @torch.no_grad()
    def sample_scores(self, z, actor=None, critic=None, num_candidates=None):
        actor = actor or self.actor; critic = critic or self.critic
        num_candidates = max(1, int(num_candidates or 1))
        if num_candidates == 1:
            return actor.sample(z)
        candidates = actor.sample(z, num_samples=num_candidates)
        values = torch.stack(
            [self._score_objective(critic, z, candidates[:, i, :])
             for i in range(num_candidates)], dim=1)
        best = torch.argmax(values, dim=1)
        return candidates[torch.arange(z.size(0), device=z.device), best, :]

    @torch.no_grad()
    def select_action(self, obs_hist: np.ndarray, act_hist: np.ndarray,
                      explore: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """obs_hist: (N, L, obs_dim)  act_hist: (N, L)"""
        oh = torch.from_numpy(obs_hist[None]).float().to(self.device)   # (1, N, L, obs_dim)
        ah = torch.from_numpy(act_hist[None]).float().to(self.device)   # (1, N, L)
        z  = self._encode(oh, ah)
        scores = self.sample_scores(z, num_candidates=self.cfg.action_candidates)[0]
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
        self._obs_hist[:, -1, :] = obs
        action, _ = self.select_action(self._obs_hist, self._act_hist, explore=False)
        self._act_hist = np.roll(self._act_hist, -1, axis=1)
        self._act_hist[:, -1] = action.astype(np.float32)
        return action

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs_hist      = batch["obs_hist"].float()       # (B, N, L, obs_dim)
        act_hist      = batch["act_hist"].float()
        obs_hist_next = batch["obs_hist_next"].float()
        act_hist_next = batch["act_hist_next"].float()
        action = batch["action"].float()
        reward = batch["reward"].float()
        done   = batch["done"].float()

        with torch.no_grad():
            r_mean = float(reward.mean().item())
            r_std  = float(reward.std().item() + 1e-6)
            self.mu_r  = (1 - self.cfg.ema_xi) * self.mu_r  + self.cfg.ema_xi * r_mean
            self.sig_r = (1 - self.cfg.ema_xi) * self.sig_r + self.cfg.ema_xi * r_std
            reward_norm = (reward - self.mu_r) / (self.sig_r + 1e-8)

        # Critic update
        z = self._encode(obs_hist, act_hist)
        with torch.no_grad():
            z_next = self._encode(obs_hist_next, act_hist_next)
            next_scores = self.sample_scores(z_next, actor=self.actor_t, critic=self.critic_t,
                                             num_candidates=self.cfg.target_action_candidates)
            next_action = topk_action(next_scores, self.K).float()
            td_target = (reward_norm
                         + self.cfg.gamma * (1.0 - done).unsqueeze(1)
                         * self.critic_t.min_q(z_next, next_action))

        q1, q2 = self.critic(z.detach(), action)
        loss_q = F.smooth_l1_loss(q1, td_target) + F.smooth_l1_loss(q2, td_target)

        obs_next      = batch["obs_next"].float()                  # (B, N, obs_dim)
        pred_obs_next = self.obs_pred_head(z, action)              # (B, N, obs_dim)
        loss_aux      = F.mse_loss(pred_obs_next, obs_next)

        self.opt_critic.zero_grad(set_to_none=True)
        (loss_q + self.cfg.aux_coef * loss_aux).backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters())
            + list(self.obs_pred_head.parameters())
            + list(self.critic.parameters()), self.cfg.grad_clip)
        self.opt_critic.step()

        # Actor update
        z_act = self._encode(obs_hist, act_hist).detach()
        old_scores = batch["indices"].float()

        with torch.no_grad():
            total_q = self.critic.min_q(z_act, action).sum(dim=1)
            self.mu_q  = 0.95 * self.mu_q  + 0.05 * float(total_q.mean())
            self.sig_q = 0.95 * self.sig_q + 0.05 * float(total_q.std() + 1e-8)
            norm_q  = (total_q - self.mu_q) / (self.sig_q + 1e-8)
            weights = torch.exp(
                torch.clamp(norm_q / max(self.lambda_md, self.cfg.lambda_min),
                            -self.cfg.clip_exp, self.cfg.clip_exp))
            weights = weights / (weights.mean() + 1e-8)

        B   = obs_hist.size(0)
        t   = torch.randint(0, self.cfg.T_diff, (B,), device=self.device)
        eps = torch.randn_like(old_scores)
        noisy_scores = self.actor.q_sample(old_scores, t, eps)
        eps_pred     = self.actor.eps_pred(noisy_scores, z_act, t)
        loss_actor   = (weights * F.mse_loss(eps_pred, eps, reduction="none").mean(dim=1)).mean()

        self.opt_actor.zero_grad(set_to_none=True)
        loss_actor.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.actor.parameters()),
            self.cfg.grad_clip)
        self.opt_actor.step()

        self.lambda_md = float(np.clip(
            self.lambda_md * (1 - self.cfg.lambda_beta) + self.cfg.lambda_target * self.cfg.lambda_beta,
            self.cfg.lambda_min, self.cfg.lambda_max))

        soft_update_(self.actor_t,  self.actor,  self.cfg.tau)
        soft_update_(self.critic_t, self.critic, self.cfg.tau)
        self.update_step += 1

        return {
            "loss_q":     float(loss_q.item()),
            "loss_actor": float(loss_actor.item()),
            "loss_aux":   float(loss_aux.item()),
            "q_mean":     float(total_q.mean().item()),
            "lambda_md":  self.lambda_md,
        }

    def save_checkpoint(self, path: str):
        os.makedirs(str(Path(path).parent), exist_ok=True)
        torch.save({
            "encoder":       self.encoder.state_dict(),
            "actor":         self.actor.state_dict(),
            "critic":        self.critic.state_dict(),
            "obs_pred_head": self.obs_pred_head.state_dict(),
            "cfg":           asdict(self.cfg),
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        if "obs_pred_head" in ckpt:
            self.obs_pred_head.load_state_dict(ckpt["obs_pred_head"])


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: DPMDTrainConfig | None = None, save_dir: str = "checkpoints_dpmd_joint_N"):
    cfg = cfg or DPMDTrainConfig()
    cfg.save_dir = save_dir
    set_seed(cfg.seed)

    env_cfg = MIMICRMABConfig(N=cfg.N, K=cfg.K, T=cfg.T)
    env     = MIMICRMABEnv(env_cfg, seed=cfg.seed)
    agent   = JointDPMDAgent(N=cfg.N, K=cfg.K, cfg=cfg)
    replay  = HistoryReplayBuffer(cfg.replay_size, n_arms=cfg.N, L=cfg.L,
                                  obs_dim=cfg.obs_dim, device=agent.device)

    actor_input_dim = cfg.N + cfg.N * cfg.z_dim + cfg.actor_t_dim
    print(f"Joint-N DPMD (MIMIC-ICU)  N={cfg.N}  K={cfg.K}  T={cfg.T}  "
          f"obs_dim={cfg.obs_dim}  actor_input_dim={actor_input_dim}  device={cfg.device}")
    print(f"  (vs per-arm: input_dim={cfg.z_dim + 1 + cfg.actor_t_dim} x N independent)")

    os.makedirs(cfg.save_dir, exist_ok=True)
    csv_path = os.path.join(cfg.save_dir, "train_log.csv")
    fieldnames = ["epoch", "ep_return", "avg10", "best_return",
                  "loss_q", "loss_actor", "q_mean", "lambda_md"]
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    for _ in range(cfg.warmup_episodes):
        _collect_episode(env, agent, replay, random_policy=True)

    returns: List[float] = []; best_return = -float("inf")

    for epoch in range(cfg.epochs):
        ep_return = _collect_episode(env, agent, replay)
        returns.append(ep_return)

        losses = {"loss_q": float("nan"), "loss_actor": float("nan"), "q_mean": float("nan")}
        if replay.size >= max(cfg.start_updates_after, cfg.batch_size):
            acc = {"loss_q": 0.0, "loss_actor": 0.0, "q_mean": 0.0}
            for _ in range(cfg.updates_per_epoch):
                m = agent.update(replay.sample(cfg.batch_size))
                for k in acc:
                    acc[k] += m[k]
            losses = {k: acc[k] / cfg.updates_per_epoch for k in acc}
            agent.sched_critic.step()

        avg10 = float(np.mean(returns[-10:]))
        if avg10 > best_return:
            best_return = avg10
            agent.save_checkpoint(os.path.join(cfg.save_dir, "best.pth"))

        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch+1:4d}/{cfg.epochs}  ep={ep_return:8.1f}  avg10={avg10:8.1f}  "
                  f"best={best_return:8.1f}  lq={losses['loss_q']:.4f}  "
                  f"la={losses['loss_actor']:.4f}  qm={losses['q_mean']:.2f}  "
                  f"lam={agent.lambda_md:.3f}")

        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow({
                "epoch": epoch + 1, "ep_return": ep_return, "avg10": avg10,
                "best_return": best_return, **losses, "lambda_md": agent.lambda_md,
            })

    agent.save_checkpoint(os.path.join(cfg.save_dir, "final.pth"))
    print(f"\nBest avg10: {best_return:.2f}")
    return agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N",        type=int, default=20)
    parser.add_argument("--K",        type=int, default=5)
    parser.add_argument("--T",        type=int, default=100)
    parser.add_argument("--epochs",   type=int, default=200)
    parser.add_argument("--seed",     type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="checkpoints_dpmd_joint_N")
    args = parser.parse_args()

    cfg = DPMDTrainConfig(N=args.N, K=args.K, T=args.T,
                          epochs=args.epochs, seed=args.seed)
    train(cfg, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
