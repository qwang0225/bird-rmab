"""
diffusion_DPMD_train.py  (mimic-icu)

Diffusion Policy Mirror Descent on MIMIC-ICU POMDP RMAB.

Key difference from the synthetic environments: observations are 5D
(HR, SBP, MBP, SpO2, RespRate)
instead of scalar. The BeliefEncoder treats each step's (obs_5D, action) as a
6-dimensional token — same Transformer architecture, richer per-step input.

BeliefEncoder:
  Input:  obs_hist (batch, N, L, obs_dim),  act_hist (batch, N, L)
  Token:  cat([obs_dim, action=1]) = obs_dim+1 per timestep
  Output: z_i  (batch, N, z_dim)

Auxiliary next-obs prediction loss:
  ObsPredHead: MLP(z_i, a_i) -> predicted y_{t+1,i} in R^obs_dim
  Forces encoder to capture (alpha_i, beta_i) in z_i because each vital's
  next value depends on the latent health transition.

Architecture follows the synthetic BIRD trainer:
  - Encoder is shared between actor and critic optimizers.
  - Global reward normalization (scalar mu_r / sig_r EMA).
  - No separate target encoder; z_next uses the live encoder.
"""
from __future__ import annotations

import csv
import os
import random
import sys
import argparse
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
from diffusion_model import (
    PerArmDiffusionActor,
    PerArmTwinCritic,
    soft_update_,
    topk_action,
)
from env import MIMICRMABConfig, MIMICRMABEnv, OBS_DIM


# ---------------------------------------------------------------------------
# Auxiliary head: predict next 5D obs from belief + action
# ---------------------------------------------------------------------------

class ObsPredHead(nn.Module):
    """
    Per-arm MLP: (z_i, a_i) -> predicted next obs for the first pred_dim channels.
    pred_dim=5 predicts only the always-observed vitals (HR, SBP, MBP, SpO2, RR),
    excluding sparse temp/glucose and their masks — those can't be reliably predicted.
    """

    def __init__(self, z_dim: int, obs_dim: int, hidden_dim: int = 64,
                 pred_dim: int = 5):
        super().__init__()
        self.obs_dim  = obs_dim
        self.pred_dim = pred_dim   # number of channels to predict (first pred_dim of obs)
        self.net = nn.Sequential(
            nn.Linear(z_dim + 1, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, pred_dim),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        z:      (batch, N, z_dim)
        action: (batch, N)
        returns: (batch, N, pred_dim)
        """
        batch, N, _ = z.shape
        a = action.unsqueeze(-1)                         # (batch, N, 1)
        x = torch.cat([z, a], dim=-1)                   # (batch, N, z_dim+1)
        return self.net(x.reshape(batch * N, -1)).reshape(batch, N, self.pred_dim)


# ---------------------------------------------------------------------------
# Belief Encoder: Transformer over (obs_5D, action) history per arm
# ---------------------------------------------------------------------------

class BeliefEncoder(nn.Module):
    """
    Per-arm Transformer over L-step (obs_5D, action) history.

    Each timestep token = cat([obs_dim vitals, action]) = obs_dim+1 dims.
    Richer than the synthetic scalar observations: 5x more signal per step for type inference,
    making patient type disambiguation faster from shorter history windows.

    Input:  obs_hist (batch, N, L, obs_dim),  act_hist (batch, N, L)
    Output: z (batch, N, z_dim)
    """

    def __init__(self, obs_dim: int, z_dim: int, hidden_dim: int = 64,
                 n_heads: int = 4, n_layers: int = 2, L: int = 20):
        super().__init__()
        self.L       = L
        self.obs_dim = obs_dim
        token_dim    = obs_dim + 1           # obs_dim vitals + 1 action
        self.input_proj = nn.Linear(token_dim, hidden_dim)
        self.pos_emb    = nn.Embedding(L, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out_proj    = nn.Linear(hidden_dim, z_dim)

    def forward(self, obs_hist: torch.Tensor, act_hist: torch.Tensor) -> torch.Tensor:
        """
        obs_hist: (batch, N, L, obs_dim)
        act_hist: (batch, N, L)
        returns:  (batch, N, z_dim)
        """
        batch, N, L, obs_dim = obs_hist.shape
        # cat along last dim: (batch, N, L, obs_dim+1) -> (batch*N, L, obs_dim+1)
        tokens = torch.cat([obs_hist, act_hist.unsqueeze(-1)], dim=-1)
        tokens = tokens.reshape(batch * N, L, obs_dim + 1)
        x = self.input_proj(tokens)                       # (batch*N, L, hidden)
        pos = torch.arange(L, device=x.device)
        x = x + self.pos_emb(pos).unsqueeze(0)
        x = self.transformer(x)                           # (batch*N, L, hidden)
        z = self.out_proj(x.mean(dim=1))                  # mean pool -> (batch*N, z_dim)
        return z.reshape(batch, N, -1)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DPMDTrainConfig:
    seed:   int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    N: int = 20
    K: int = 5
    T: int = 100

    obs_dim: int = OBS_DIM

    # Belief encoder
    L:               int   = 80
    z_dim:           int   = 64
    encoder_hidden:  int   = 128
    encoder_heads:   int   = 4
    encoder_layers:  int   = 3

    # Auxiliary loss: predict next obs (all 5 vitals = full obs_dim)
    aux_coef:     float = 0.5

    # Diffusion actor
    T_diff:             int   = 20
    actor_hidden:       int   = 64
    actor_t_dim:        int   = 32
    score_clip:         float = 6.0
    action_candidates:  int   = 4
    target_action_candidates: int = 4

    # Critic
    critic_hidden: int   = 64
    gamma:         float = 0.99
    tau:           float = 0.005

    # Training
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

    # DPMD lambda
    lambda_init:   float = 2.0
    lambda_target: float = 1.0
    lambda_beta:   float = 0.005
    lambda_min:    float = 0.5
    lambda_max:    float = 20.0
    ema_xi:        float = 0.05
    clip_exp:      float = 3.0

    policy_score_noise: float = 0.15

    save_dir:    str = "checkpoints_dpmd"
    save_every:  int = 25
    resume_path: str = ""


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _random_action(n_arms: int, budget: int, rng: np.random.Generator) -> np.ndarray:
    a = np.zeros(n_arms, dtype=np.int64)
    if budget > 0:
        chosen = rng.choice(n_arms, size=min(budget, n_arms), replace=False)
        a[chosen] = 1
    return a


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class DPMDAgent:
    def __init__(self, N: int, K: int, cfg: DPMDTrainConfig):
        self.N = int(N); self.K = int(K); self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.rng    = np.random.default_rng(cfg.seed)

        self.encoder = BeliefEncoder(
            obs_dim=cfg.obs_dim, z_dim=cfg.z_dim, hidden_dim=cfg.encoder_hidden,
            n_heads=cfg.encoder_heads, n_layers=cfg.encoder_layers, L=cfg.L,
        ).to(self.device)

        self.obs_pred_head = ObsPredHead(
            z_dim=cfg.z_dim, obs_dim=cfg.obs_dim, hidden_dim=cfg.encoder_hidden,
            pred_dim=cfg.obs_dim,
        ).to(self.device)

        self.actor = PerArmDiffusionActor(
            n_arms=self.N, z_dim=cfg.z_dim, hidden_dim=cfg.actor_hidden,
            t_dim=cfg.actor_t_dim, diffusion_steps=cfg.T_diff,
            device=self.device, score_clip=cfg.score_clip,
        ).to(self.device)
        self.actor_t = PerArmDiffusionActor(
            n_arms=self.N, z_dim=cfg.z_dim, hidden_dim=cfg.actor_hidden,
            t_dim=cfg.actor_t_dim, diffusion_steps=cfg.T_diff,
            device=self.device, score_clip=cfg.score_clip,
        ).to(self.device)
        self.actor_t.load_state_dict(self.actor.state_dict())

        self.critic   = PerArmTwinCritic(
            n_arms=self.N, z_dim=cfg.z_dim, hidden_dim=cfg.critic_hidden).to(self.device)
        self.critic_t = PerArmTwinCritic(
            n_arms=self.N, z_dim=cfg.z_dim, hidden_dim=cfg.critic_hidden).to(self.device)
        self.critic_t.load_state_dict(self.critic.state_dict())

        self.opt_actor  = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.actor.parameters()),
            lr=cfg.lr_actor)
        self.opt_critic = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.obs_pred_head.parameters())
            + list(self.critic.parameters()), lr=cfg.lr_critic)

        self.sched_critic = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_critic, T_max=cfg.epochs, eta_min=cfg.lr_critic_min)

        self.lambda_md = float(cfg.lambda_init)
        self.mu_q  = 0.0; self.sig_q = 1.0; self.update_step = 0
        self.mu_r  = 0.0
        self.sig_r = 1.0

        self._obs_hist: np.ndarray | None = None
        self._act_hist: np.ndarray | None = None

    def reset_history(self):
        self._obs_hist = np.zeros((self.N, self.cfg.L, self.cfg.obs_dim), dtype=np.float32)
        self._act_hist = np.zeros((self.N, self.cfg.L), dtype=np.float32)

    def _encode(self, obs_hist: torch.Tensor, act_hist: torch.Tensor) -> torch.Tensor:
        """obs_hist (batch,N,L,obs_dim), act_hist (batch,N,L) -> z (batch,N,z_dim)"""
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
        """
        obs_hist: (N, L, obs_dim)   act_hist: (N, L)
        Returns (action (N,), scores (N,))
        """
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
        """Greedy evaluation action. obs: (N, obs_dim). Call reset_history() first."""
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
        act_hist      = batch["act_hist"].float()       # (B, N, L)
        obs_hist_next = batch["obs_hist_next"].float()
        act_hist_next = batch["act_hist_next"].float()
        action = batch["action"].float()                # (B, N)
        reward = batch["reward"].float()                # (B, N)
        done   = batch["done"].float()                  # (B,)

        with torch.no_grad():
            r_mean = float(reward.mean().item())
            r_std  = float(reward.std().item() + 1e-6)
            self.mu_r  = (1 - self.cfg.ema_xi) * self.mu_r  + self.cfg.ema_xi * r_mean
            self.sig_r = (1 - self.cfg.ema_xi) * self.sig_r + self.cfg.ema_xi * r_std
            reward_norm = (reward - self.mu_r) / (self.sig_r + 1e-8)

        # Critic update
        z = self._encode(obs_hist, act_hist)
        with torch.no_grad():
            z_next      = self._encode(obs_hist_next, act_hist_next)
            next_scores = self.sample_scores(z_next, actor=self.actor_t, critic=self.critic_t,
                                             num_candidates=self.cfg.target_action_candidates)
            next_action = topk_action(next_scores, self.K).float()
            td_target   = (reward_norm
                           + self.cfg.gamma * (1.0 - done).unsqueeze(1)
                           * self.critic_t.min_q(z_next, next_action))

        q1, q2  = self.critic(z.detach(), action)
        loss_q  = F.smooth_l1_loss(q1, td_target) + F.smooth_l1_loss(q2, td_target)

        # Auxiliary: predict next full observation
        obs_next      = batch["obs_next"].float()                          # (B, N, obs_dim)
        pred_obs_next = self.obs_pred_head(z, action)                     # (B, N, obs_dim)
        loss_aux      = F.mse_loss(pred_obs_next, obs_next)

        loss_critic_total = loss_q + self.cfg.aux_coef * loss_aux

        self.opt_critic.zero_grad(set_to_none=True)
        loss_critic_total.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters())
            + list(self.obs_pred_head.parameters())
            + list(self.critic.parameters()), self.cfg.grad_clip)
        self.opt_critic.step()

        z_actor  = self._encode(obs_hist, act_hist)
        a0_old   = batch["indices"].float()             # (B, N) scores at collection time
        with torch.no_grad():
            total_q = self._score_objective(self.critic, z_actor.detach(), a0_old)
            mean_q  = float(total_q.mean().item())
            self.mu_q  = (1 - self.cfg.ema_xi) * self.mu_q  + self.cfg.ema_xi * mean_q
            self.sig_q = (1 - self.cfg.ema_xi) * self.sig_q + \
                         self.cfg.ema_xi * float(total_q.std().item() + 1e-6)
            norm_q  = (total_q - self.mu_q) / (self.sig_q + 1e-6)
            logits  = torch.clamp(norm_q / max(self.lambda_md, 1e-6),
                                  -self.cfg.clip_exp, self.cfg.clip_exp)
            weights = (torch.exp(logits) / (torch.exp(logits).mean() + 1e-8)).detach()

        t_diff   = torch.randint(0, self.cfg.T_diff, (obs_hist.size(0),),
                                 device=self.device, dtype=torch.long)
        eps      = torch.randn_like(a0_old)
        eps_pred = self.actor.eps_pred(self.actor.q_sample(a0_old, t_diff, eps), z_actor, t_diff)
        actor_loss = torch.mean(weights.unsqueeze(1) * (eps_pred - eps) ** 2)

        self.opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.actor.parameters()),
            self.cfg.grad_clip)
        self.opt_actor.step()

        soft_update_(self.actor_t,  self.actor,  self.cfg.tau)
        soft_update_(self.critic_t, self.critic, self.cfg.tau)
        self.lambda_md = float(np.clip(
            self.lambda_md + self.cfg.lambda_beta * (self.cfg.lambda_target - self.lambda_md),
            self.cfg.lambda_min, self.cfg.lambda_max))
        self.update_step += 1

        return {"loss_q": float(loss_q.item()),
                "loss_actor": float(actor_loss.item()),
                "q_mean": mean_q}

    def checkpoint_dict(self, epoch: int = 0, best_return: float = -float("inf")) -> Dict:
        return {
            "epoch": int(epoch), "best_return": float(best_return),
            "lambda_md": float(self.lambda_md),
            "mu_q": float(self.mu_q), "sig_q": float(self.sig_q),
            "mu_r": float(self.mu_r), "sig_r": float(self.sig_r),
            "cfg": asdict(self.cfg),
            "encoder":      self.encoder.state_dict(),
            "obs_pred_head": self.obs_pred_head.state_dict(),
            "actor":        self.actor.state_dict(),
            "actor_t":      self.actor_t.state_dict(),
            "critic":       self.critic.state_dict(),
            "critic_t":     self.critic_t.state_dict(),
            "opt_actor":    self.opt_actor.state_dict(),
            "opt_critic":   self.opt_critic.state_dict(),
            "sched_critic": self.sched_critic.state_dict(),
        }

    def load_checkpoint(self, path: str | Path) -> Tuple[int, float]:
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"], strict=False)
        if "obs_pred_head" in ckpt:
            self.obs_pred_head.load_state_dict(ckpt["obs_pred_head"], strict=False)
        self.actor.load_state_dict(ckpt["actor"], strict=False)
        self.actor_t.load_state_dict(ckpt.get("actor_t", ckpt["actor"]), strict=False)
        self.critic.load_state_dict(ckpt["critic"], strict=False)
        self.critic_t.load_state_dict(ckpt.get("critic_t", ckpt["critic"]), strict=False)
        for opt, key in [(self.opt_actor, "opt_actor"), (self.opt_critic, "opt_critic")]:
            if key in ckpt:
                try:
                    opt.load_state_dict(ckpt[key])
                except ValueError:
                    pass
        if "sched_critic" in ckpt:
            self.sched_critic.load_state_dict(ckpt["sched_critic"])
        self.lambda_md = float(ckpt.get("lambda_md", self.cfg.lambda_init))
        self.mu_q  = float(ckpt.get("mu_q",  0.0))
        self.sig_q = float(ckpt.get("sig_q", 1.0))
        self.mu_r  = float(ckpt.get("mu_r",  0.0))
        self.sig_r = float(ckpt.get("sig_r", 1.0))
        return int(ckpt.get("epoch", 0)), float(ckpt.get("best_return", -float("inf")))


# ---------------------------------------------------------------------------
# Replay buffer — obs stored as (N, obs_dim) instead of (N,)
# ---------------------------------------------------------------------------

class HistoryReplayBuffer:
    def __init__(self, max_size: int, n_arms: int, L: int, obs_dim: int,
                 device: torch.device):
        self.max_size = int(max_size)
        self.n_arms   = int(n_arms)
        self.L        = int(L)
        self.obs_dim  = int(obs_dim)
        self.device   = device

        H  = (max_size, n_arms, L, obs_dim)   # history shape
        HN = (max_size, n_arms, L)             # act history shape
        S  = (max_size, n_arms)
        SO = (max_size, n_arms, obs_dim)

        self._obs_hist      = np.zeros(H,  dtype=np.float32)
        self._act_hist      = np.zeros(HN, dtype=np.float32)
        self._obs_hist_next = np.zeros(H,  dtype=np.float32)
        self._act_hist_next = np.zeros(HN, dtype=np.float32)
        self._obs           = np.zeros(SO, dtype=np.float32)
        self._obs_next      = np.zeros(SO, dtype=np.float32)
        self._action        = np.zeros(S,  dtype=np.float32)
        self._reward        = np.zeros(S,  dtype=np.float32)
        self._done          = np.zeros(max_size, dtype=np.float32)
        self._indices       = np.zeros(S,  dtype=np.float32)
        self._ptr = 0; self.size = 0

    def push(self, obs_hist, act_hist, obs, action, reward,
             obs_next, obs_hist_next, act_hist_next, done, indices):
        p = self._ptr
        self._obs_hist[p]      = obs_hist
        self._act_hist[p]      = act_hist
        self._obs_hist_next[p] = obs_hist_next
        self._act_hist_next[p] = act_hist_next
        self._obs[p]           = obs
        self._obs_next[p]      = obs_next
        self._action[p]        = action
        self._reward[p]        = reward
        self._done[p]          = float(done)
        self._indices[p]       = indices
        self._ptr  = (self._ptr + 1) % self.max_size
        self.size  = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        to  = lambda arr: torch.from_numpy(arr[idx]).to(self.device)
        return {
            "obs_hist":      to(self._obs_hist),
            "act_hist":      to(self._act_hist),
            "obs_hist_next": to(self._obs_hist_next),
            "act_hist_next": to(self._act_hist_next),
            "obs":           to(self._obs),
            "obs_next":      to(self._obs_next),
            "action":        to(self._action),
            "reward":        to(self._reward),
            "done":          to(self._done),
            "indices":       to(self._indices),
        }


# ---------------------------------------------------------------------------
# Episode collection
# ---------------------------------------------------------------------------

def _collect_episode(env: MIMICRMABEnv, agent: DPMDAgent,
                     replay: HistoryReplayBuffer,
                     random_policy: bool = False) -> float:
    obs, _ = env.reset()                         # obs: (N, obs_dim)
    N      = env.cfg.N
    L      = agent.cfg.L
    obs_dim = env.cfg.obs_dim

    obs_hist = np.zeros((N, L, obs_dim), dtype=np.float32)
    act_hist = np.zeros((N, L),          dtype=np.float32)
    ep_return = 0.0

    for _ in range(env.cfg.T):
        obs_hist = np.roll(obs_hist, -1, axis=1)
        obs_hist[:, -1, :] = obs

        if random_policy:
            action  = _random_action(N, env.cfg.K, env.rng)
            indices = action.astype(np.float32)
        else:
            action, indices = agent.select_action(obs_hist, act_hist, explore=True)

        obs_next, reward_vec, done, _ = env.step(action)
        ep_return += float(reward_vec.sum())

        obs_hist_next = np.roll(obs_hist, -1, axis=1)
        obs_hist_next[:, -1, :] = obs_next
        act_hist_next = np.roll(act_hist, -1, axis=1)
        act_hist_next[:, -1]   = action.astype(np.float32)

        replay.push(
            obs_hist=obs_hist, act_hist=act_hist,
            obs=obs, action=action.astype(np.float32), reward=reward_vec,
            obs_next=obs_next,
            obs_hist_next=obs_hist_next, act_hist_next=act_hist_next,
            done=done, indices=indices,
        )

        obs      = obs_next
        act_hist = act_hist_next

        if done:
            break

    return ep_return


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _save_training_plots(save_dir, epochs, ep_returns, avg10_returns,
                         loss_q, loss_actor, q_mean):
    fig, axes = plt.subplots(3, 1, figsize=(10, 11))
    axes[0].plot(epochs, ep_returns, alpha=0.4, label="ep_return")
    axes[0].plot(epochs, avg10_returns, label="avg10_return", linewidth=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Return")
    axes[0].set_title("DPMD Training Return (MIMIC-ICU RMAB)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    rq_ep, rel_q = zip(*[(e, lq / abs(qm)) for e, lq, qm in zip(epochs, loss_q, q_mean)
                          if abs(qm) >= 1.0]) if any(abs(q) >= 1.0 for q in q_mean) \
                   else (epochs, [float("nan")] * len(epochs))
    axes[1].plot(rq_ep, rel_q, color="tab:blue")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss Q / |Q mean|")
    axes[1].set_title("Relative Critic Error"); axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, loss_actor, color="tab:orange")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Loss")
    axes[2].set_title("Loss Actor"); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "dpmd_training_curves.png")
    plt.savefig(path, dpi=150); plt.close(fig)
    print(f"[plot] saved {path}")


def train(env: MIMICRMABEnv, cfg: DPMDTrainConfig | None = None,
          checkpoint_dir: str | None = None) -> DPMDAgent:
    cfg = cfg or DPMDTrainConfig()
    cfg.N       = env.cfg.N
    cfg.K       = env.cfg.K
    cfg.T       = env.cfg.T
    cfg.obs_dim = env.cfg.obs_dim
    if checkpoint_dir is not None:
        cfg.save_dir = checkpoint_dir

    set_seed(cfg.seed)
    agent  = DPMDAgent(N=cfg.N, K=cfg.K, cfg=cfg)
    replay = HistoryReplayBuffer(cfg.replay_size, n_arms=cfg.N, L=cfg.L,
                                 obs_dim=cfg.obs_dim, device=agent.device)

    best_return = -float("inf"); start_epoch = 0
    if cfg.resume_path:
        start_epoch, best_return = agent.load_checkpoint(cfg.resume_path)
        print(f"[resume] loaded {cfg.resume_path} at epoch {start_epoch}")

    os.makedirs(cfg.save_dir, exist_ok=True)
    metrics_path = os.path.join(cfg.save_dir, "dpmd_training_metrics.csv")
    fieldnames   = ["epoch", "ep_return", "avg10_return", "best_return",
                    "loss_q", "loss_actor", "q_mean", "lambda_md"]
    metrics_mode = "a" if (cfg.resume_path and Path(metrics_path).exists()) else "w"

    for _ in range(cfg.warmup_episodes):
        _collect_episode(env, agent, replay, random_policy=True)

    returns: List = []
    plot_epochs: List = []; plot_ep: List = []; plot_avg10: List = []
    plot_lq: List = [];     plot_la: List = []; plot_qm: List = []

    with open(metrics_path, metrics_mode, newline="", encoding="utf-8") as mf:
        writer = csv.DictWriter(mf, fieldnames=fieldnames)
        if metrics_mode == "w":
            writer.writeheader(); mf.flush()

        for epoch in range(start_epoch, cfg.epochs):
            ep_return = _collect_episode(env, agent, replay)
            returns.append(ep_return)

            if replay.size >= max(cfg.start_updates_after, cfg.batch_size):
                losses = {"loss_q": 0.0, "loss_actor": 0.0, "q_mean": 0.0}
                for _ in range(cfg.updates_per_epoch):
                    m = agent.update(replay.sample(cfg.batch_size))
                    for k in losses:
                        losses[k] += m[k]
                for k in losses:
                    losses[k] /= cfg.updates_per_epoch
            else:
                losses = {k: float("nan") for k in ("loss_q", "loss_actor", "q_mean")}

            avg10 = float(np.mean(returns[-10:]))
            plot_epochs.append(epoch + 1); plot_ep.append(ep_return)
            plot_avg10.append(avg10); plot_lq.append(losses["loss_q"])
            plot_la.append(losses["loss_actor"]); plot_qm.append(losses["q_mean"])

            if (epoch + 1) % 10 == 0:
                print(f"[DPMD] Epoch {epoch+1:04d} | avg10 {avg10:8.1f} | "
                      f"lam {agent.lambda_md:.3f} | LossQ {losses['loss_q']:.4f} | "
                      f"LossA {losses['loss_actor']:.4f}")

            agent.sched_critic.step()

            if (epoch + 1) % cfg.save_every == 0:
                torch.save(agent.checkpoint_dict(epoch + 1, best_return),
                           os.path.join(cfg.save_dir, "latest.pth"))

            if avg10 > best_return:
                best_return = avg10
                best_path   = os.path.join(cfg.save_dir, "best.pth")
                torch.save(agent.checkpoint_dict(epoch + 1, best_return), best_path)
                print(f"[DPMD] saved BEST {best_path} (avg10={avg10:.1f})")

            writer.writerow({
                "epoch": epoch + 1, "ep_return": ep_return, "avg10_return": avg10,
                "best_return": best_return, "loss_q": losses["loss_q"],
                "loss_actor": losses["loss_actor"], "q_mean": losses["q_mean"],
                "lambda_md": agent.lambda_md,
            })
            mf.flush()

    _save_training_plots(cfg.save_dir, plot_epochs, plot_ep, plot_avg10,
                         plot_lq, plot_la, plot_qm)
    return agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--L", type=int, default=80)
    parser.add_argument("--z_dim", type=int, default=64)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints_dpmd")
    parser.add_argument("--resume_path", type=str, default="")
    args = parser.parse_args()

    cfg = DPMDTrainConfig(
        N=args.N, K=args.K, T=args.T, obs_dim=OBS_DIM,
        seed=args.seed, epochs=args.epochs, L=args.L, z_dim=args.z_dim,
        save_dir=args.ckpt_dir, resume_path=args.resume_path,
    )
    env = MIMICRMABEnv(MIMICRMABConfig(N=cfg.N, K=cfg.K, T=cfg.T), seed=cfg.seed)
    train(env, cfg=cfg)


if __name__ == "__main__":
    main()
