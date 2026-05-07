"""
diffusion_DPMD_train.py  (markov2)

Diffusion Policy Mirror Descent on the 2-state Markov RMAB.

Env interface:
    obs, info          = env.reset()          obs = binary state (N,)
    obs_next, r, done, info = env.step(a)     r   = new state (0/1 float per arm)

Two encoder modes (controlled by DPMDTrainConfig.use_oracle_encoder):

  OracleEncoder (default, use_oracle_encoder=True):
    MLP(state_i, q0_i, q1_i, p0_i, p1_i) → z_i
    Known dynamics supplied directly — arm identity is encoded from step 1.
    Designed to test whether DPMD can recover true Whittle indices given full
    information (NIPS Experiment 1).  No history needed.

  BeliefEncoder (use_oracle_encoder=False):
    Transformer over L-step (obs, action) history per arm.
    Used for fair comparison with WIQL (which accumulates implicit history
    via per-arm Q-tables) when transition dynamics are unknown.

History rolling convention (BeliefEncoder only):
  obs_hist[i, -1] = most recent observation for arm i
  act_hist[i, -1] = most recent action taken on arm i
"""
from __future__ import annotations

import csv
import os
import random
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

from env import MarkovRMABConfig, MarkovRMABEnv
from diffusion_model import (
    PerArmDiffusionActor,
    PerArmTwinCritic,
    soft_update_,
    topk_action,
)


# ---------------------------------------------------------------------------
# BeliefEncoder  — Transformer over per-arm (obs, action) history
# ---------------------------------------------------------------------------

class BeliefEncoder(nn.Module):
    """
    Per-arm Transformer encoder over L-step (obs, action) history.

    Even for binary-state envs, L steps of history allow the encoder to
    infer each arm's transition probabilities (q0, q1, p0, p1) from
    empirical trajectories, enabling arm-specific Whittle index estimation.

    Input:  obs_hist (batch, N, L)  — binary state per arm over last L steps
            act_hist (batch, N, L)  — actions taken per arm over last L steps
    Output: z        (batch, N, z_dim)
    """
    def __init__(self, z_dim: int, hidden_dim: int = 64,
                 n_heads: int = 2, n_layers: int = 2, L: int = 20):
        super().__init__()
        self.L = L
        self.input_proj = nn.Linear(2, hidden_dim)
        self.pos_emb    = nn.Embedding(L, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(hidden_dim, z_dim)

    def forward(self, obs_hist: torch.Tensor, act_hist: torch.Tensor) -> torch.Tensor:
        batch, N, L = obs_hist.shape
        x = torch.stack([obs_hist, act_hist], dim=-1).reshape(batch * N, L, 2)
        x = self.input_proj(x)
        pos = torch.arange(L, device=x.device)
        x = x + self.pos_emb(pos).unsqueeze(0)
        x = self.transformer(x)
        z = self.out_proj(x.mean(dim=1))
        return z.reshape(batch, N, -1)


# ---------------------------------------------------------------------------
# OracleEncoder  — MLP over (state, q0, q1, p0, p1) per arm
# ---------------------------------------------------------------------------

class OracleEncoder(nn.Module):
    """
    Per-arm MLP encoder given full transition dynamics.

    Concatenates each arm's current binary state with its known transition
    probabilities (q0, q1, p0, p1) → unique z_i per arm from step 1.
    No history needed: arm identity is encoded directly in the dynamics.

    Input:  obs         (batch, N)     — current binary state
            transitions (N, 4)         — [q0, q1, p0, p1] per arm (fixed)
    Output: z           (batch, N, z_dim)
    """
    def __init__(self, z_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, obs: torch.Tensor, transitions: torch.Tensor) -> torch.Tensor:
        # obs: (batch, N), transitions: (N, 4)
        batch, N = obs.shape
        t = transitions.unsqueeze(0).expand(batch, -1, -1)  # (batch, N, 4)
        x = torch.cat([obs.unsqueeze(-1), t], dim=-1)       # (batch, N, 5)
        return self.net(x.reshape(batch * N, 5)).reshape(batch, N, -1)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DPMDTrainConfig:
    seed:   int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    N: int = 50
    K: int = 10
    T: int = 100

    # Encoder selection
    # use_oracle_encoder=True:  MLP(state, q0, q1, p0, p1) -> z  [known dynamics]
    # use_oracle_encoder=False: BeliefEncoder Transformer over L-step history
    use_oracle_encoder: bool = True

    # OracleEncoder / shared hidden dim
    z_dim:          int = 64
    encoder_hidden: int = 64

    # BeliefEncoder only (ignored when use_oracle_encoder=True)
    L:              int = 20
    encoder_heads:  int = 2
    encoder_layers: int = 2

    # Diffusion actor
    T_diff:                   int   = 20
    actor_hidden:             int   = 64
    actor_t_dim:              int   = 32
    score_clip:               float = 6.0
    action_candidates:        int   = 8
    target_action_candidates: int   = 8

    # Critic
    critic_hidden: int   = 128
    gamma:         float = 0.99
    tau:           float = 0.01

    # Training
    batch_size:          int   = 128
    replay_size:         int   = 20_000
    epochs:              int   = 400
    updates_per_epoch:   int   = 200
    start_updates_after: int   = 500
    warmup_episodes:     int   = 40
    lr_actor:            float = 3e-4
    lr_encoder:          float = 1e-4   # slower than actor — encoder shifts cause critic distribution shift
    lr_critic:           float = 1e-3
    lr_critic_min:       float = 1e-4
    grad_clip:           float = 1.0

    # N-step returns: reduces bootstrapping bias for high-gamma environments.
    # TD target = Σ_{k=0}^{n-1} γ^k r_{t+k}  +  γ^n Q(s_{t+n}, a*)
    # n=1 recovers standard 1-step TD; n=5 is a good default for γ=0.99.
    n_step: int = 5

    # DPMD lambda — operates on normalized Q (after mu_q/sig_q standardization),
    # so scale of raw Q does not matter here.
    lambda_init:   float = 2.0
    lambda_target: float = 1.0
    lambda_beta:   float = 0.005
    lambda_min:    float = 0.5
    lambda_max:    float = 20.0
    ema_xi:        float = 0.05
    clip_exp:      float = 3.0

    policy_score_noise: float = 0.40

    save_dir:    str = "checkpoints_dpmd"
    save_every:  int = 25
    resume_path: str = ""
    # Set resume_path = "checkpoints_dpmd/latest.pth" to continue from checkpoint


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _random_action(n_arms: int, budget: int, rng: np.random.Generator) -> np.ndarray:
    a = np.zeros(n_arms, dtype=np.int64)
    if budget > 0:
        a[rng.choice(n_arms, size=min(budget, n_arms), replace=False)] = 1
    return a


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class DPMDAgent:
    def __init__(self, N: int, K: int, cfg: DPMDTrainConfig,
                 transitions: np.ndarray | None = None):
        """
        transitions: (N, 4) array [q0, q1, p0, p1] per arm.
                     Required when cfg.use_oracle_encoder=True.
        """
        self.N = int(N); self.K = int(K); self.cfg = cfg
        self.device = torch.device(cfg.device)

        if cfg.use_oracle_encoder:
            assert transitions is not None, "transitions required for OracleEncoder"
            self.encoder = OracleEncoder(
                z_dim=cfg.z_dim, hidden_dim=cfg.encoder_hidden,
            ).to(self.device)
            self._transitions = torch.from_numpy(
                transitions.astype(np.float32)).to(self.device)  # (N, 4)
        else:
            self.encoder = BeliefEncoder(
                z_dim=cfg.z_dim, hidden_dim=cfg.encoder_hidden,
                n_heads=cfg.encoder_heads, n_layers=cfg.encoder_layers,
                L=cfg.L,
            ).to(self.device)
            self._transitions = None

        self.actor   = PerArmDiffusionActor(N, cfg.z_dim, cfg.actor_hidden, cfg.actor_t_dim,
                                            cfg.T_diff, self.device, cfg.score_clip).to(self.device)
        self.actor_t = PerArmDiffusionActor(N, cfg.z_dim, cfg.actor_hidden, cfg.actor_t_dim,
                                            cfg.T_diff, self.device, cfg.score_clip).to(self.device)
        self.actor_t.load_state_dict(self.actor.state_dict())

        self.critic   = PerArmTwinCritic(N, cfg.z_dim, cfg.critic_hidden).to(self.device)
        self.critic_t = PerArmTwinCritic(N, cfg.z_dim, cfg.critic_hidden).to(self.device)
        self.critic_t.load_state_dict(self.critic.state_dict())

        # Encoder gets its own lower LR in both optimizers.
        # The actor drives encoder updates; if encoder LR == lr_actor the encoder
        # shifts too fast for the critic to track, causing q_mean oscillation.
        self.opt_actor = torch.optim.Adam([
            {"params": self.encoder.parameters(), "lr": cfg.lr_encoder},
            {"params": self.actor.parameters(),   "lr": cfg.lr_actor},
        ])
        self.opt_critic = torch.optim.Adam([
            {"params": self.encoder.parameters(), "lr": cfg.lr_encoder},
            {"params": self.critic.parameters(),  "lr": cfg.lr_critic},
        ])

        # Anneal only the critic param group (index 1); encoder group stays at lr_encoder.
        self.sched_critic = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_critic, T_max=cfg.epochs, eta_min=cfg.lr_critic_min)
        # After scheduler is created, pin the encoder group's LR so annealing
        # doesn't accidentally lower it (CosineAnnealingLR scales all groups).
        self._encoder_lr = cfg.lr_encoder

        self.lambda_md = float(cfg.lambda_init)
        self.mu_q = 0.0; self.sig_q = 1.0; self.update_step = 0
        self.mu_r = 0.0; self.sig_r = 1.0  # running reward normalizer

        # Internal history for act_hard() / eval rollouts
        self._obs_hist: np.ndarray | None = None
        self._act_hist: np.ndarray | None = None

    def reset_history(self):
        """No-op for oracle encoder; resets history buffers for belief encoder."""
        if not self.cfg.use_oracle_encoder:
            self._obs_hist = np.zeros((self.N, self.cfg.L), dtype=np.float32)
            self._act_hist = np.zeros((self.N, self.cfg.L), dtype=np.float32)

    def _encode(self, obs_or_hist: torch.Tensor,
                act_hist: torch.Tensor | None = None) -> torch.Tensor:
        if self.cfg.use_oracle_encoder:
            return self.encoder(obs_or_hist, self._transitions)
        return self.encoder(obs_or_hist, act_hist)

    def _score_obj(self, critic, z, scores):
        return critic.min_q(z, topk_action(scores, self.K).float()).sum(dim=1)

    @torch.no_grad()
    def sample_scores(self, z, actor=None, critic=None, num_candidates=None):
        actor = actor or self.actor; critic = critic or self.critic
        n = max(1, int(num_candidates or 1))
        if n == 1:
            return actor.sample(z)
        cands  = actor.sample(z, num_samples=n)
        values = torch.stack([self._score_obj(critic, z, cands[:, i]) for i in range(n)], dim=1)
        best   = torch.argmax(values, dim=1)
        return cands[torch.arange(z.size(0), device=z.device), best]

    @torch.no_grad()
    def select_action(self, obs: np.ndarray,
                      obs_hist: np.ndarray | None = None,
                      act_hist: np.ndarray | None = None,
                      explore: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Oracle encoder:  uses obs (N,) directly.
        Belief encoder:  uses obs_hist (N,L) and act_hist (N,L).
        Returns (action (N,), clean_scores (N,))
        """
        if self.cfg.use_oracle_encoder:
            obs_t = torch.from_numpy(obs[None].astype(np.float32)).to(self.device)
            z = self._encode(obs_t)
        else:
            oh = torch.from_numpy(obs_hist[None]).float().to(self.device)
            ah = torch.from_numpy(act_hist[None]).float().to(self.device)
            z  = self._encode(oh, ah)
        scores = self.sample_scores(z, num_candidates=self.cfg.action_candidates)[0]
        clean  = scores.cpu().numpy().astype(np.float32)
        if explore and self.cfg.policy_score_noise > 0:
            scores = scores + self.cfg.policy_score_noise * torch.randn_like(scores)
        action = topk_action(scores.unsqueeze(0), self.K)[0].cpu().numpy().astype(np.int64)
        return action, clean

    @torch.no_grad()
    def act_hard(self, obs: np.ndarray, **_) -> np.ndarray:
        """Greedy action for evaluation. Call reset_history() before each episode."""
        if self.cfg.use_oracle_encoder:
            action, _ = self.select_action(obs.astype(np.float32), explore=False)
        else:
            if self._obs_hist is None:
                self.reset_history()
            self._obs_hist = np.roll(self._obs_hist, -1, axis=1)
            self._obs_hist[:, -1] = obs.astype(np.float32)
            action, _ = self.select_action(obs, self._obs_hist, self._act_hist, explore=False)
            self._act_hist = np.roll(self._act_hist, -1, axis=1)
            self._act_hist[:, -1] = action.astype(np.float32)
        return action

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        action   = batch["action"].float()
        reward   = batch["reward"].float()
        done     = batch["done"].float()

        if self.cfg.use_oracle_encoder:
            obs      = batch["obs"].float()
            obs_next = batch["obs_next"].float()
            z      = self._encode(obs)
            with torch.no_grad():
                z_next = self._encode(obs_next)
        else:
            obs_hist      = batch["obs_hist"].float()
            act_hist      = batch["act_hist"].float()
            obs_hist_next = batch["obs_hist_next"].float()
            act_hist_next = batch["act_hist_next"].float()
            z      = self._encode(obs_hist, act_hist)
            with torch.no_grad():
                z_next = self._encode(obs_hist_next, act_hist_next)

        # Scale-only normalization: divide by EMA std of total reward.
        # No centering — binary rewards have fixed mean ≈ 0.65, centering would
        # collapse total_q → uniform DPMD weights.
        with torch.no_grad():
            r_total = reward.sum(dim=1)
            self.mu_r  = (1 - self.cfg.ema_xi) * self.mu_r  + self.cfg.ema_xi * float(r_total.mean().item())
            self.sig_r = (1 - self.cfg.ema_xi) * self.sig_r + self.cfg.ema_xi * float(r_total.std().item() + 1e-6)
            reward_norm = reward / (self.sig_r + 1e-8)

        with torch.no_grad():
            ns        = self.sample_scores(z_next, self.actor_t, self.critic_t,
                                           self.cfg.target_action_candidates)
            na        = topk_action(ns, self.K).float()
            # gamma_n = γ^n (0 when episode ended before n steps → no bootstrap)
            gamma_n   = batch["gamma_n"].float() if "gamma_n" in batch else \
                        torch.full((done.shape[0],), self.cfg.gamma, device=self.device)
            td_target = (reward_norm + gamma_n.unsqueeze(1)
                         * self.critic_t.min_q(z_next, na))

        q1, q2 = self.critic(z.detach(), action)
        loss_q = F.smooth_l1_loss(q1, td_target) + F.smooth_l1_loss(q2, td_target)

        self.opt_critic.zero_grad(set_to_none=True)
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.critic.parameters()),
            self.cfg.grad_clip)
        self.opt_critic.step()

        if self.cfg.use_oracle_encoder:
            z_a = self._encode(obs)
        else:
            z_a = self._encode(obs_hist, act_hist)

        a0_old = batch["indices"].float()
        with torch.no_grad():
            tq     = self._score_obj(self.critic, z_a.detach(), a0_old)
            mq     = float(tq.mean())
            self.mu_q  = (1 - self.cfg.ema_xi) * self.mu_q  + self.cfg.ema_xi * mq
            self.sig_q = (1 - self.cfg.ema_xi) * self.sig_q + self.cfg.ema_xi * float(tq.std() + 1e-6)
            nq     = (tq - self.mu_q) / (self.sig_q + 1e-6)
            logits = torch.clamp(nq / max(self.lambda_md, 1e-6),
                                 -self.cfg.clip_exp, self.cfg.clip_exp)
            w      = (torch.exp(logits) / (torch.exp(logits).mean() + 1e-8)).detach()

        t_diff   = torch.randint(0, self.cfg.T_diff, (action.size(0),),
                                 device=self.device, dtype=torch.long)
        eps      = torch.randn_like(a0_old)
        eps_pred = self.actor.eps_pred(self.actor.q_sample(a0_old, t_diff, eps), z_a, t_diff)
        aloss    = torch.mean(w.unsqueeze(1) * (eps_pred - eps) ** 2)

        self.opt_actor.zero_grad(set_to_none=True)
        aloss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.actor.parameters()),
            self.cfg.grad_clip)
        self.opt_actor.step()

        soft_update_(self.actor_t, self.actor, self.cfg.tau)
        soft_update_(self.critic_t, self.critic, self.cfg.tau)
        self.lambda_md = float(np.clip(
            self.lambda_md + self.cfg.lambda_beta * (self.cfg.lambda_target - self.lambda_md),
            self.cfg.lambda_min, self.cfg.lambda_max))
        self.update_step += 1
        return {"loss_q": float(loss_q.detach()), "loss_actor": float(aloss.detach()), "q_mean": mq}

    def checkpoint_dict(self, epoch=0, best_return=-float("inf")):
        return {"epoch": epoch, "best_return": best_return,
                "lambda_md": self.lambda_md, "mu_q": self.mu_q, "sig_q": self.sig_q,
                "mu_r": float(self.mu_r), "sig_r": float(self.sig_r),
                "cfg": asdict(self.cfg),
                "encoder": self.encoder.state_dict(),
                "actor": self.actor.state_dict(), "actor_t": self.actor_t.state_dict(),
                "critic": self.critic.state_dict(), "critic_t": self.critic_t.state_dict(),
                "opt_actor": self.opt_actor.state_dict(), "opt_critic": self.opt_critic.state_dict(),
                "sched_critic": self.sched_critic.state_dict()}

    def load_checkpoint(self, path, load_optimizers=True):
        ck = torch.load(path, map_location=self.device)
        if "encoder" in ck: self.encoder.load_state_dict(ck["encoder"], strict=False)
        self.actor.load_state_dict(ck["actor"], strict=False)
        self.actor_t.load_state_dict(ck.get("actor_t", ck["actor"]), strict=False)
        self.critic.load_state_dict(ck["critic"], strict=False)
        self.critic_t.load_state_dict(ck.get("critic_t", ck["critic"]), strict=False)
        if load_optimizers:
            for opt, key in [(self.opt_actor, "opt_actor"), (self.opt_critic, "opt_critic")]:
                if key in ck:
                    try: opt.load_state_dict(ck[key])
                    except ValueError: print(f"[resume] skipped {key}")
        if "sched_critic" in ck: self.sched_critic.load_state_dict(ck["sched_critic"])
        self.lambda_md = float(ck.get("lambda_md", self.cfg.lambda_init))
        self.mu_q  = float(ck.get("mu_q",  0.0))
        self.sig_q = float(ck.get("sig_q", 1.0))
        self.mu_r  = float(ck.get("mu_r",  0.0))
        self.sig_r = float(ck.get("sig_r", 1.0))
        return int(ck.get("epoch", 0)), float(ck.get("best_return", -float("inf")))


# ---------------------------------------------------------------------------
# History Replay Buffer
# ---------------------------------------------------------------------------

class HistoryReplayBuffer:
    """Replay buffer storing L-step history windows per arm."""

    def __init__(self, max_size: int, n_arms: int, L: int, device):
        self.max_size = int(max_size)
        self.n_arms   = int(n_arms)
        self.L        = int(L)
        self.device   = device

        H = (max_size, n_arms, L)
        S = (max_size, n_arms)

        self._obs_hist      = np.zeros(H, dtype=np.float32)
        self._act_hist      = np.zeros(H, dtype=np.float32)
        self._obs_hist_next = np.zeros(H, dtype=np.float32)
        self._act_hist_next = np.zeros(H, dtype=np.float32)
        self._action        = np.zeros(S, dtype=np.float32)
        self._reward        = np.zeros(S, dtype=np.float32)
        self._indices       = np.zeros(S, dtype=np.float32)
        self._done          = np.zeros(max_size, dtype=np.float32)
        self._ptr = 0; self.size = 0

    def push(self, obs_hist, act_hist, action, reward,
             obs_hist_next, act_hist_next, done, indices):
        p = self._ptr
        self._obs_hist[p]      = obs_hist
        self._act_hist[p]      = act_hist
        self._obs_hist_next[p] = obs_hist_next
        self._act_hist_next[p] = act_hist_next
        self._action[p]        = action
        self._reward[p]        = reward
        self._done[p]          = float(done)
        self._indices[p]       = indices
        self._ptr  = (self._ptr + 1) % self.max_size
        self.size  = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        t   = lambda x: torch.from_numpy(x[idx]).to(self.device)
        return {
            "obs_hist":      t(self._obs_hist),
            "act_hist":      t(self._act_hist),
            "obs_hist_next": t(self._obs_hist_next),
            "act_hist_next": t(self._act_hist_next),
            "action":        t(self._action),
            "reward":        t(self._reward),
            "done":          t(self._done),
            "indices":       t(self._indices),
        }


# ---------------------------------------------------------------------------
# Oracle Replay Buffer (no history windows — just obs, obs_next)
# ---------------------------------------------------------------------------

class OracleReplayBuffer:
    """Replay buffer for oracle encoder — stores plain obs instead of history.

    Supports n-step returns: reward stores the n-step discounted sum
    Σ_{k=0}^{n-1} γ^k r_{t+k}, and gamma_n stores γ^n (0 if episode ended
    before n steps) for use in the TD bootstrap target.
    """

    def __init__(self, max_size: int, n_arms: int, device):
        self.max_size = int(max_size)
        self.n_arms   = int(n_arms)
        self.device   = device

        S = (max_size, n_arms)
        self._obs      = np.zeros(S, dtype=np.float32)
        self._obs_next = np.zeros(S, dtype=np.float32)
        self._action   = np.zeros(S, dtype=np.float32)
        self._reward   = np.zeros(S, dtype=np.float32)
        self._indices  = np.zeros(S, dtype=np.float32)
        self._done     = np.zeros(max_size, dtype=np.float32)
        self._gamma_n  = np.ones(max_size, dtype=np.float32)   # γ^n per transition
        self._ptr = 0; self.size = 0

    def push(self, obs, action, reward, obs_next, done, indices, gamma_n=None):
        p = self._ptr
        self._obs[p]      = obs
        self._obs_next[p] = obs_next
        self._action[p]   = action
        self._reward[p]   = reward
        self._done[p]     = float(done)
        self._indices[p]  = indices
        self._gamma_n[p]  = float(gamma_n) if gamma_n is not None else 0.0
        self._ptr  = (self._ptr + 1) % self.max_size
        self.size  = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        t   = lambda x: torch.from_numpy(x[idx]).to(self.device)
        return {
            "obs":      t(self._obs),
            "obs_next": t(self._obs_next),
            "action":   t(self._action),
            "reward":   t(self._reward),
            "done":     t(self._done),
            "indices":  t(self._indices),
            "gamma_n":  t(self._gamma_n),
        }


# ---------------------------------------------------------------------------
# Episode collection
# ---------------------------------------------------------------------------

def _collect_episode(env: MarkovRMABEnv, agent: DPMDAgent,
                     replay,
                     random_policy: bool = False) -> float:
    """Works with both OracleReplayBuffer and HistoryReplayBuffer.

    Oracle path uses n-step returns (cfg.n_step).  Each stored transition has:
      reward  = Σ_{k=0}^{n-1} γ^k r_{t+k}   (n-step discounted sum, per arm)
      obs_next = s_{t+n}                       (state n steps later)
      gamma_n  = γ^n  (0 if episode ended within n steps)
    """
    from collections import deque

    obs, _ = env.reset()
    ep_ret = 0.0

    if agent.cfg.use_oracle_encoder:
        n     = agent.cfg.n_step
        gamma = agent.cfg.gamma
        buf: deque = deque()   # each entry: (obs_f, action, reward_vec, indices)

        def _flush_one(obs_n: np.ndarray, terminal: bool) -> None:
            """Push the oldest buffered transition using accumulated n-step return."""
            if not buf:
                return
            r_n = sum(gamma ** k * buf[k][2] for k in range(len(buf)))
            g_n = 0.0 if terminal else gamma ** len(buf)
            obs_0, act_0, _, idx_0 = buf[0]
            replay.push(
                obs=obs_0,
                action=act_0,
                reward=r_n.astype(np.float32),
                obs_next=obs_n,
                done=terminal,
                indices=idx_0,
                gamma_n=g_n,
            )
            buf.popleft()

        for _ in range(env.T):
            obs_f = obs.astype(np.float32)
            if random_policy:
                action  = _random_action(env.N, env.K, env.rng)
                indices = action.astype(np.float32)
            else:
                action, indices = agent.select_action(obs_f, explore=True)

            obs_next, reward_vec, done, _ = env.step(action)
            ep_ret += float(reward_vec.sum())
            buf.append((obs_f, action.astype(np.float32),
                        reward_vec.astype(np.float32), indices))

            if len(buf) == n:
                _flush_one(obs_next.astype(np.float32), done)

            obs = obs_next
            if done:
                # Flush remaining transitions with shorter horizons
                while buf:
                    _flush_one(obs_next.astype(np.float32), terminal=True)
                break
    else:
        N, L = env.N, agent.cfg.L
        obs_hist = np.zeros((N, L), dtype=np.float32)
        act_hist = np.zeros((N, L), dtype=np.float32)

        for _ in range(env.T):
            obs_hist = np.roll(obs_hist, -1, axis=1)
            obs_hist[:, -1] = obs.astype(np.float32)

            if random_policy:
                action  = _random_action(env.N, env.K, env.rng)
                indices = action.astype(np.float32)
            else:
                action, indices = agent.select_action(obs, obs_hist, act_hist, explore=True)

            obs_next, reward_vec, done, _ = env.step(action)
            ep_ret += float(reward_vec.sum())

            obs_hist_next = np.roll(obs_hist, -1, axis=1)
            obs_hist_next[:, -1] = obs_next.astype(np.float32)
            act_hist_next = np.roll(act_hist, -1, axis=1)
            act_hist_next[:, -1] = action.astype(np.float32)

            replay.push(
                obs_hist=obs_hist, act_hist=act_hist,
                action=action.astype(np.float32), reward=reward_vec.astype(np.float32),
                obs_hist_next=obs_hist_next, act_hist_next=act_hist_next,
                done=done, indices=indices,
            )

            obs      = obs_next
            act_hist = act_hist_next
            if done:
                break

    return ep_ret


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: DPMDTrainConfig | None = None) -> DPMDAgent:
    cfg = cfg or DPMDTrainConfig()
    set_seed(cfg.seed)

    env_cfg = MarkovRMABConfig(N=cfg.N, K=cfg.K, T=cfg.T, seed_params=cfg.seed)
    env     = MarkovRMABEnv(env_cfg, seed=cfg.seed)

    # Build (N, 4) transitions array from env's known dynamics
    transitions = np.stack([env.q0, env.q1, env.p0, env.p1], axis=1).astype(np.float32)

    agent = DPMDAgent(cfg.N, cfg.K, cfg,
                      transitions=transitions if cfg.use_oracle_encoder else None)

    if cfg.use_oracle_encoder:
        replay = OracleReplayBuffer(cfg.replay_size, cfg.N, agent.device)
    else:
        replay = HistoryReplayBuffer(cfg.replay_size, cfg.N, cfg.L, agent.device)

    best_return = -float("inf"); start_epoch = 0
    if cfg.resume_path:
        start_epoch, best_return = agent.load_checkpoint(cfg.resume_path)
        print(f"[resume] {cfg.resume_path}  epoch={start_epoch}")

    os.makedirs(cfg.save_dir, exist_ok=True)
    metrics_path = os.path.join(cfg.save_dir, "metrics.csv")
    mode = "a" if (cfg.resume_path and Path(metrics_path).exists()) else "w"
    fields = ["epoch", "ep_return", "avg10_return", "best_return",
              "loss_q", "loss_actor", "q_mean", "lambda_md"]

    for _ in range(cfg.warmup_episodes):
        _collect_episode(env, agent, replay, random_policy=True)

    returns: List = []
    plot_e: List = []; plot_r: List = []; plot_a10: List = []
    plot_lq: List = []; plot_la: List = []; plot_qm: List = []

    with open(metrics_path, mode, newline="", encoding="utf-8") as mf:
        w = csv.DictWriter(mf, fieldnames=fields)
        if mode == "w": w.writeheader(); mf.flush()

        for epoch in range(start_epoch, cfg.epochs):
            ep_ret = _collect_episode(env, agent, replay)
            returns.append(ep_ret)

            if replay.size >= max(cfg.start_updates_after, cfg.batch_size):
                losses = {"loss_q": 0.0, "loss_actor": 0.0, "q_mean": 0.0}
                for _ in range(cfg.updates_per_epoch):
                    m = agent.update(replay.sample(cfg.batch_size))
                    for k in losses: losses[k] += m[k]
                for k in losses: losses[k] /= cfg.updates_per_epoch
            else:
                losses = {k: float("nan") for k in ("loss_q", "loss_actor", "q_mean")}

            avg10 = float(np.mean(returns[-10:]))
            plot_e.append(epoch + 1); plot_r.append(ep_ret); plot_a10.append(avg10)
            plot_lq.append(losses["loss_q"]); plot_la.append(losses["loss_actor"]); plot_qm.append(losses["q_mean"])

            if (epoch + 1) % 10 == 0:
                print(f"[DPMD] {epoch+1:04d} | avg10 {avg10:7.1f} | "
                      f"lam {agent.lambda_md:.3f} | LossQ {losses['loss_q']:.4f} | "
                      f"LossA {losses['loss_actor']:.4f} | qmean {losses['q_mean']:.2f}")

            if (epoch + 1) % cfg.save_every == 0:
                torch.save(agent.checkpoint_dict(epoch + 1, best_return),
                           os.path.join(cfg.save_dir, "latest.pth"))

            agent.sched_critic.step()
            # Keep encoder LR fixed — only critic LR should anneal
            agent.opt_critic.param_groups[0]["lr"] = agent._encoder_lr

            if avg10 > best_return:
                best_return = avg10
                bp = os.path.join(cfg.save_dir, "best.pth")
                torch.save(agent.checkpoint_dict(epoch + 1, best_return), bp)
                print(f"[DPMD] BEST {bp}  avg10={avg10:.1f}")

            w.writerow({"epoch": epoch + 1, "ep_return": ep_ret, "avg10_return": avg10,
                        "best_return": best_return, "loss_q": losses["loss_q"],
                        "loss_actor": losses["loss_actor"], "q_mean": losses["q_mean"],
                        "lambda_md": agent.lambda_md})
            mf.flush()

    # plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    axes[0].plot(plot_e, plot_r, alpha=0.4, label="ep_return")
    axes[0].plot(plot_e, plot_a10, linewidth=2, label="avg10")
    enc_label = "OracleEncoder" if cfg.use_oracle_encoder else "BeliefEncoder"
    axes[0].set_title(f"DPMD — 2-State Markov RMAB ({enc_label})"); axes[0].legend(); axes[0].grid(alpha=0.3)
    rq_ep, rel_q = zip(*[(e, lq / abs(qm)) for e, lq, qm in zip(plot_e, plot_lq, plot_qm)
                          if abs(qm) >= 1.0]) if any(abs(q) >= 1.0 for q in plot_qm) else (plot_e, [float("nan")] * len(plot_e))
    axes[1].plot(rq_ep, rel_q, color="tab:blue"); axes[1].set_title("Relative Critic Error (Loss Q / |Q mean|)"); axes[1].grid(alpha=0.3)
    axes[2].plot(plot_e, plot_la, color="tab:orange"); axes[2].set_title("Loss Actor"); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, "training_curves.png"), dpi=150)
    plt.close(fig)
    print(f"[plot] {cfg.save_dir}/training_curves.png")
    return agent


if __name__ == "__main__":
    train()
