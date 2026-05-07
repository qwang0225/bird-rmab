"""
diffusion_model.py  (heartstep)

Shared neural network primitives used by diffusion_DPMD_train.py:
  - PerArmDiffusionActor  (cosine-schedule DDPM, per-arm shared score net)
  - PerArmTwinCritic      (twin Q-networks, per-arm)
  - soft_update_          (EMA target update)
  - topk_action           (differentiable top-K selection)
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


# ── Diffusion schedule ─────────────────────────────────────────────────────

def cosine_beta_schedule(steps: int, s: float = 0.008) -> torch.Tensor:
    grid      = torch.arange(steps + 1, dtype=torch.float32)
    alpha_bar = torch.cos(((grid / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas     = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clamp(betas, 1e-5, 0.999)


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half  = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


# ── Helpers ────────────────────────────────────────────────────────────────

def topk_action(scores: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return torch.zeros_like(scores, dtype=torch.long)
    k   = min(int(k), scores.size(1))
    idx = torch.topk(scores, k=k, dim=1).indices
    a   = torch.zeros_like(scores, dtype=torch.long)
    a.scatter_(1, idx, 1)
    return a


def soft_update_(target: nn.Module, source: nn.Module, tau: float):
    with torch.no_grad():
        for p_t, p in zip(target.parameters(), source.parameters()):
            p_t.data.mul_(1.0 - tau).add_(tau * p.data)


# ── Diffusion actor ────────────────────────────────────────────────────────

class ArmDiffusionScoreNet(nn.Module):
    def __init__(self, z_dim: int, t_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + 1 + t_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),          nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, noisy_w: torch.Tensor,
                z_i: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([noisy_w.unsqueeze(-1), z_i, t_emb], dim=-1)
        return self.net(x).squeeze(-1)


class PerArmDiffusionActor(nn.Module):
    """Shared diffusion actor applied independently to each arm."""

    def __init__(self, n_arms: int, z_dim: int, hidden_dim: int, t_dim: int,
                 diffusion_steps: int, device: torch.device, score_clip: float = 6.0):
        super().__init__()
        self.n_arms          = int(n_arms)
        self.t_dim           = int(t_dim)
        self.diffusion_steps = int(diffusion_steps)
        self.device          = device
        self.score_clip      = float(score_clip)
        self.score_net       = ArmDiffusionScoreNet(z_dim=z_dim, t_dim=t_dim, hidden_dim=hidden_dim)

        betas     = cosine_beta_schedule(self.diffusion_steps).to(device)
        alphas    = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas",     betas)
        self.register_buffer("alphas",    alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    def eps_pred(self, noisy_w: torch.Tensor,
                 z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb  = timestep_embedding(t, self.t_dim)
        batch, n_arms = noisy_w.shape
        t_exp  = t_emb.unsqueeze(1).expand(batch, n_arms, -1)
        eps    = self.score_net(
            noisy_w.reshape(batch * n_arms),
            z.reshape(batch * n_arms, z.size(-1)),
            t_exp.reshape(batch * n_arms, t_emb.size(-1)),
        )
        return eps.view(batch, n_arms)

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
    def sample(self, z: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        if num_samples <= 1:
            return self._sample_once(z)
        return torch.stack([self._sample_once(z) for _ in range(num_samples)], dim=1)

    @torch.no_grad()
    def _sample_once(self, z: torch.Tensor) -> torch.Tensor:
        w = torch.randn((z.size(0), z.size(1)), device=z.device)
        for step in reversed(range(self.diffusion_steps)):
            w = self.p_sample(w, z, step)
        return torch.clamp(w, -self.score_clip, self.score_clip)


# ── Twin critic ────────────────────────────────────────────────────────────

class ArmCriticNet(nn.Module):
    def __init__(self, z_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_i: torch.Tensor, action_i: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_i, action_i.unsqueeze(-1)], dim=-1)).squeeze(-1)


class PerArmTwinCritic(nn.Module):
    def __init__(self, n_arms: int, z_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.n_arms = int(n_arms)
        self.q1_net = ArmCriticNet(z_dim=z_dim, hidden_dim=hidden_dim)
        self.q2_net = ArmCriticNet(z_dim=z_dim, hidden_dim=hidden_dim)

    def forward(self, z: torch.Tensor,
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action = action.float()
        batch, n_arms, z_dim = z.shape
        fz = z.reshape(batch * n_arms, z_dim)
        fa = action.reshape(batch * n_arms)
        q1 = self.q1_net(fz, fa).view(batch, n_arms)
        q2 = self.q2_net(fz, fa).view(batch, n_arms)
        return q1, q2

    def min_q(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(z, action)
        return torch.min(q1, q2)
