"""
baselines.py  (mimic-icu v3)

Policy classes only; no training code.

Policies:
  RandomPolicy            -- random K arms each step
  GreedyWorstObs          -- treat K arms with lowest mean vital score
  OracleLookahead         -- knows x_true, alpha, beta, theta; H-step lookahead
  IndexNet                -- MLP: obs_dim -> scalar index
  NeurWINPolicy           -- evaluation wrapper for trained IndexNet
  BeliefIndexNet          -- MLP: z_dim -> scalar index
  NeurWINEncoderPolicy    -- evaluation wrapper for BeliefEncoder + BeliefIndexNet
  PPOConfig
  _PPONet
  PPOPolicy               -- evaluation wrapper for trained PPO network
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from env import (MIMICRMABConfig, OBS_DIM, TYPE_PARAMS,
                 COUPLING_C_RH, COUPLING_C_HR, REWARD_WEIGHTS)


def _topk_action(scores: np.ndarray, K: int) -> np.ndarray:
    a = np.zeros(len(scores), dtype=np.int64)
    a[np.argsort(scores)[-K:]] = 1
    return a


# ---------------------------------------------------------------------------
# Random
# ---------------------------------------------------------------------------

class RandomPolicy:
    def __init__(self, N: int, K: int, seed: int = 0):
        self.N = N; self.K = K
        self.rng = np.random.default_rng(seed)

    def reset(self): pass

    def act(self, obs: np.ndarray, info: dict = None) -> np.ndarray:
        a = np.zeros(self.N, dtype=np.int64)
        a[self.rng.choice(self.N, size=min(self.K, self.N), replace=False)] = 1
        return a


# ---------------------------------------------------------------------------
# Greedy worst obs
# ---------------------------------------------------------------------------

class GreedyWorstObs:
    """Treat K arms with lowest mean over the first 5 (always-observed) vitals."""

    def __init__(self, N: int, K: int):
        self.N = N; self.K = K

    def reset(self): pass

    def act(self, obs: np.ndarray, info: dict = None) -> np.ndarray:
        # obs[:, :5] = always-observed vitals; obs[:, 5:] = sparse meta + masks
        return _topk_action(-obs[:, :5].mean(axis=1), self.K)


# ---------------------------------------------------------------------------
# Oracle Greedy (fairer upper bound; knows x_true but no lookahead)
# ---------------------------------------------------------------------------

class OracleGreedy:
    """
    Greedy oracle: knows x_true (N,2) but no lookahead.
    Ranks arms by current reward proxy: 0.5*x_hemo + 0.5*x_resp.
    Fairer upper bound than OracleLookahead for gap reporting.
    """

    def __init__(self, N: int, K: int):
        self.N = N; self.K = K

    def reset(self): pass

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        x = info["x_true"]   # (N, 2)
        scores = 0.5 * x[:, 0] + 0.5 * x[:, 1]
        return _topk_action(scores, self.K)


# ---------------------------------------------------------------------------
# Oracle Lookahead (true upper bound; requires info dict from env)
# ---------------------------------------------------------------------------

class OracleLookahead:
    """
    H-step deterministic lookahead per arm with 2D latent state.
    Requires info["x_true"] (N,2), info["alpha_true"] (N,2),
              info["beta_true"] (N,2), info["theta_true"] (N,).
    Uses OU mean reversion (eps=0); activates top-K by marginal value.
    """

    def __init__(self, cfg: MIMICRMABConfig, H: int = 10, gamma: float = 0.99):
        self.cfg = cfg; self.H = H; self.gamma = gamma

    def reset(self): pass

    def _marginal(self, x0: np.ndarray, alpha0: np.ndarray, beta0: np.ndarray,
                  theta: int) -> float:
        """
        x0:     (2,) initial [x_hemo, x_resp]
        alpha0: (2,) initial [alpha_hemo, alpha_resp]
        beta0:  (2,) initial [beta_hemo, beta_resp]
        """
        cfg = self.cfg
        tp  = TYPE_PARAMS[theta]
        alpha_bar = np.array([tp[0], tp[1]], dtype=np.float64)
        beta_bar  = np.array([tp[2], tp[3]], dtype=np.float64)
        drift     = np.array([tp[4], tp[5]], dtype=np.float64)

        x_a = x0.astype(np.float64).copy()
        x_p = x0.astype(np.float64).copy()
        al  = alpha0.astype(np.float64).copy()
        be  = beta0.astype(np.float64).copy()
        val = 0.0

        for h in range(self.H):
            al = np.clip(alpha_bar + cfg.ou_rho * (al - alpha_bar),
                         cfg.alpha_lo, cfg.alpha_hi)
            be = np.clip(beta_bar  + cfg.ou_rho * (be - beta_bar),
                         cfg.beta_lo, cfg.beta_hi)

            coup_a = np.array([COUPLING_C_RH * x_a[1], COUPLING_C_HR * x_a[0]])
            coup_p = np.array([COUPLING_C_RH * x_p[1], COUPLING_C_HR * x_p[0]])
            beta_eff = be * (1.0 if h == 0 else 0.0)

            x_a = al * x_a + coup_a + beta_eff + drift
            x_p = al * x_p + coup_p + drift

            r_a = max(float(REWARD_WEIGHTS[0]) * x_a[0]
                      + float(REWARD_WEIGHTS[1]) * x_a[1], 0.0)
            r_p = max(float(REWARD_WEIGHTS[0]) * x_p[0]
                      + float(REWARD_WEIGHTS[1]) * x_p[1], 0.0)
            val += (self.gamma ** (h + 1)) * (r_a - r_p)

        return val

    def act(self, obs: np.ndarray, info: dict = None) -> np.ndarray:
        scores = np.array([
            self._marginal(
                info["x_true"][i],       # (2,)
                info["alpha_true"][i],   # (2,)
                info["beta_true"][i],    # (2,)
                int(info["theta_true"][i]),
            )
            for i in range(self.cfg.N)
        ])
        return _topk_action(scores, self.cfg.K)


# ---------------------------------------------------------------------------
# NeurWIN (memoryless): IndexNet MLP + evaluation policy
# ---------------------------------------------------------------------------

class IndexNet(nn.Module):
    """MLP: (batch, N, obs_dim) -> (batch, N) scalar index per arm (memoryless)."""

    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, N, D = x.shape
        return self.net(x.reshape(b * N, D)).reshape(b, N)


class NeurWINPolicy:
    def __init__(self, net: IndexNet, K: int, device: torch.device):
        self.net = net; self.K = K; self.device = device

    def reset(self): pass

    @torch.no_grad()
    def act(self, obs: np.ndarray, info: dict = None) -> np.ndarray:
        o = torch.from_numpy(obs[None]).float().to(self.device)
        return _topk_action(self.net(o)[0].cpu().numpy(), self.K)

    def act_hard(self, obs: np.ndarray, **_) -> np.ndarray:
        return self.act(obs)

    @classmethod
    def load(cls, path: str, N: int, K: int, obs_dim: int = OBS_DIM,
             hidden: int = 64, device: str = "cpu") -> "NeurWINPolicy":
        dev  = torch.device(device)
        net  = IndexNet(obs_dim, hidden).to(dev)
        ckpt = torch.load(path, map_location=dev)
        state = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
        net.load_state_dict(state)
        net.eval()
        return cls(net, K, dev)


# ---------------------------------------------------------------------------
# NeurWIN with encoder: BeliefIndexNet + evaluation policy
# ---------------------------------------------------------------------------

class BeliefIndexNet(nn.Module):
    """Per-arm MLP: z_i (belief embedding) -> scalar Whittle index."""

    def __init__(self, z_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (batch, N, z_dim) -> indices: (batch, N)"""
        b, N, D = z.shape
        return self.net(z.reshape(b * N, D)).reshape(b, N)


@dataclass
class NeurWINEncoderConfig:
    obs_dim:        int   = OBS_DIM
    L:              int   = 40
    z_dim:          int   = 64
    encoder_hidden: int   = 128
    encoder_heads:  int   = 4
    encoder_layers: int   = 3
    index_hidden:   int   = 64
    seed:           int   = 0
    device:         str   = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir:       str   = "checkpoints_neurwin"


class NeurWINEncoderPolicy:
    """Evaluation-only wrapper for BeliefEncoder + BeliefIndexNet. Use train_neurwin.py to train."""

    def __init__(self, N: int, K: int, cfg: NeurWINEncoderConfig):
        self.N = N; self.K = K; self.cfg = cfg
        self.device = torch.device(cfg.device)
        from diffusion_DPMD_train import BeliefEncoder
        self.encoder   = BeliefEncoder(
            obs_dim=cfg.obs_dim, z_dim=cfg.z_dim, hidden_dim=cfg.encoder_hidden,
            n_heads=cfg.encoder_heads, n_layers=cfg.encoder_layers, L=cfg.L,
        ).to(self.device)
        self.index_net = BeliefIndexNet(cfg.z_dim, cfg.index_hidden).to(self.device)
        self._obs_hist: np.ndarray | None = None
        self._act_hist: np.ndarray | None = None

    def reset_history(self):
        self._obs_hist = np.zeros((self.N, self.cfg.L, self.cfg.obs_dim), dtype=np.float32)
        self._act_hist = np.zeros((self.N, self.cfg.L), dtype=np.float32)

    @torch.no_grad()
    def act_hard(self, obs: np.ndarray, **_) -> np.ndarray:
        if self._obs_hist is None:
            self.reset_history()
        self._obs_hist = np.roll(self._obs_hist, -1, axis=1)
        self._obs_hist[:, -1, :] = obs
        oh = torch.from_numpy(self._obs_hist[None]).float().to(self.device)
        ah = torch.from_numpy(self._act_hist[None]).float().to(self.device)
        z  = self.encoder(oh, ah)
        scores = self.index_net(z)[0].cpu().numpy()
        action = _topk_action(scores, self.K)
        self._act_hist = np.roll(self._act_hist, -1, axis=1)
        self._act_hist[:, -1] = action.astype(np.float32)
        return action

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"], strict=False)
        self.index_net.load_state_dict(ckpt["index_net"], strict=False)

# ---------------------------------------------------------------------------
# PPO: network + evaluation policy
# ---------------------------------------------------------------------------

class _PPONet(nn.Module):
    def __init__(self, N: int, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(N * obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),      nn.Tanh(),
        )
        self.actor  = nn.Linear(hidden, N)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)


@dataclass
class PPOConfig:
    N:          int   = 20
    K:          int   = 5
    T:          int   = 72
    obs_dim:    int   = OBS_DIM
    hidden:     int   = 256
    lr:         float = 3e-4
    gamma:      float = 0.99
    clip_eps:   float = 0.2
    epochs:     int   = 200
    ppo_epochs: int   = 4
    batch_size: int   = 64
    seed:       int   = 0
    device:     str   = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir:   str   = "checkpoints_ppo"


class PPOPolicy:
    def __init__(self, net: _PPONet, N: int, K: int, obs_dim: int, device: torch.device):
        self.net = net; self.N = N; self.K = K
        self.obs_dim = obs_dim; self.device = device

    def reset(self): pass

    @torch.no_grad()
    def act(self, obs: np.ndarray, info: dict = None) -> np.ndarray:
        s = torch.from_numpy(obs.reshape(1, -1)).float().to(self.device)
        logits, _ = self.net(s)
        return _topk_action(torch.sigmoid(logits)[0].cpu().numpy(), self.K)

    def act_hard(self, obs: np.ndarray, **_) -> np.ndarray:
        return self.act(obs)

    @classmethod
    def load(cls, path: str, N: int, K: int, obs_dim: int = OBS_DIM,
             hidden: int = 256, device: str = "cpu") -> "PPOPolicy":
        dev  = torch.device(device)
        net  = _PPONet(N, obs_dim, hidden).to(dev)
        ckpt = torch.load(path, map_location=dev)
        state = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
        net.load_state_dict(state)
        net.eval()
        return cls(net, N, K, obs_dim, dev)
