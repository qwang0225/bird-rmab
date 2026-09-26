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
  PPOPolicy               -- evaluation wrapper for trained PPO network
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env import (MIMICRMABConfig, OBS_DIM, TYPE_PARAMS,
                 COUPLING_C_RH, COUPLING_C_HR, REWARD_WEIGHTS,
                 LOADING_MATRIX, OBS_SIGMA)


_TYPE_PARAMS = np.asarray(TYPE_PARAMS, dtype=np.float64)
_ALPHA_TYPES = _TYPE_PARAMS[:, 0:2]
_BETA_TYPES = _TYPE_PARAMS[:, 2:4]
_DRIFT_TYPES = _TYPE_PARAMS[:, 4:6]
_SIGMA_TYPES = _TYPE_PARAMS[:, 6:8]


def _effective_sample_size(weights: np.ndarray) -> np.ndarray:
    return 1.0 / np.sum(weights ** 2, axis=1)


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


class ActivationGreedy:
    """
    Observation-only one-step activation-advantage greedy baseline.

    Estimates the 2D latent state from observed vitals by least squares under
    the population observation model, then ranks arms by immediate marginal
    value of activation using population-average dynamics.
    """

    def __init__(self, cfg: MIMICRMABConfig):
        self.cfg = cfg
        self.alpha = _ALPHA_TYPES.mean(axis=0)
        self.beta = _BETA_TYPES.mean(axis=0)
        self.drift = _DRIFT_TYPES.mean(axis=0)
        self.loading_pinv = np.linalg.pinv(LOADING_MATRIX.T)

    def reset(self): pass

    def act(self, obs: np.ndarray, info: dict = None) -> np.ndarray:
        y = np.asarray(obs[:, :5], dtype=np.float64)
        x_hat = y @ self.loading_pinv
        coupling = np.column_stack([
            COUPLING_C_RH * x_hat[:, 1],
            COUPLING_C_HR * x_hat[:, 0],
        ])
        x_active = self.alpha[None, :] * x_hat + coupling + self.beta[None, :] + self.drift[None, :]
        x_passive = self.alpha[None, :] * x_hat + coupling + self.drift[None, :]
        r_active = np.maximum(x_active @ REWARD_WEIGHTS.astype(np.float64), 0.0)
        r_passive = np.maximum(x_passive @ REWARD_WEIGHTS.astype(np.float64), 0.0)
        return _topk_action(r_active - r_passive, self.cfg.K)


class PFRolloutPolicy:
    """
    Model-informed, non-oracle particle-filter rollout baseline.

    The policy maintains per-arm particles over hidden patient type, 2D latent
    health, and drifting alpha/beta. It uses only observed vitals and previous
    actions online. It is given the population simulator family, but does not
    access realized x_true, theta_true, alpha_true, or beta_true.
    """

    def __init__(self, cfg: MIMICRMABConfig, n_particles: int = 200,
                 H: int = 10, gamma: float = 0.99, ess_frac: float = 0.5,
                 seed: int = 0):
        self.cfg = cfg
        self.S = int(n_particles)
        self.H = int(H)
        self.gamma = float(gamma)
        self.ess_thresh = float(ess_frac) * self.S
        self.rng = np.random.default_rng(seed)
        self.theta_p = None
        self.x_p = None
        self.alpha_p = None
        self.beta_p = None
        self.w_p = None
        self.prev_action = None
        self._t = 0

    def reset(self, **_):
        cfg = self.cfg
        self.theta_p = self.rng.integers(0, cfg.M, size=(cfg.N, self.S))
        self.x_p = (
            cfg.init_mean
            + cfg.init_std * self.rng.standard_normal((cfg.N, self.S, 2))
        )
        self.alpha_p = _ALPHA_TYPES[self.theta_p].copy()
        self.beta_p = _BETA_TYPES[self.theta_p].copy()
        self.w_p = np.full((cfg.N, self.S), 1.0 / self.S, dtype=np.float64)
        self.prev_action = np.zeros(cfg.N, dtype=np.int32)
        self._t = 0

    def _propagate(self, action: np.ndarray) -> None:
        cfg = self.cfg
        drift = _DRIFT_TYPES[self.theta_p]
        sigma = _SIGMA_TYPES[self.theta_p]
        coupling = np.stack(
            [
                COUPLING_C_RH * self.x_p[..., 1],
                COUPLING_C_HR * self.x_p[..., 0],
            ],
            axis=-1,
        )
        beta_effect = self.beta_p * action[:, None, None]
        noise = sigma * self.rng.standard_normal(self.x_p.shape)
        self.x_p = self.alpha_p * self.x_p + coupling + beta_effect + drift + noise

        alpha_bar = _ALPHA_TYPES[self.theta_p]
        beta_bar = _BETA_TYPES[self.theta_p]
        self.alpha_p = np.clip(
            alpha_bar + cfg.ou_rho * (self.alpha_p - alpha_bar)
            + cfg.sigma_alpha * self.rng.standard_normal(self.alpha_p.shape),
            cfg.alpha_lo, cfg.alpha_hi,
        )
        self.beta_p = np.clip(
            beta_bar + cfg.ou_rho * (self.beta_p - beta_bar)
            + cfg.sigma_beta * self.rng.standard_normal(self.beta_p.shape),
            cfg.beta_lo, cfg.beta_hi,
        )

    def _reweight(self, obs: np.ndarray) -> None:
        pred = np.einsum("nsd,kd->nsk", self.x_p, LOADING_MATRIX)
        diff = (obs[:, None, :] - pred) / OBS_SIGMA[None, None, :]
        loglik = -0.5 * np.sum(diff ** 2, axis=2)
        loglik -= loglik.max(axis=1, keepdims=True)
        weights = self.w_p * np.exp(loglik)
        denom = weights.sum(axis=1, keepdims=True)
        denom = np.where(denom <= 0.0, 1.0, denom)
        self.w_p = weights / denom

    def _resample_if_needed(self) -> None:
        ess = _effective_sample_size(self.w_p)
        for i in range(self.cfg.N):
            if ess[i] < self.ess_thresh:
                idx = self.rng.choice(self.S, size=self.S, replace=True, p=self.w_p[i])
                self.theta_p[i] = self.theta_p[i, idx]
                self.x_p[i] = self.x_p[i, idx]
                self.alpha_p[i] = self.alpha_p[i, idx]
                self.beta_p[i] = self.beta_p[i, idx]
                self.w_p[i] = 1.0 / self.S

    def _rollout_scores(self) -> np.ndarray:
        cfg = self.cfg
        alpha_bar = _ALPHA_TYPES[self.theta_p]
        beta_bar = _BETA_TYPES[self.theta_p]
        drift = _DRIFT_TYPES[self.theta_p]
        alpha = self.alpha_p.copy()
        beta = self.beta_p.copy()
        x_active = self.x_p.copy()
        x_passive = self.x_p.copy()
        value = np.zeros((cfg.N, self.S), dtype=np.float64)
        rw = REWARD_WEIGHTS

        for h in range(self.H):
            coup_a = np.stack(
                [COUPLING_C_RH * x_active[..., 1], COUPLING_C_HR * x_active[..., 0]],
                axis=-1,
            )
            coup_p = np.stack(
                [COUPLING_C_RH * x_passive[..., 1], COUPLING_C_HR * x_passive[..., 0]],
                axis=-1,
            )
            beta_eff = beta * (1.0 if h == 0 else 0.0)
            x_active = alpha * x_active + coup_a + beta_eff + drift
            x_passive = alpha * x_passive + coup_p + drift
            r_active = np.maximum(rw[0] * x_active[..., 0] + rw[1] * x_active[..., 1], 0.0)
            r_passive = np.maximum(rw[0] * x_passive[..., 0] + rw[1] * x_passive[..., 1], 0.0)
            value += (self.gamma ** (h + 1)) * (r_active - r_passive)

            alpha = np.clip(alpha_bar + cfg.ou_rho * (alpha - alpha_bar),
                            cfg.alpha_lo, cfg.alpha_hi)
            beta = np.clip(beta_bar + cfg.ou_rho * (beta - beta_bar),
                           cfg.beta_lo, cfg.beta_hi)

        return np.sum(self.w_p * value, axis=1)

    def act(self, obs: np.ndarray, info: dict = None) -> np.ndarray:
        if self._t > 0:
            self._propagate(self.prev_action)
        self._reweight(obs)
        self._resample_if_needed()
        scores = self._rollout_scores()
        action = np.zeros(self.cfg.N, dtype=np.int64)
        action[np.argsort(-scores)[:self.cfg.K]] = 1
        self.prev_action = action
        self._t += 1
        return action


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
# PPO baseline
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# PPO: network + evaluation policy
# ---------------------------------------------------------------------------

from ppo import PPOAgent, PPOConfig


class PPOPolicy(PPOAgent):
    """Shared per-arm MLP PPO policy; weights transfer across population sizes."""

    def reset(self):
        self.reset_history()

    def act(self, obs: np.ndarray, info: dict = None) -> np.ndarray:
        return self.act_hard(obs)

    @classmethod
    def load(cls, path: str, N: int, K: int, obs_dim: int = OBS_DIM,
             device: str = "cpu") -> "PPOPolicy":
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not all(key in checkpoint for key in ("encoder", "actor", "critic")):
            raise ValueError("PPO requires shared per-arm MLP weights; retrain the observation-only checkpoint.")
        saved = checkpoint.get("cfg", {})
        cfg = PPOConfig()
        for key, value in saved.items():
            if key in cfg.__dataclass_fields__ and key not in {"N", "K", "device", "obs_dim"}:
                setattr(cfg, key, value)
        if saved.get("obs_dim", obs_dim) != obs_dim:
            raise ValueError("PPO observation dimension does not match the environment")
        cfg.N, cfg.K, cfg.obs_dim, cfg.device = N, K, obs_dim, device
        agent = cls(N, K, cfg)
        agent.load_checkpoint(path)
        agent.encoder.eval()
        agent.actor.eval()
        agent.critic.eval()
        return agent
