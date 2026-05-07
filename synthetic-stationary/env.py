"""
env.py  (adapt)

Bayesian-Adaptive RMAB: each arm independently different.

Each arm i has:
  - Hidden type  theta_i in {0,...,M-1}  sampled independently, UNKNOWN to agent
  - Time-varying alpha_i(t), beta_i(t)   OU around type mean, UNKNOWN to agent
  - Continuous state x_i(t)

Dynamics:
  x_{i,t+1} = alpha_i(t)*x_i(t) + beta_i(t)*a_i(t) - drift + w_i(t)
  alpha_i(t+1) = clip(alpha_bar[theta_i] + ou_rho*(alpha_i(t)-alpha_bar[theta_i])
                      + sigma_alpha*eps, alpha_lo, alpha_hi)
  beta_i(t+1)  = clip(beta_bar[theta_i]  + ou_rho*(beta_i(t)-beta_bar[theta_i])
                      + sigma_beta*eps,  beta_lo,  beta_hi)

Observation: y_i(t) = x_i(t) + v_i(t),  v_i ~ N(0, sigma_v^2)

Reward: r_i(t) = max(x_i(t), 0)

Types (M=4, more types -> harder classification):
  0  "dead"        alpha_bar=0.95  beta_bar=0.20  active_SS~+0.2   (rarely worth it)
  1  "stubborn"    alpha_bar=0.92  beta_bar=0.80  active_SS~+6.2
  2  "normal"      alpha_bar=0.88  beta_bar=1.80  active_SS~+12.5
  3  "responsive"  alpha_bar=0.82  beta_bar=3.20  active_SS~+16.1

High stochasticity parameters (chosen to break NeurWIN's index):
  - sigma_w = 0.80   large process noise -> state jumps each step
  - sigma_v = 0.80   large obs noise -> hard to read current state
  - sigma_beta = 0.40  large beta drift -> responsiveness changes meaningfully
  - sigma_alpha = 0.05 moderate alpha drift
  - ou_rho = 0.70   faster reversion -> parameters change faster

Why NeurWIN breaks:
  - w_i(t) = IndexNet(y_i) maps a noisy scalar to an index
  - With sigma_v=0.80 and sigma_w=0.80, two arms of different types can have
    identical y_i at any given step -> index cannot distinguish them
  - With beta_i drifting by +-0.40 per step, an arm's responsiveness changes
    enough that a fixed index trained on average behavior is often wrong
  - NeurWIN sees only current obs; cannot integrate history to infer theta_i

Why BeliefEncoder + Diffusion helps:
  - Transformer over L=40 steps of (y_i, a_i) history -> z_i
  - z_i encodes inferred distribution over (theta_i, alpha_i(t), beta_i(t))
  - Diffusion actor samples from p(scores | z_1,...,z_N) capturing uncertainty
    in which arms are currently worth activating given noisy type estimates
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

# Per-type base parameters: (alpha_bar, beta_bar)
TYPE_ALPHA_MEAN = [0.95, 0.92, 0.88, 0.82]
TYPE_BETA_MEAN  = [0.20, 0.80, 1.80, 3.20]

# Human-readable type names
TYPE_NAMES = ["dead", "stubborn", "normal", "responsive"]


@dataclass
class AdaptRMABConfig:
    N: int = 20
    K: int = 5
    T: int = 100

    M: int = 4   # number of hidden arm types

    # alpha_i, beta_i are fixed at type means for the entire episode (stationary POMDP)
    # Hidden type theta_i is the only unknown — belief encoder must infer it from history

    # Dynamics noise -- moderate: challenging but not impossible to learn from
    drift:   float = 0.30
    sigma_w: float = 0.40   # meaningful process noise
    sigma_v: float = 0.50   # meaningful obs noise; single obs unreliable, history helps

    # Initial state
    init_mean: float = 0.0
    init_std:  float = 2.0


class AdaptRMABEnv:
    """
    Each arm is independently drawn from M types with its own
    drifting (alpha_i, beta_i). No group structure.

    info always contains x_true, theta_true, alpha_true, beta_true.
    """

    def __init__(self, cfg: AdaptRMABConfig, seed: int = 0):
        self.cfg   = cfg
        self.rng   = np.random.default_rng(seed)
        self.x:     np.ndarray | None = None
        self.theta: np.ndarray | None = None
        self.alpha: np.ndarray | None = None
        self.beta:  np.ndarray | None = None
        self.t:     int = 0

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        cfg = self.cfg
        self.t = 0

        # Each arm independently draws its hidden type
        self.theta = self.rng.integers(0, cfg.M, size=cfg.N)

        # Initialize parameters at type means
        self.alpha = np.array([TYPE_ALPHA_MEAN[m] for m in self.theta], dtype=np.float32)
        self.beta  = np.array([TYPE_BETA_MEAN[m]  for m in self.theta], dtype=np.float32)

        self.x = (cfg.init_mean
                  + cfg.init_std * self.rng.standard_normal(cfg.N)).astype(np.float32)

        return self._observe(), self._make_info()

    def step(self, a: np.ndarray):
        a = np.asarray(a, dtype=np.int32).reshape(-1)
        cfg = self.cfg
        assert a.shape[0] == cfg.N
        assert int(a.sum()) <= cfg.K, f"budget violated: {int(a.sum())} > {cfg.K}"

        reward_vec = np.maximum(self.x, 0.0).astype(np.float32)

        # State transition
        w      = (cfg.sigma_w * self.rng.standard_normal(cfg.N)).astype(np.float32)
        self.x = (self.alpha * self.x
                  + self.beta * a.astype(np.float32)
                  - cfg.drift + w)

        # alpha_i, beta_i stay fixed — no OU update
        self.t += 1
        done = self.t >= cfg.T
        info = self._make_info()
        info["reward_per_arm"] = reward_vec
        return self._observe(), reward_vec, done, info

    def _observe(self) -> np.ndarray:
        v = (self.cfg.sigma_v * self.rng.standard_normal(self.cfg.N)).astype(np.float32)
        return self.x + v

    def _make_info(self) -> dict:
        return {
            "t":          self.t,
            "x_true":     self.x.copy(),
            "theta_true": self.theta.copy(),
            "alpha_true": self.alpha.copy(),
            "beta_true":  self.beta.copy(),
        }
