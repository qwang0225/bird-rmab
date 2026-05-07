"""
env.py  (mimic-icu v3)

MIMIC-ICU POMDP RMAB: 2D latent health state, 5D observation, M=4 types.

Each arm i is one ICU patient with:
  - Hidden type  theta_i in {0,1,2,3}  (septic_shock / resp_failure / hemo_instability / recovering)
  - 2D latent health  x_i(t) = [x_hemo, x_resp]  (high=healthy, low=sick)
  - 5D noisy observation  y_i(t) = LOADING_MATRIX @ x_i + OBS_SIGMA * noise
  - Time-varying treatment effect  beta_i(t) = [beta_hemo, beta_resp]  (OU around type mean)

Latent dynamics:
  x_hemo(t+1) = alpha_hemo * x_hemo + C_RH * x_resp + beta_hemo * a + drift_hemo + w
  x_resp(t+1) = alpha_resp * x_resp + C_HR * x_hemo + beta_resp * a + drift_resp + w
  alpha_i(t), beta_i(t) drift via OU around type-specific means.

Reward:  r_i(t) = max(0.5*x_hemo_i + 0.5*x_resp_i, 0)

M=4 patient types (alpha_hemo, alpha_resp,
                   beta_hemo, beta_resp,
                   drift_hemo, drift_resp,
                   sigma_hemo, sigma_resp):
  Type 0 "septic_shock"       — both deteriorating, poor treatment ROI
  Type 1 "resp_failure"       — resp deteriorating, high resp treatment response
  Type 2 "hemo_instability"   — hemo deteriorating, high hemo treatment response
  Type 3 "recovering"         — mild positive drift, moderate treatment response
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

# ── Observation / state dimensions ───────────────────────────────────────────
OBS_DIM   = 5   # 5 always-observed vitals: HR, SBP, MBP, SpO2, RR
STATE_DIM = 2   # hemo, resp

# ── Loading matrix: (5 vitals) × (2 latent dims) ─────────────────────────────
# y = LOADING_MATRIX @ x + OBS_SIGMA * noise
# Fitted from MIMIC factor analysis (fit_mimic_params_v2.py).
LOADING_MATRIX = np.array([
    [ 0.4766, -0.1852],   # HR:   hemodynamic primary, negative resp (tachycardia)
    [ 1.1006,  0.0655],   # SBP:  hemodynamic
    [ 1.3883,  0.1199],   # MBP:  hemodynamic
    [ 0.0850,  0.7815],   # SpO2: respiratory
    [-0.0850,  1.2185],   # RR:   respiratory
], dtype=np.float32)

OBS_SIGMA = np.array([0.60, 0.70, 0.65, 0.35, 0.90], dtype=np.float32)

# ── Reward weights ────────────────────────────────────────────────────────────
REWARD_WEIGHTS = np.array([0.5, 0.5], dtype=np.float32)   # w_hemo, w_resp

# ── M=4 type parameters ───────────────────────────────────────────────────────
# Each tuple: (alpha_hemo, alpha_resp,
#              beta_hemo, beta_resp,
#              drift_hemo, drift_resp,
#              sigma_hemo, sigma_resp)
#
# alpha/sigma: fitted from MIMIC AR(1) cluster centroids (fit_mimic_params_v2.py).
# beta: designed — observational data gives beta≈0 (Rubin 1974).
# drift: designed — fitted drifts have survival bias; we impose realistic deterioration.
TYPE_PARAMS = [
    # Type 0: septic shock — both deteriorating, poor treatment ROI
    (0.455, 0.645,   0.20, 0.15,   -0.120, -0.100,   0.388, 0.364),
    # Type 1: resp failure — resp deteriorating, high resp treatment response
    (0.653, 0.400,   0.25, 1.80,   -0.020, -0.160,   0.350, 0.370),
    # Type 2: hemo instability — hemo deteriorating, high hemo treatment response
    (0.400, 0.400,   1.80, 0.25,   -0.160, -0.020,   0.356, 0.362),
    # Type 3: recovering — mild positive drift, moderate treatment response
    (0.424, 0.400,   0.90, 0.90,   +0.080, +0.070,   0.386, 0.398),
]
TYPE_NAMES = ["septic_shock", "resp_failure", "hemo_instability", "recovering"]

# ── Cross-system coupling (population-level) ──────────────────────────────────
COUPLING_C_RH = 0.04   # resp → hemo
COUPLING_C_HR = 0.04   # hemo → resp


@dataclass
class MIMICRMABConfig:
    N: int = 20
    K: int = 5
    T: int = 100
    M: int = 4

    obs_dim: int = OBS_DIM

    # OU drift on per-arm alpha (2D) and beta (2D)
    ou_rho:      float = 0.85
    sigma_alpha: float = 0.03
    sigma_beta:  float = 0.15

    alpha_lo: float = 0.30
    alpha_hi: float = 0.95
    beta_lo:  float = 0.00
    beta_hi:  float = 5.00

    init_mean: float = 0.0
    init_std:  float = 2.0


class MIMICRMABEnv:
    """
    N-arm POMDP RMAB with M=4 ICU patient types and 2D latent health state.

    obs shape:  (N, obs_dim=5)
    reward:     (N,) per-arm health surplus

    info always contains x_true (N,2), theta_true (N,), alpha_true (N,2), beta_true (N,2).
    """

    def __init__(self, cfg: MIMICRMABConfig, seed: int = 0):
        self.cfg   = cfg
        self.rng   = np.random.default_rng(seed)
        self.x:     np.ndarray | None = None   # (N, 2) [hemo, resp]
        self.theta: np.ndarray | None = None   # (N,)
        self.alpha: np.ndarray | None = None   # (N, 2) [alpha_hemo, alpha_resp]
        self.beta:  np.ndarray | None = None   # (N, 2) [beta_hemo, beta_resp]
        self.t:     int = 0

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        cfg = self.cfg
        self.t = 0

        self.theta = self.rng.integers(0, cfg.M, size=cfg.N)

        self.alpha = np.array(
            [[TYPE_PARAMS[m][0], TYPE_PARAMS[m][1]] for m in self.theta],
            dtype=np.float32)   # (N, 2)
        self.beta = np.array(
            [[TYPE_PARAMS[m][2], TYPE_PARAMS[m][3]] for m in self.theta],
            dtype=np.float32)   # (N, 2)

        self.x = (cfg.init_mean
                  + cfg.init_std * self.rng.standard_normal((cfg.N, 2))
                  ).astype(np.float32)

        return self._observe(), self._make_info()

    def step(self, a: np.ndarray):
        a   = np.asarray(a, dtype=np.int32).reshape(-1)
        cfg = self.cfg
        assert a.shape[0] == cfg.N
        assert int(a.sum()) <= cfg.K, f"budget violated: {int(a.sum())} > {cfg.K}"

        # Reward before transition
        reward_vec = np.maximum(
            REWARD_WEIGHTS[0] * self.x[:, 0] + REWARD_WEIGHTS[1] * self.x[:, 1],
            0.0
        ).astype(np.float32)

        # Per-arm type parameters
        drift = np.array(
            [[TYPE_PARAMS[m][4], TYPE_PARAMS[m][5]] for m in self.theta],
            dtype=np.float32)   # (N, 2)
        sigma_w = np.array(
            [[TYPE_PARAMS[m][6], TYPE_PARAMS[m][7]] for m in self.theta],
            dtype=np.float32)   # (N, 2)

        # Cross-system coupling
        coupling = np.column_stack([
            COUPLING_C_RH * self.x[:, 1],   # resp → hemo
            COUPLING_C_HR * self.x[:, 0],   # hemo → resp
        ]).astype(np.float32)

        # Treatment effect
        beta_effect = np.column_stack([
            self.beta[:, 0] * a,
            self.beta[:, 1] * a,
        ]).astype(np.float32)

        # Noise
        w = (sigma_w * self.rng.standard_normal((cfg.N, 2))).astype(np.float32)

        # State transition
        self.x = (self.alpha * self.x + coupling + beta_effect + drift + w).astype(np.float32)

        # OU drift on per-arm parameters
        alpha_bar = np.array(
            [[TYPE_PARAMS[m][0], TYPE_PARAMS[m][1]] for m in self.theta],
            dtype=np.float32)
        beta_bar = np.array(
            [[TYPE_PARAMS[m][2], TYPE_PARAMS[m][3]] for m in self.theta],
            dtype=np.float32)

        self.alpha = np.clip(
            alpha_bar + cfg.ou_rho * (self.alpha - alpha_bar)
            + cfg.sigma_alpha * self.rng.standard_normal((cfg.N, 2)).astype(np.float32),
            cfg.alpha_lo, cfg.alpha_hi).astype(np.float32)
        self.beta = np.clip(
            beta_bar + cfg.ou_rho * (self.beta - beta_bar)
            + cfg.sigma_beta * self.rng.standard_normal((cfg.N, 2)).astype(np.float32),
            cfg.beta_lo, cfg.beta_hi).astype(np.float32)

        self.t += 1
        done = self.t >= cfg.T
        info = self._make_info()
        info["reward_per_arm"] = reward_vec
        return self._observe(), reward_vec, done, info

    def _observe(self) -> np.ndarray:
        """Returns (N, 5): 5 vitals from loading matrix + noise."""
        return (
            self.x @ LOADING_MATRIX.T
            + OBS_SIGMA[None, :] * self.rng.standard_normal((self.cfg.N, 5))
        ).astype(np.float32)

    def _make_info(self) -> dict:
        return {
            "t":          self.t,
            "x_true":     self.x.copy(),       # (N, 2)
            "theta_true": self.theta.copy(),    # (N,)
            "alpha_true": self.alpha.copy(),    # (N, 2)
            "beta_true":  self.beta.copy(),     # (N, 2)
        }
