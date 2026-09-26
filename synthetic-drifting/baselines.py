"""
baselines.py  (adapt)

RandomPolicy        -- activate K arms uniformly at random.
OracleGreedyPolicy  -- knows true (alpha_i, beta_i, x_i); ranks by active steady-state.
OracleLookaheadPolicy -- knows true (alpha_i, beta_i, x_i, theta_i); H-step deterministic
                         lookahead per arm, selects top-K by discounted marginal value.

Run standalone:
    python baselines.py
"""
from __future__ import annotations

import numpy as np
import time
from env import AdaptRMABConfig, AdaptRMABEnv, TYPE_ALPHA_MEAN, TYPE_BETA_MEAN, TYPE_NAMES


def _effective_sample_size(weights: np.ndarray) -> np.ndarray:
    return 1.0 / np.sum(weights ** 2, axis=1)


class RandomPolicy:
    def __init__(self, cfg: AdaptRMABConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def reset(self, **_):
        pass

    def act(self, obs: np.ndarray, **_) -> np.ndarray:
        a = np.zeros(self.cfg.N, dtype=np.int32)
        a[self.rng.choice(self.cfg.N, size=self.cfg.K, replace=False)] = 1
        return a


class OracleGreedyPolicy:
    """
    Greedy oracle: knows true (alpha_i, beta_i) and ranks by active steady-state.

        active_SS_i = (beta_i - drift) / (1 - alpha_i)

    This is the one-step myopic upper bound — optimal only if dynamics were stationary
    and the arm's immediate steady state were the right objective.
    """

    def __init__(self, cfg: AdaptRMABConfig):
        self.cfg = cfg

    def reset(self, **_):
        pass

    def act(self, obs: np.ndarray,
            alpha_true: np.ndarray, beta_true: np.ndarray, **_) -> np.ndarray:
        ss = (beta_true - self.cfg.drift) / (1.0 - alpha_true + 1e-8)
        a  = np.zeros(self.cfg.N, dtype=np.int32)
        a[np.argsort(-ss)[:self.cfg.K]] = 1
        return a


class OracleLookaheadPolicy:
    """
    H-step lookahead oracle: knows true (x_i, alpha_i, beta_i, theta_i).

    Per-arm marginal value of activating arm i now vs staying passive:

        V_i = sum_{h=1}^{H} gamma^h * (max(x_active_i(h), 0) - max(x_passive_i(h), 0))

    where both branches use deterministic rollout (process noise w=0):
      - active branch:  a_i=1 at h=0, a_i=0 for h=1..H-1
      - passive branch: a_i=0 throughout
      - alpha/beta evolve via OU mean reversion to known type means (eps=0)

    Activates top-K arms by V_i.  Strictly dominates OracleGreedyPolicy by
    accounting for how the current state x_i(t) shapes multi-step future rewards.
    """

    def __init__(self, cfg: AdaptRMABConfig, H: int = 10, gamma: float = 0.99):
        self.cfg   = cfg
        self.H     = H
        self.gamma = gamma

    def reset(self, **_):
        pass

    def _marginal(self, x0: float, a0: float, b0: float,
                  a_bar: float, b_bar: float) -> float:
        """H-step discounted marginal value of one activation at step 0."""
        cfg = self.cfg
        ou  = cfg.ou_rho
        x_a = x0;  x_p = x0
        a   = a0;  b   = b0
        val = 0.0
        for h in range(self.H):
            # OU mean reversion (deterministic, eps=0)
            a = float(np.clip(a_bar + ou * (a - a_bar), cfg.alpha_lo, cfg.alpha_hi))
            b = float(np.clip(b_bar + ou * (b - b_bar), cfg.beta_lo,  cfg.beta_hi))
            # State transition (w=0); action applied only at h=0
            x_a = a * x_a + b * (1.0 if h == 0 else 0.0) - cfg.drift
            x_p = a * x_p - cfg.drift
            val += (self.gamma ** (h + 1)) * (max(x_a, 0.0) - max(x_p, 0.0))
        return val

    def act(self, obs: np.ndarray,
            x_true: np.ndarray,
            alpha_true: np.ndarray,
            beta_true: np.ndarray,
            theta_true: np.ndarray, **_) -> np.ndarray:
        scores = np.array([
            self._marginal(
                float(x_true[i]),
                float(alpha_true[i]),
                float(beta_true[i]),
                TYPE_ALPHA_MEAN[int(theta_true[i])],
                TYPE_BETA_MEAN[int(theta_true[i])],
            )
            for i in range(self.cfg.N)
        ])
        a = np.zeros(self.cfg.N, dtype=np.int32)
        a[np.argsort(-scores)[:self.cfg.K]] = 1
        return a


class GreedyObsPolicy:
    """Ranks arms by noisy observation y_i — no oracle info, no history."""

    def __init__(self, cfg: AdaptRMABConfig):
        self.cfg = cfg

    def reset(self, **_):
        pass

    def act(self, obs: np.ndarray, **_) -> np.ndarray:
        a = np.zeros(self.cfg.N, dtype=np.int32)
        a[np.argsort(-obs)[:self.cfg.K]] = 1
        return a


class ActivationGreedyPolicy:
    """
    Observation-only one-step activation-advantage greedy baseline.

    Uses current noisy observation as the state estimate and population-average
    dynamics, then ranks arms by immediate marginal value of activation:
    max(x_active_next, 0) - max(x_passive_next, 0).
    """

    def __init__(self, cfg: AdaptRMABConfig):
        self.cfg = cfg
        self.alpha = float(np.mean(TYPE_ALPHA_MEAN))
        self.beta = float(np.mean(TYPE_BETA_MEAN))

    def reset(self, **_):
        pass

    def act(self, obs: np.ndarray, **_) -> np.ndarray:
        x_hat = np.asarray(obs, dtype=np.float64)
        x_active = self.alpha * x_hat + self.beta - self.cfg.drift
        x_passive = self.alpha * x_hat - self.cfg.drift
        scores = np.maximum(x_active, 0.0) - np.maximum(x_passive, 0.0)
        a = np.zeros(self.cfg.N, dtype=np.int32)
        a[np.argsort(-scores)[:self.cfg.K]] = 1
        return a


class PFRolloutPolicy:
    """
    Model-informed, non-oracle particle-filter rollout baseline.

    Particles track hidden type, latent state, and drifting alpha/beta. Online
    filtering uses only observations and previous actions. The policy is given
    population simulator parameters, but not realized simulator state/type or
    per-arm hidden parameters.
    """

    def __init__(self, cfg: AdaptRMABConfig, n_particles: int = 200,
                 H: int = 10, gamma: float = 0.99, ess_frac: float = 0.5,
                 seed: int = 0):
        self.cfg = cfg
        self.S = int(n_particles)
        self.H = int(H)
        self.gamma = float(gamma)
        self.ess_thresh = float(ess_frac) * self.S
        self.rng = np.random.default_rng(seed)
        self.alpha_types = np.asarray(TYPE_ALPHA_MEAN, dtype=np.float64)
        self.beta_types = np.asarray(TYPE_BETA_MEAN, dtype=np.float64)
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
            + cfg.init_std * self.rng.standard_normal((cfg.N, self.S))
        )
        self.alpha_p = self.alpha_types[self.theta_p].copy()
        self.beta_p = self.beta_types[self.theta_p].copy()
        self.w_p = np.full((cfg.N, self.S), 1.0 / self.S, dtype=np.float64)
        self.prev_action = np.zeros(cfg.N, dtype=np.int32)
        self._t = 0

    def _ou_update(self, alpha: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        alpha_bar = self.alpha_types[self.theta_p]
        beta_bar = self.beta_types[self.theta_p]
        alpha_next = np.clip(
            alpha_bar + cfg.ou_rho * (alpha - alpha_bar)
            + cfg.sigma_alpha * self.rng.standard_normal(alpha.shape),
            cfg.alpha_lo, cfg.alpha_hi,
        )
        beta_next = np.clip(
            beta_bar + cfg.ou_rho * (beta - beta_bar)
            + cfg.sigma_beta * self.rng.standard_normal(beta.shape),
            cfg.beta_lo, cfg.beta_hi,
        )
        return alpha_next, beta_next

    def _propagate(self, action: np.ndarray) -> None:
        cfg = self.cfg
        noise = cfg.sigma_w * self.rng.standard_normal(self.x_p.shape)
        self.x_p = self.alpha_p * self.x_p + self.beta_p * action[:, None] - cfg.drift + noise
        self.alpha_p, self.beta_p = self._ou_update(self.alpha_p, self.beta_p)

    def _reweight(self, obs: np.ndarray) -> None:
        cfg = self.cfg
        loglik = -0.5 * ((obs[:, None] - self.x_p) / cfg.sigma_v) ** 2
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
        alpha_bar = self.alpha_types[self.theta_p]
        beta_bar = self.beta_types[self.theta_p]
        alpha = self.alpha_p.copy()
        beta = self.beta_p.copy()
        x_active = self.x_p.copy()
        x_passive = self.x_p.copy()
        value = np.zeros_like(self.x_p, dtype=np.float64)
        for h in range(self.H):
            x_active = alpha * x_active + beta * (1.0 if h == 0 else 0.0) - cfg.drift
            x_passive = alpha * x_passive - cfg.drift
            value += (self.gamma ** (h + 1)) * (
                np.maximum(x_active, 0.0) - np.maximum(x_passive, 0.0)
            )
            alpha = np.clip(alpha_bar + cfg.ou_rho * (alpha - alpha_bar), cfg.alpha_lo, cfg.alpha_hi)
            beta = np.clip(beta_bar + cfg.ou_rho * (beta - beta_bar), cfg.beta_lo, cfg.beta_hi)
        return np.sum(self.w_p * value, axis=1)

    def act(self, obs: np.ndarray, **_) -> np.ndarray:
        if self._t > 0:
            self._propagate(self.prev_action)
        self._reweight(obs)
        self._resample_if_needed()
        scores = self._rollout_scores()
        action = np.zeros(self.cfg.N, dtype=np.int32)
        action[np.argsort(-scores)[:self.cfg.K]] = 1
        self.prev_action = action
        self._t += 1
        return action


def evaluate_policy(policy_name, policy, env_cfg, n_episodes, seed_offset=0,
                    return_timing: bool = False):
    returns = []
    action_time = 0.0
    action_count = 0
    for ep in range(n_episodes):
        env = AdaptRMABEnv(env_cfg, seed=seed_offset + ep)
        obs, info = env.reset()
        policy.reset()
        ep_return = 0.0
        for _ in range(env_cfg.T):
            t0 = time.perf_counter()
            if policy_name == "oracle_greedy":
                a = policy.act(obs, alpha_true=info["alpha_true"],
                               beta_true=info["beta_true"])
            elif policy_name == "oracle_lookahead":
                a = policy.act(obs, x_true=info["x_true"],
                               alpha_true=info["alpha_true"],
                               beta_true=info["beta_true"],
                               theta_true=info["theta_true"])
            else:
                a = policy.act(obs)
            action_time += time.perf_counter() - t0
            action_count += 1
            obs, reward_vec, done, info = env.step(a)
            ep_return += float(reward_vec.sum())
            if done:
                break
        returns.append(ep_return)
    returns = np.array(returns, dtype=np.float32)
    if return_timing:
        return returns, action_time / max(action_count, 1)
    return returns


def run_baselines(env_cfg=None, n_episodes=30, seed=42, verbose=True):
    cfg = env_cfg or AdaptRMABConfig()
    if verbose:
        print(f"AdaptRMAB  N={cfg.N}  K={cfg.K}  T={cfg.T}  M={cfg.M}  "
              f"episodes={n_episodes}")
        print(f"  sigma_v={cfg.sigma_v}  sigma_w={cfg.sigma_w}")
        print(f"  ou_rho={cfg.ou_rho}  sigma_alpha={cfg.sigma_alpha}  "
              f"sigma_beta={cfg.sigma_beta}")
        print()
        for m in range(cfg.M):
            ss_p = -cfg.drift / (1 - TYPE_ALPHA_MEAN[m])
            ss_a = (TYPE_BETA_MEAN[m] - cfg.drift) / (1 - TYPE_ALPHA_MEAN[m])
            print(f"  type {m} ({TYPE_NAMES[m]:12s}): "
                  f"alpha_bar={TYPE_ALPHA_MEAN[m]:.2f}  "
                  f"beta_bar={TYPE_BETA_MEAN[m]:.2f}  "
                  f"passive_SS={ss_p:.1f}  active_SS={ss_a:.1f}")
        print()

    results = {}
    for name, pname, policy in [
        ("random",           "random",           RandomPolicy(cfg, seed=seed)),
        ("oracle_greedy",    "oracle_greedy",    OracleGreedyPolicy(cfg)),
        ("oracle_lookahead", "oracle_lookahead", OracleLookaheadPolicy(cfg)),
    ]:
        rets = evaluate_policy(pname, policy, cfg, n_episodes, seed_offset=seed)
        results[name] = rets
        if verbose:
            print(f"  {name:20s}  mean={rets.mean():.1f}  std={rets.std():.1f}"
                  f"  min={rets.min():.1f}  max={rets.max():.1f}")
    return results


if __name__ == "__main__":
    run_baselines()
