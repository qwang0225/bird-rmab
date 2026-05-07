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
from env import AdaptRMABConfig, AdaptRMABEnv, TYPE_ALPHA_MEAN, TYPE_BETA_MEAN, TYPE_NAMES


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
        """H-step discounted marginal value of one activation at step 0.
        Stationary: alpha/beta fixed, no OU update needed."""
        cfg = self.cfg
        x_a = x0;  x_p = x0
        val = 0.0
        for h in range(self.H):
            # State transition (w=0); action applied only at h=0
            x_a = a0 * x_a + b0 * (1.0 if h == 0 else 0.0) - cfg.drift
            x_p = a0 * x_p - cfg.drift
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
    """
    Greedy policy using only current noisy observation y_i = x_i + noise.
    Activates top-K arms by highest y_i — no oracle info, no history.
    """

    def __init__(self, cfg: AdaptRMABConfig):
        self.cfg = cfg

    def reset(self, **_):
        pass

    def act(self, obs: np.ndarray, **_) -> np.ndarray:
        a = np.zeros(self.cfg.N, dtype=np.int32)
        a[np.argsort(-obs)[:self.cfg.K]] = 1
        return a


def evaluate_policy(policy_name, policy, env_cfg, n_episodes, seed_offset=0):
    returns = []
    for ep in range(n_episodes):
        env = AdaptRMABEnv(env_cfg, seed=seed_offset + ep)
        obs, info = env.reset()
        policy.reset()
        ep_return = 0.0
        for _ in range(env_cfg.T):
            if policy_name == "oracle_greedy":
                a = policy.act(obs, alpha_true=info["alpha_true"],
                               beta_true=info["beta_true"])
            elif policy_name == "oracle_lookahead":
                a = policy.act(obs, x_true=info["x_true"],
                               alpha_true=info["alpha_true"],
                               beta_true=info["beta_true"],
                               theta_true=info["theta_true"])
            else:
                a = policy.act(obs, **info)
            obs, reward_vec, done, info = env.step(a)
            ep_return += float(reward_vec.sum())
            if done:
                break
        returns.append(ep_return)
    return np.array(returns, dtype=np.float32)


def run_baselines(env_cfg=None, n_episodes=30, seed=42, verbose=True):
    cfg = env_cfg or AdaptRMABConfig()
    if verbose:
        print(f"AdaptRMAB  N={cfg.N}  K={cfg.K}  T={cfg.T}  M={cfg.M}  "
              f"episodes={n_episodes}")
        print(f"  sigma_v={cfg.sigma_v}  sigma_w={cfg.sigma_w}")
        print(f"  (stationary: alpha/beta fixed at type means)")
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
