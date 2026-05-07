"""
baselines.py  —  Policies for the 2-state Markov RMAB benchmark.

Policies
--------
RandomPolicy
    Activates K arms chosen uniformly at random each step.

GreedyPolicy
    Activates the K arms that are currently in state 0 (most in need of help).
    Tie-broken by arm index.  A simple myopic "rescue the worst" heuristic.

OracleWhittlePolicy  (imported from whittle.py)
    Pre-computes exact Whittle indices from the true arm parameters.

BayesianWhittlePolicy  (Thompson Sampling variant)
    Maintains a Beta-Bernoulli posterior over each arm's 4 transition
    probabilities.  At each step:
        1. Sample (q0_s, q1_s, p0_s, p1_s) from current posterior.
        2. Compute Whittle index from sampled parameters.
        3. Activate top-K arms by sampled index.
    Thompson Sampling naturally trades off exploration / exploitation.

BayesianWhittleMeanPolicy  (posterior-mean variant)
    Same as above but uses the posterior MEAN instead of sampling.
    Greedy Bayesian — typically converges faster than TS but can get
    stuck in locally suboptimal arm rankings early on.

UCBWhittlePolicy
    Uses UCB confidence interval on each transition probability:
        p_ucb = posterior_mean + sqrt(log(t+1) / (count + 1))
    clamped to [0, 1], then compute Whittle index.

Helper
------
evaluate_policy(policy, env_cfg, n_episodes, seed_offset, verbose)
    Run n_episodes episodes and return array of total rewards.

Note: NeurWIN policy has moved to neurwin.py.
"""
from __future__ import annotations

import numpy as np

from env import MarkovRMABConfig, MarkovRMABEnv
from whittle import whittle_single, OracleWhittlePolicy


# ─────────────────────────────────────────────────────────────────────────────
# Random policy
# ─────────────────────────────────────────────────────────────────────────────

class RandomPolicy:
    def __init__(self, N: int, K: int, seed: int = 0):
        self.N   = N
        self.K   = K
        self.rng = np.random.default_rng(seed)

    def reset(self, **_):
        pass

    def act(self, state: np.ndarray, **_) -> np.ndarray:
        a = np.zeros(self.N, dtype=np.int32)
        a[self.rng.choice(self.N, size=self.K, replace=False)] = 1
        return a


# ─────────────────────────────────────────────────────────────────────────────
# Greedy "rescue worst" policy
# ─────────────────────────────────────────────────────────────────────────────

class GreedyPolicy:
    """
    Activates the K arms currently in state 0 (prioritises arms most in need).
    If fewer than K arms are in state 0, fills remaining slots with
    state-1 arms (arbitrary order).
    """

    def __init__(self, N: int, K: int):
        self.N = N
        self.K = K

    def reset(self, **_):
        pass

    def act(self, state: np.ndarray, **_) -> np.ndarray:
        a     = np.zeros(self.N, dtype=np.int32)
        bad   = np.where(state == 0)[0]
        good  = np.where(state == 1)[0]
        picks = list(bad[: self.K])
        if len(picks) < self.K:
            picks += list(good[: self.K - len(picks)])
        a[picks] = 1
        return a


# ─────────────────────────────────────────────────────────────────────────────
# Beta-Bernoulli posterior tracker
# ─────────────────────────────────────────────────────────────────────────────

class BetaBernoulliTracker:
    """
    Maintains Beta(alpha, beta) posterior for each arm × state × action.

    Indexing:
        self.alpha[i, s, a]   successes (transitions to state 1)
        self.beta_ [i, s, a]  failures  (transitions to state 0)
    """

    def __init__(self, N: int, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.N = N
        self.alpha  = np.full((N, 2, 2), prior_alpha)
        self.beta_  = np.full((N, 2, 2), prior_beta)

    def update(self, old_state: np.ndarray, action: np.ndarray, new_state: np.ndarray):
        """Batch update from one time step's transitions."""
        for i in range(self.N):
            s, a, sn = int(old_state[i]), int(action[i]), int(new_state[i])
            if sn == 1:
                self.alpha[i, s, a] += 1.0
            else:
                self.beta_[i, s, a] += 1.0

    def mean(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns posterior means: q0, q1, p0, p1 each shape (N,)."""
        mu = self.alpha / (self.alpha + self.beta_)
        return mu[:, 0, 0], mu[:, 1, 0], mu[:, 0, 1], mu[:, 1, 1]

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Thompson sample from posteriors."""
        raw = rng.beta(self.alpha, self.beta_)   # (N, 2, 2)
        return raw[:, 0, 0], raw[:, 1, 0], raw[:, 0, 1], raw[:, 1, 1]

    def ucb(self, t: int, c: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """UCB estimate = mean + c * sqrt(log(t+1) / count)."""
        count = self.alpha + self.beta_ - 2.0          # subtract priors
        count = np.maximum(count, 1.0)
        bonus = c * np.sqrt(np.log(t + 1) / count)
        mu    = self.alpha / (self.alpha + self.beta_)
        ucb   = np.clip(mu + bonus, 0.0, 1.0)
        return ucb[:, 0, 0], ucb[:, 1, 0], ucb[:, 0, 1], ucb[:, 1, 1]


# ─────────────────────────────────────────────────────────────────────────────
# Bayesian Whittle policies (Thompson Sampling & Posterior Mean)
# ─────────────────────────────────────────────────────────────────────────────

class BayesianWhittlePolicy:
    """
    Thompson-Sampling Whittle index policy.

    At each step:
      1. Draw parameter sample from Beta posteriors.
      2. Compute Whittle index for each arm using sampled params.
      3. Activate top-K arms by W[i, state_i].

    The Beta-Bernoulli tracker is updated after each observed transition.
    """

    def __init__(
        self,
        N: int,
        K: int,
        gamma: float = 0.99,
        seed:  int   = 0,
        prior_alpha: float = 1.0,
        prior_beta:  float = 1.0,
        n_bisect: int = 50,
        n_vi:     int = 1_000,
    ):
        self.N = N
        self.K = K
        self.gamma    = gamma
        self.rng      = np.random.default_rng(seed)
        self.tracker  = BetaBernoulliTracker(N, prior_alpha, prior_beta)
        self.n_bisect = n_bisect
        self.n_vi     = n_vi

    def reset(self, **_):
        pass  # tracker is NOT reset between episodes (online learning)

    def _whittle_from_params(self, q0, q1, p0, p1, state: np.ndarray) -> np.ndarray:
        """Compute W[i, state_i] for each arm given parameter arrays."""
        scores = np.empty(self.N)
        for i in range(self.N):
            W = whittle_single(
                float(q0[i]), float(q1[i]),
                float(p0[i]), float(p1[i]),
                gamma=self.gamma,
                n_bisect=self.n_bisect,
                n_vi=self.n_vi,
            )
            scores[i] = W[int(state[i])]
        return scores

    def act(self, state: np.ndarray, **_) -> np.ndarray:
        q0, q1, p0, p1 = self.tracker.sample(self.rng)
        # Enforce p >= q (indexability) on sampled values
        p0 = np.maximum(p0, q0 + 1e-3)
        p1 = np.maximum(p1, q1 + 1e-3)
        scores = self._whittle_from_params(q0, q1, p0, p1, state)
        a = np.zeros(self.N, dtype=np.int32)
        a[np.argsort(-scores)[: self.K]] = 1
        return a

    def observe(self, old_state: np.ndarray, action: np.ndarray, new_state: np.ndarray):
        self.tracker.update(old_state, action, new_state)


class BayesianWhittleMeanPolicy(BayesianWhittlePolicy):
    """
    Posterior-mean Whittle index policy (greedy Bayes, no exploration).
    Identical to BayesianWhittlePolicy except act() uses posterior MEAN
    instead of Thompson sampling.
    """

    def act(self, state: np.ndarray, **_) -> np.ndarray:
        q0, q1, p0, p1 = self.tracker.mean()
        p0 = np.maximum(p0, q0 + 1e-3)
        p1 = np.maximum(p1, q1 + 1e-3)
        scores = self._whittle_from_params(q0, q1, p0, p1, state)
        a = np.zeros(self.N, dtype=np.int32)
        a[np.argsort(-scores)[: self.K]] = 1
        return a


class UCBWhittlePolicy(BayesianWhittlePolicy):
    """
    UCB-based Whittle index policy.
    Uses optimistic upper-confidence-bound on each transition probability.
    """

    def __init__(self, *args, ucb_c: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ucb_c = ucb_c
        self._t = 0

    def reset(self, **_):
        self._t = 0

    def act(self, state: np.ndarray, **_) -> np.ndarray:
        q0, q1, p0, p1 = self.tracker.ucb(self._t, c=self.ucb_c)
        p0 = np.clip(np.maximum(p0, q0 + 1e-3), 0.0, 1.0 - 1e-3)
        p1 = np.clip(np.maximum(p1, q1 + 1e-3), 0.0, 1.0 - 1e-3)
        scores = self._whittle_from_params(q0, q1, p0, p1, state)
        a = np.zeros(self.N, dtype=np.int32)
        a[np.argsort(-scores)[: self.K]] = 1
        return a

    def observe(self, old_state, action, new_state):
        super().observe(old_state, action, new_state)
        self._t += 1


# ─────────────────────────────────────────────────────────────────────────────
# WIQL — Whittle Index Q-Learning (tabular per-arm)
# ─────────────────────────────────────────────────────────────────────────────

class WIQLPolicy:
    """
    Whittle Index Q-Learning for 2-state RMAB.

    For each arm i, maintains a Q-table Q[i, s, a] updated via TD(0).
    The Whittle index estimate for arm i in state s is the Q-advantage:

        W_hat(i, s) = Q[i, s, 1] - Q[i, s, 0]

    This is exact for 2-state because adding subsidy m to the passive action
    shifts Q(s, passive) by m, so indifference (m* = Q-advantage) holds
    directly without binary search.

    Activation policy: top-K arms by W_hat(i, state_i).
    Online learning: Q-tables updated after every observed transition.
    """

    def __init__(self, N: int, K: int, gamma: float = 0.99,
                 lr: float = 0.05, epsilon: float = 0.05, seed: int = 0):
        self.N       = N
        self.K       = K
        self.gamma   = gamma
        self.lr      = lr
        self.epsilon = epsilon          # epsilon-greedy exploration
        self.rng     = np.random.default_rng(seed)

        # Q[i, s, a]  — initialised optimistically to encourage exploration
        self.Q = np.ones((N, 2, 2), dtype=np.float64) * 0.5

        self._prev_state  = np.zeros(N, dtype=int)
        self._prev_action = np.zeros(N, dtype=int)
        self._step        = 0

    def reset(self, **_):
        self._prev_state.fill(0)
        self._prev_action.fill(0)
        self._step = 0

    def _update(self, old_state: np.ndarray, action: np.ndarray,
                reward: np.ndarray, new_state: np.ndarray) -> None:
        for i in range(self.N):
            s, a, r, sn = int(old_state[i]), int(action[i]), float(reward[i]), int(new_state[i])
            best_next = float(np.max(self.Q[i, sn]))
            td_target = r + self.gamma * best_next
            self.Q[i, s, a] += self.lr * (td_target - self.Q[i, s, a])

    def whittle_scores(self, state: np.ndarray) -> np.ndarray:
        """W_hat(i) = Q[i, state_i, 1] - Q[i, state_i, 0]  for each arm."""
        return np.array([self.Q[i, int(state[i]), 1] - self.Q[i, int(state[i]), 0]
                         for i in range(self.N)], dtype=np.float32)

    def act(self, state: np.ndarray, **_) -> np.ndarray:
        if self._step > 0:
            # Update Q from last transition (reward = new state value)
            self._update(self._prev_state, self._prev_action,
                         state.astype(float), state)

        # Epsilon-greedy: explore randomly with prob epsilon
        if self.rng.random() < self.epsilon:
            a = np.zeros(self.N, dtype=np.int32)
            a[self.rng.choice(self.N, size=self.K, replace=False)] = 1
        else:
            scores = self.whittle_scores(state)
            a = np.zeros(self.N, dtype=np.int32)
            a[np.argsort(-scores)[:self.K]] = 1

        self._prev_state  = state.copy()
        self._prev_action = a.copy()
        self._step       += 1
        return a

    def observe(self, old_state: np.ndarray, action: np.ndarray,
                new_state: np.ndarray) -> None:
        """Explicit update hook (used by evaluate_policy when is_bayesian=True)."""
        self._update(old_state, action, new_state.astype(float), new_state)


# ─────────────────────────────────────────────────────────────────────────────
# WIQLOraclePolicy — WIQL warm-started with VI on known transitions
# ─────────────────────────────────────────────────────────────────────────────

class WIQLOraclePolicy:
    """
    WIQL with known transition dynamics.

    Instead of learning Q-values online via TD, each arm's Q-table is
    pre-computed by value iteration on the known (q0, q1, p0, p1).
    The Whittle score is the Q-advantage Q[s,1] - Q[s,0], which is
    exact at convergence.

    This is a strong oracle-equipped baseline that isolates the effect of
    knowing vs not knowing dynamics, without needing any rollout data.
    """

    def __init__(self, N: int, K: int, q0: np.ndarray, q1: np.ndarray,
                 p0: np.ndarray, p1: np.ndarray,
                 gamma: float = 0.99, n_vi: int = 2_000):
        self.N = N
        self.K = K
        # Pre-compute Q-tables via single-arm VI
        self.Q = np.zeros((N, 2, 2), dtype=np.float64)
        for i in range(N):
            self.Q[i] = _arm_vi(float(q0[i]), float(q1[i]),
                                 float(p0[i]), float(p1[i]),
                                 gamma=gamma, n_vi=n_vi)

    def reset(self, **_):
        pass

    def whittle_scores(self, state: np.ndarray) -> np.ndarray:
        return np.array([self.Q[i, int(state[i]), 1] - self.Q[i, int(state[i]), 0]
                         for i in range(self.N)], dtype=np.float32)

    def act(self, state: np.ndarray, **_) -> np.ndarray:
        scores = self.whittle_scores(state)
        a = np.zeros(self.N, dtype=np.int32)
        a[np.argsort(-scores)[:self.K]] = 1
        return a


def _arm_vi(q0: float, q1: float, p0: float, p1: float,
            gamma: float = 0.99, n_vi: int = 2_000) -> np.ndarray:
    """
    Single-arm value iteration → Q-table of shape (2, 2).

    State s ∈ {0,1}, action a ∈ {0=passive, 1=active}.
    Reward r(s) = s  (reward equals state value).
    Transition:  P(s'=1 | s=0, a=0)=q0,  P(s'=1 | s=1, a=0)=q1,
                 P(s'=1 | s=0, a=1)=p0,  P(s'=1 | s=1, a=1)=p1.
    """
    # Transition matrix T[s, a, s'] = P(s'|s,a)
    T = np.array([
        [[1-q0, q0], [1-p0, p0]],   # s=0: passive → (q0 to 1), active → (p0 to 1)
        [[1-q1, q1], [1-p1, p1]],   # s=1: passive → (q1 to 1), active → (p1 to 1)
    ], dtype=np.float64)
    # Reward r[s, a] = s (state value, action-independent)
    r = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)

    Q = np.zeros((2, 2), dtype=np.float64)
    for _ in range(n_vi):
        V = Q.max(axis=1)                          # (2,)
        Q = r + gamma * (T @ V)                    # (2, 2)
    return Q


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_policy(
    policy,
    env_cfg:    MarkovRMABConfig,
    n_episodes: int = 50,
    seed_offset: int = 0,
    verbose:    bool = False,
) -> np.ndarray:
    """
    Roll out `policy` for n_episodes and return per-episode total reward.

    For Bayesian policies (have `.observe()` method), belief updates are
    carried across steps within each episode but NOT reset between episodes
    (i.e., the agent learns online across the full evaluation run).

    Returns
    -------
    returns : np.ndarray shape (n_episodes,)
    """
    returns = np.empty(n_episodes)
    is_bayesian = hasattr(policy, "observe")

    for ep in range(n_episodes):
        env  = MarkovRMABEnv(env_cfg, seed=seed_offset + ep)
        obs, info = env.reset(seed=seed_offset + ep)

        if hasattr(policy, "reset"):
            policy.reset()

        ep_return = 0.0
        for _ in range(env_cfg.T):
            a = policy.act(obs)
            obs_next, reward_vec, done, info = env.step(a)

            ep_return += float(reward_vec.sum())

            if is_bayesian:
                policy.observe(info["old_state"], info["a"], obs_next)

            obs = obs_next
            if done:
                break

        returns[ep] = ep_return
        if verbose:
            print(f"  ep {ep+1:3d}/{n_episodes}  return={ep_return:.1f}", end="\r")
    if verbose:
        print()
    return returns


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = MarkovRMABConfig(N=20, K=4, T=50)
    env = MarkovRMABEnv(cfg)
    env.arm_summary()

    print("\nRunning quick baselines (20 episodes)...")
    policies = {
        "random":   RandomPolicy(cfg.N, cfg.K, seed=0),
        "greedy":   GreedyPolicy(cfg.N, cfg.K),
        "oracle":   OracleWhittlePolicy(env.q0, env.q1, env.p0, env.p1, cfg.K, gamma=0.99),
        "bayes_ts": BayesianWhittlePolicy(cfg.N, cfg.K, gamma=0.99, seed=0),
        "bayes_mu": BayesianWhittleMeanPolicy(cfg.N, cfg.K, gamma=0.99, seed=0),
    }
    for name, pol in policies.items():
        rets = evaluate_policy(pol, cfg, n_episodes=20, seed_offset=100)
        print(f"  {name:10s}  mean={rets.mean():.2f}  std={rets.std():.2f}")
