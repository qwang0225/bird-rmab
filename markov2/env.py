"""
env.py  —  2-state Markov RMAB environment.

Each arm i has a fully-observable binary state s_i ∈ {0, 1}.
Transitions are governed by 4 per-arm probabilities:
    q0[i]  = P(s'=1 | s=0, a=0)   passive recovery
    q1[i]  = P(s'=1 | s=1, a=0)   passive retention
    p0[i]  = P(s'=1 | s=0, a=1)   active recovery     (>= q0)
    p1[i]  = P(s'=1 | s=1, a=1)   active retention    (>= q1)

Reward per step: r_t = sum_i  s_i(t)   (total arms in "good" state 1).
Budget:          sum_i a_i(t) <= K     (at most K activations per step).

Arms are drawn at __init__ time from a fixed mixture of 2 types so
every policy faces the same arm population.  The true parameters are
stored in self.q0, self.q1, self.p0, self.p1 and are available to
oracle policies.

Usage
-----
    cfg = MarkovRMABConfig(N=50, K=10, T=100)
    env = MarkovRMABEnv(cfg, seed=0)
    obs, info = env.reset()                 # obs = state vector (N,)
    obs, reward_vec, done, info = env.step(action)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ─────────────────────────────────────────────
# Arm type definitions  (2-type binary setup)
# ─────────────────────────────────────────────
#  Each row: (q0, q1, p0, p1)
#   q0 = passive recovery,  q1 = passive retention
#   p0 = active  recovery,  p1 = active  retention
#
# Two types chosen to be clearly separable yet both realistic:
#   low  responder: poor passive dynamics, modest treatment lift
#   high responder: decent passive dynamics, large treatment lift
# The belief encoder must infer which type each arm is from obs history.
#
# (5-type version commented out below for reference)
# ARM_TYPES_5 = np.array([
#     [0.05, 0.85, 0.30, 0.92],   # "stubborn"
#     [0.08, 0.25, 0.50, 0.75],   # "fragile"
#     [0.25, 0.65, 0.60, 0.85],   # "resilient"
#     [0.40, 0.80, 0.72, 0.94],   # "easy"
#     [0.03, 0.12, 0.38, 0.62],   # "critical"
# ], dtype=np.float64)

ARM_TYPES = np.array([
    [0.05, 0.30, 0.35, 0.60],   # type 0  "low responder"  – poor baseline, modest treatment lift
    [0.20, 0.60, 0.65, 0.90],   # type 1  "high responder" – decent baseline, large treatment lift
], dtype=np.float64)

# ARM_TYPES = np.array([
#     [0.05, 0.10, 0.40, 0.65],   # type 0  "fragile"  – unstable in s=1, large treatment benefit (p1-q1=0.55)
#     [0.20, 0.70, 0.65, 0.85],   # type 1  "stable"   – stable  in s=1, modest treatment benefit (p1-q1=0.15)
# ], dtype=np.float64)

ARM_TYPE_NAMES = ["fragile", "stable"]


@dataclass
class MarkovRMABConfig:
    N:           int   = 50     # number of arms
    K:           int   = 10     # budget (activations per step)
    T:           int   = 100    # episode length
    seed_params: int   = 0      # RNG seed for drawing arm parameters
    noise:       float = 0.04   # per-arm Gaussian noise added to type params


class MarkovRMABEnv:
    """
    2-state RMAB with heterogeneous arms (fully observable).

    After construction the arm parameters q0, q1, p0, p1 are FIXED for
    the life of the object.  Only the state trajectory is stochastic.

    Attributes (all shape (N,)):
        q0, q1   – passive transition probabilities
        p0, p1   – active  transition probabilities
        arm_type – integer type index per arm (0‥4)
        state    – current state of each arm (updated by step)
    """

    def __init__(self, cfg: MarkovRMABConfig, seed: int = 0):
        self.cfg = cfg
        self.N   = cfg.N
        self.K   = cfg.K
        self.T   = cfg.T

        # ── generate fixed arm parameters ────────────────────────────────
        rng_params = np.random.default_rng(cfg.seed_params)

        n_types = len(ARM_TYPES)
        # Assign arms roughly evenly across types
        types = np.array([i % n_types for i in range(self.N)], dtype=int)
        rng_params.shuffle(types)
        self.arm_type = types

        base = ARM_TYPES[types]  # (N, 4)
        noise = rng_params.normal(0.0, cfg.noise, size=(self.N, 4))
        params = np.clip(base + noise, 1e-3, 1.0 - 1e-3)

        self.q0 = params[:, 0].copy()
        self.q1 = params[:, 1].copy()
        self.p0 = params[:, 2].copy()
        self.p1 = params[:, 3].copy()

        # Enforce active >= passive (indexability condition)
        self.p0 = np.maximum(self.p0, self.q0 + 1e-3)
        self.p1 = np.maximum(self.p1, self.q1 + 1e-3)
        self.p0 = np.minimum(self.p0, 1.0 - 1e-3)
        self.p1 = np.minimum(self.p1, 1.0 - 1e-3)

        # ── episode RNG (re-seeded in reset) ────────────────────────────
        self.rng   = np.random.default_rng(seed)
        self.state = np.zeros(self.N, dtype=np.int32)
        self.t     = 0

    # ─────────────────────────────────────────────────────────────────────
    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # Initialise each arm independently: state ~ Bernoulli(0.5)
        self.state = self.rng.integers(0, 2, size=self.N).astype(np.int32)
        self.t     = 0
        info = {
            "t":        self.t,
            "q0":       self.q0.copy(),
            "q1":       self.q1.copy(),
            "p0":       self.p0.copy(),
            "p1":       self.p1.copy(),
            "arm_type": self.arm_type.copy(),
        }
        return self.state.copy(), info

    # ─────────────────────────────────────────────────────────────────────
    def step(self, a: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, dict]:
        a = np.asarray(a, dtype=int).reshape(-1)
        if a.shape[0] != self.N:
            raise ValueError(f"action must have shape ({self.N},), got {a.shape}")
        if np.any((a != 0) & (a != 1)):
            raise ValueError("action must be binary (0/1).")
        if int(a.sum()) > self.K:
            raise ValueError(f"budget violated: sum(a)={int(a.sum())} > K={self.K}")

        old_state = self.state.copy()

        # Vectorised transition
        # P(next=1 | s=0, a=0) = q0, P(next=1 | s=0, a=1) = p0
        # P(next=1 | s=1, a=0) = q1, P(next=1 | s=1, a=1) = p1
        prob_move_to_1 = np.where(
            a == 0,
            np.where(old_state == 0, self.q0, self.q1),
            np.where(old_state == 0, self.p0, self.p1),
        )
        self.state = (self.rng.random(self.N) < prob_move_to_1).astype(np.int32)

        reward_vec = self.state.astype(np.float32)   # reward_i = new state
        self.t    += 1
        done       = self.t >= self.T

        info = {
            "t":         self.t,
            "old_state": old_state,
            "a":         a.copy(),
            "transitions": np.stack([old_state, a, self.state], axis=1),  # (N,3)
        }
        return self.state.copy(), reward_vec, done, info

    # ─────────────────────────────────────────────────────────────────────
    def arm_summary(self) -> None:
        """Print per-type breakdown of arms."""
        from collections import Counter
        cnt = Counter(self.arm_type.tolist())
        print(f"Arm parameter summary  (N={self.N}  K={self.K}  T={self.T})")
        print(f"{'Type':10s}  {'Count':5s}  {'q0':6s}  {'q1':6s}  {'p0':6s}  {'p1':6s}  {'dWI(1)':>8s}")
        for t in sorted(cnt):
            mask = self.arm_type == t
            q0 = self.q0[mask].mean(); q1 = self.q1[mask].mean()
            p0 = self.p0[mask].mean(); p1 = self.p1[mask].mean()
            # Rough effective benefit: active - passive steady state gap
            delta = ((p0 - q0) + (p1 - q1)) / 2
            print(f"{ARM_TYPE_NAMES[t]:10s}  {cnt[t]:5d}  {q0:.3f}  {q1:.3f}  {p0:.3f}  {p1:.3f}  {delta:+.3f}")
