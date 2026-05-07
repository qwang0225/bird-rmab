"""
whittle.py  —  Exact Whittle index for 2-state RMAB arms.

For a single arm with parameters:
    q0  = P(s'=1 | s=0, a=0)   passive recovery
    q1  = P(s'=1 | s=1, a=0)   passive retention
    p0  = P(s'=1 | s=0, a=1)   active  recovery
    p1  = P(s'=1 | s=1, a=1)   active  retention
and discount γ, the Whittle index W[s] for state s is the subsidy m
at which the optimal policy is indifferent between activating and
resting the arm while it is in state s.

Single-arm MDP with subsidy m (added to the passive reward):
    R_m(s, a=0) = s + m
    R_m(s, a=1) = s
    T(s'|s, a)  = as above

Algorithm: value iteration + bisection over m.

Public API
----------
    whittle_single(q0, q1, p0, p1, gamma) -> np.ndarray shape (2,)
    whittle_batch(q0_arr, q1_arr, p0_arr, p1_arr, gamma) -> np.ndarray shape (N, 2)
    oracle_whittle_policy(env) -> callable  policy(state) -> action
"""
from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Core VI solver
# ─────────────────────────────────────────────────────────────────────────────

def _vi_q_values(
    q0: float,
    q1: float,
    p0: float,
    p1: float,
    m:  float,
    gamma: float = 0.99,
    n_iter: int  = 2_000,
    tol:    float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Value iteration for a 2-state arm under subsidy m on the passive action.

    Returns
    -------
    Q_act : (2,)  Q(s, active)
    Q_pas : (2,)  Q(s, passive) including subsidy m
    """
    V = np.zeros(2)
    for _ in range(n_iter):
        # Q(s, active)  — no subsidy
        Q_act = np.array([
            0.0 + gamma * (p0 * V[1] + (1.0 - p0) * V[0]),   # s=0
            1.0 + gamma * (p1 * V[1] + (1.0 - p1) * V[0]),   # s=1
        ])
        # Q(s, passive) + subsidy m
        Q_pas = np.array([
            0.0 + m + gamma * (q0 * V[1] + (1.0 - q0) * V[0]),
            1.0 + m + gamma * (q1 * V[1] + (1.0 - q1) * V[0]),
        ])
        V_new = np.maximum(Q_act, Q_pas)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    return Q_act, Q_pas


# ─────────────────────────────────────────────────────────────────────────────
# Whittle index for a single arm
# ─────────────────────────────────────────────────────────────────────────────

def whittle_single(
    q0: float,
    q1: float,
    p0: float,
    p1: float,
    gamma:    float = 0.99,
    m_lo:     float = -2.0,
    m_hi:     float =  3.0,
    n_bisect: int   = 60,
    n_vi:     int   = 2_000,
) -> np.ndarray:
    """
    Compute Whittle index for both states of a 2-state arm.

    W[s] = subsidy m* such that Q(s, active) == Q(s, passive) under the
    optimal policy for the single-arm MDP with subsidy m*.

    Bisection: for m < W[s], active is preferred; for m > W[s], passive
    is preferred.  We binary-search the sign-change of Q_act[s] - Q_pas[s].

    Returns
    -------
    W : np.ndarray shape (2,)  [W(s=0), W(s=1)]
    """
    W = np.empty(2)
    for s in range(2):
        lo, hi = float(m_lo), float(m_hi)
        for _ in range(n_bisect):
            m = 0.5 * (lo + hi)
            Q_act, Q_pas = _vi_q_values(q0, q1, p0, p1, m, gamma, n_vi)
            if Q_act[s] >= Q_pas[s]:
                lo = m   # active still preferred → need larger subsidy
            else:
                hi = m   # passive already preferred → subsidy too large
        W[s] = 0.5 * (lo + hi)
    return W


# ─────────────────────────────────────────────────────────────────────────────
# Batch computation for N arms
# ─────────────────────────────────────────────────────────────────────────────

def whittle_batch(
    q0: np.ndarray,
    q1: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    gamma:    float = 0.99,
    m_lo:     float = -2.0,
    m_hi:     float =  3.0,
    n_bisect: int   = 60,
    n_vi:     int   = 2_000,
    verbose:  bool  = False,
) -> np.ndarray:
    """
    Compute Whittle indices for N arms in parallel (pure Python loop).

    Returns
    -------
    W : np.ndarray shape (N, 2)  W[i, s] = Whittle index of arm i in state s
    """
    N = len(q0)
    W = np.empty((N, 2))
    for i in range(N):
        W[i] = whittle_single(
            float(q0[i]), float(q1[i]), float(p0[i]), float(p1[i]),
            gamma=gamma, m_lo=m_lo, m_hi=m_hi, n_bisect=n_bisect, n_vi=n_vi,
        )
        if verbose and (i % 10 == 0):
            print(f"  whittle_batch  {i+1}/{N} done", end="\r", flush=True)
    if verbose:
        print()
    return W


# ─────────────────────────────────────────────────────────────────────────────
# Oracle Whittle policy  (knows true params)
# ─────────────────────────────────────────────────────────────────────────────

class OracleWhittlePolicy:
    """
    Pre-computes exact Whittle indices from the true arm parameters.
    At each step: W_i = W[i, state_i]; activate top-K arms by index.
    """

    def __init__(
        self,
        q0: np.ndarray,
        q1: np.ndarray,
        p0: np.ndarray,
        p1: np.ndarray,
        K: int,
        gamma: float = 0.99,
        verbose: bool = True,
    ):
        self.K = K
        if verbose:
            print(f"[OracleWhittlePolicy] computing Whittle indices for {len(q0)} arms … ", end="", flush=True)
        self.W = whittle_batch(q0, q1, p0, p1, gamma=gamma, verbose=False)
        if verbose:
            print("done.")

    def reset(self, **_):
        pass

    def act(self, state: np.ndarray, **_) -> np.ndarray:
        N  = len(state)
        wi = self.W[np.arange(N), state]   # W[i, state_i]
        a  = np.zeros(N, dtype=np.int32)
        a[np.argsort(-wi)[: self.K]] = 1
        return a


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_whittle_table(
    q0: np.ndarray,
    q1: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    W:  np.ndarray,
    arm_type: np.ndarray | None = None,
    max_rows: int = 20,
) -> None:
    """Pretty-print per-arm Whittle indices."""
    N = len(q0)
    print(f"\n{'i':>4}  {'type':9}  {'q0':6}  {'q1':6}  {'p0':6}  {'p1':6}  {'W(0)':8}  {'W(1)':8}")
    print("-" * 68)
    for i in range(min(N, max_rows)):
        tname = str(arm_type[i]) if arm_type is not None else "-"
        print(
            f"{i:4d}  {tname:9s}  "
            f"{q0[i]:.3f}  {q1[i]:.3f}  {p0[i]:.3f}  {p1[i]:.3f}  "
            f"{W[i,0]:+.4f}  {W[i,1]:+.4f}"
        )
    if N > max_rows:
        print(f"  … ({N - max_rows} more arms)")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Sanity check: for a symmetric arm where active = passive, W should be ~0
    W_sym = whittle_single(0.3, 0.7, 0.3, 0.7, gamma=0.95)
    print(f"Symmetric arm  W=[{W_sym[0]:.4f}, {W_sym[1]:.4f}]  (expect both ≈ 0)")

    # Arms where active helps a lot → high Whittle index
    W_high = whittle_single(0.05, 0.20, 0.50, 0.75, gamma=0.95)
    print(f"High-benefit arm W=[{W_high[0]:.4f}, {W_high[1]:.4f}]  (expect W(0) > W(1))")

    # Batch
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from env import MarkovRMABConfig, MarkovRMABEnv, ARM_TYPE_NAMES

    cfg = MarkovRMABConfig(N=10, K=2, T=50)
    env = MarkovRMABEnv(cfg)
    W   = whittle_batch(env.q0, env.q1, env.p0, env.p1, gamma=0.95, verbose=True)
    print_whittle_table(env.q0, env.q1, env.p0, env.p1, W, env.arm_type)
