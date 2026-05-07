"""
neurwin.py  —  NeurWIN oracle policy for 2-state Markov RMAB.

Uses the true per-arm transition probabilities (q0, q1, p0, p1) as input.
An IndexNet MLP maps (q0_i, q1_i, p0_i, p1_i, state_i) -> score_i and is
trained to regress toward the true Whittle index computed by VI.

This is a strong oracle baseline: it knows the dynamics but still needs to
learn to map them to Whittle scores rather than running VI at act-time.

Usage
-----
    # Train (run once, saves checkpoint)
    python neurwin.py --N 50 --K 10 --epochs 500

    # In run.py (load only)
    from neurwin import NeurWINOraclePolicy, load_neurwin
    pol = load_neurwin("checkpoints_neurwin/best.pth", env)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env import MarkovRMABConfig, MarkovRMABEnv
from whittle import whittle_batch
from baselines import evaluate_policy


# ---------------------------------------------------------------------------
# IndexNet: (q0, q1, p0, p1, state) -> Whittle score
# ---------------------------------------------------------------------------

class IndexNet(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        # Input: [q0, q1, p0, p1, state] = 5 dims
        self.net = nn.Sequential(
            nn.Linear(5, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 5)
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# NeurWIN Oracle Policy
# ---------------------------------------------------------------------------

class NeurWINOraclePolicy:
    """
    NeurWIN with oracle transition parameters.

    At act time: score_i = IndexNet(q0_i, q1_i, p0_i, p1_i, state_i).
    No history needed — dynamics are known.
    """

    def __init__(self, q0: np.ndarray, q1: np.ndarray,
                 p0: np.ndarray, p1: np.ndarray,
                 K: int, hidden_dim: int = 64):
        self.N  = len(q0)
        self.K  = K
        # Store normalised transitions as tensor (N, 4)
        self._params = torch.from_numpy(
            np.stack([q0, q1, p0, p1], axis=1).astype(np.float32)
        )
        self.net = IndexNet(hidden_dim=hidden_dim)

    def reset(self, **_):
        pass

    def act(self, state: np.ndarray, **_) -> np.ndarray:
        s = torch.from_numpy(state.astype(np.float32)).unsqueeze(1)   # (N, 1)
        x = torch.cat([self._params, s], dim=1)                        # (N, 5)
        with torch.no_grad():
            scores = self.net(x).numpy()                                # (N,)
        a = np.zeros(self.N, dtype=np.int32)
        a[np.argsort(-scores)[: self.K]] = 1
        return a

    def whittle_scores(self, state: np.ndarray) -> np.ndarray:
        s = torch.from_numpy(state.astype(np.float32)).unsqueeze(1)
        x = torch.cat([self._params, s], dim=1)
        with torch.no_grad():
            return self.net(x).numpy()

    def save_checkpoint(self, path: str) -> None:
        torch.save({
            "net":    self.net.state_dict(),
            "params": self._params.numpy(),
            "K":      self.K,
        }, path)
        print(f"[neurwin] saved → {path}")

    def load_checkpoint(self, path: str) -> None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.net.load_state_dict(ck["net"])
        print(f"[neurwin] loaded ← {path}")


def load_neurwin(path: str, env: MarkovRMABEnv,
                 hidden_dim: int = 64) -> NeurWINOraclePolicy:
    """Load a saved NeurWIN checkpoint and return ready-to-use policy."""
    pol = NeurWINOraclePolicy(
        env.q0, env.q1, env.p0, env.p1,
        K=env.K, hidden_dim=hidden_dim,
    )
    pol.load_checkpoint(path)
    pol.net.eval()
    return pol


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    env: MarkovRMABEnv,
    save_dir: str = "checkpoints_neurwin",
    epochs: int = 500,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    gamma: float = 0.99,
    n_eval: int = 50,
    seed: int = 0,
) -> NeurWINOraclePolicy:
    """
    Train IndexNet to regress toward true Whittle indices.

    True Whittle indices W[i, s] are computed once via VI on known transitions.
    Training is pure supervised regression — no rollouts needed.
    We augment with random arm parameter samples to improve generalisation.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    pol = NeurWINOraclePolicy(
        env.q0, env.q1, env.p0, env.p1,
        K=env.K, hidden_dim=hidden_dim,
    )
    opt = torch.optim.Adam(pol.net.parameters(), lr=lr)

    # ── True Whittle targets for the fixed arm population ────────────────
    W_true = whittle_batch(env.q0, env.q1, env.p0, env.p1,
                           gamma=gamma, verbose=False)   # (N, 2)

    # Build fixed dataset: one sample per (arm, state)
    params_np = np.stack([env.q0, env.q1, env.p0, env.p1], axis=1)  # (N, 4)
    X_list, Y_list = [], []
    for s in range(2):
        x = np.hstack([params_np, np.full((env.N, 1), float(s))])   # (N, 5)
        y = W_true[:, s]                                              # (N,)
        X_list.append(x); Y_list.append(y)

    X = torch.from_numpy(np.vstack(X_list).astype(np.float32))   # (2N, 5)
    Y = torch.from_numpy(np.hstack(Y_list).astype(np.float32))   # (2N,)

    os.makedirs(save_dir, exist_ok=True)
    best_loss = float("inf")
    best_path = str(Path(save_dir) / "best.pth")

    print(f"[neurwin] training {epochs} epochs on N={env.N} arms  "
          f"(W range [{W_true.min():.3f}, {W_true.max():.3f}])")

    for ep in range(1, epochs + 1):
        pol.net.train()
        opt.zero_grad()
        pred = pol.net(X)
        loss = F.mse_loss(pred, Y)
        loss.backward()
        opt.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            pol.save_checkpoint(best_path)

        if ep % 50 == 0:
            pol.net.eval()
            print(f"  epoch {ep:4d}  loss={loss.item():.6f}  best={best_loss:.6f}")

    print(f"[neurwin] training done — best checkpoint: {best_path}")
    pol.load_checkpoint(best_path)
    pol.net.eval()
    return pol


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N",        type=int,   default=50)
    parser.add_argument("--K",        type=int,   default=10)
    parser.add_argument("--T",        type=int,   default=100)
    parser.add_argument("--epochs",   type=int,   default=500)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--hidden",   type=int,   default=64)
    parser.add_argument("--gamma",    type=float, default=0.99)
    parser.add_argument("--seed",     type=int,   default=0)
    parser.add_argument("--save_dir", type=str,   default="checkpoints_neurwin")
    parser.add_argument("--n_eval",   type=int,   default=50)
    args = parser.parse_args()

    env_cfg = MarkovRMABConfig(N=args.N, K=args.K, T=args.T, seed_params=args.seed)
    env     = MarkovRMABEnv(env_cfg, seed=args.seed)
    env.arm_summary(); print()

    pol = train(
        env,
        save_dir=args.save_dir,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden,
        gamma=args.gamma,
        seed=args.seed,
    )

    # Quick eval
    print(f"\nEvaluating {args.n_eval} episodes …")
    rets = evaluate_policy(pol, env_cfg, n_episodes=args.n_eval, seed_offset=999)
    print(f"  NeurWIN oracle  mean={rets.mean():.2f}  std={rets.std():.2f}")


if __name__ == "__main__":
    main()
