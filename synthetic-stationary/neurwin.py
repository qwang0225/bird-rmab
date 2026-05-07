"""
neurwin.py  (adapt)

NeurWIN for Bayesian-Adaptive RMAB with hidden types + drifting parameters.

Architecture:
  BeliefEncoder  : Transformer over L-step (obs, action) history -> z_i per arm
  BeliefIndexNet : MLP  z_i -> scalar Whittle index w_i

Algorithm (same pairwise REINFORCE as NeurWIN paper):
  - Sample target arm i, reference arm j (j acts as Lagrange subsidy threshold)
  - p(activate i) = sigmoid(scale * (w_i - w_j))
  - Train with REINFORCE on single-arm discounted return
  - Evaluate: activate top-K arms by Whittle index

Why BeliefEncoder here:
  - Without history, w_i = f(obs_i) cannot distinguish arm types
    (a "stubborn" arm and a "responsive" arm with same x look identical)
  - With history encoder, w_i = f(z_i) where z_i captures inferred type
    -> better Whittle index estimates -> better arm ranking
"""
from __future__ import annotations

import csv
import os
import random
import sys
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from env import AdaptRMABConfig, AdaptRMABEnv
from diffusion_DPMD_train import BeliefEncoder


# ---------------------------------------------------------------------------
# Model: belief embedding -> Whittle index
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
        """z: (batch, n_arms, z_dim) -> indices: (batch, n_arms)"""
        batch, n_arms, z_dim = z.shape
        return self.net(z.reshape(batch * n_arms, z_dim)).reshape(batch, n_arms)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    N: int = 20
    K: int = 5
    T: int = 100

    # Belief encoder (same as DPMD/RL for fair comparison)
    L: int = 40
    z_dim: int = 64
    encoder_hidden: int = 128
    encoder_heads: int = 4
    encoder_layers: int = 3

    # Index network
    index_hidden: int = 64
    sigmoid_scale: float = 8.0
    score_noise_std: float = 0.10

    # REINFORCE
    gamma: float = 0.99
    mini_batch_size: int = 16
    epochs: int = 200
    lr: float = 3e-4
    grad_clip: float = 1.0
    entropy_bonus: float = 1e-3

    save_dir: str = "checkpoints_neurwin"
    save_every: int = 25
    resume_path: str = ""


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _topk_action(scores: np.ndarray, k: int) -> np.ndarray:
    a = np.zeros(scores.shape[0], dtype=np.int64)
    a[np.argsort(-scores)[: min(k, scores.shape[0])]] = 1
    return a


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self, N: int, K: int, cfg: TrainConfig):
        self.N = int(N); self.K = int(K); self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.rng = np.random.default_rng(cfg.seed)

        self.encoder = BeliefEncoder(
            z_dim=cfg.z_dim, hidden_dim=cfg.encoder_hidden,
            n_heads=cfg.encoder_heads, n_layers=cfg.encoder_layers,
            L=cfg.L,
        ).to(self.device)

        self.index_net = BeliefIndexNet(
            z_dim=cfg.z_dim, hidden_dim=cfg.index_hidden,
        ).to(self.device)

        params = list(self.encoder.parameters()) + list(self.index_net.parameters())
        self.opt = torch.optim.Adam(params, lr=cfg.lr)

        # Internal history buffers for act_hard() (evaluation)
        self._obs_hist: np.ndarray | None = None
        self._act_hist: np.ndarray | None = None

    def reset_history(self):
        """Call before each evaluation episode."""
        self._obs_hist = np.zeros((self.N, self.cfg.L), dtype=np.float32)
        self._act_hist = np.zeros((self.N, self.cfg.L), dtype=np.float32)

    def _encode(self, obs_hist: torch.Tensor, act_hist: torch.Tensor) -> torch.Tensor:
        """(batch, N, L) x2 -> (batch, N, z_dim)"""
        return self.encoder(obs_hist, act_hist)

    @torch.no_grad()
    def select_action(self, obs_hist: np.ndarray, act_hist: np.ndarray,
                      explore: bool = True) -> np.ndarray:
        oh = torch.from_numpy(obs_hist[None]).float().to(self.device)   # (1, N, L)
        ah = torch.from_numpy(act_hist[None]).float().to(self.device)
        z  = self._encode(oh, ah)                                        # (1, N, z_dim)
        scores = self.index_net(z)[0].cpu().numpy()                      # (N,)
        if explore and self.cfg.score_noise_std > 0.0:
            scores = scores + self.cfg.score_noise_std * np.random.randn(*scores.shape)
        return _topk_action(scores, self.K)

    @torch.no_grad()
    def act_hard(self, obs: np.ndarray, **_) -> np.ndarray:
        """Greedy action. Maintains internal history; call reset_history() first."""
        if self._obs_hist is None:
            self.reset_history()
        self._obs_hist = np.roll(self._obs_hist, -1, axis=1)
        self._obs_hist[:, -1] = obs
        action = self.select_action(self._obs_hist, self._act_hist, explore=False)
        self._act_hist = np.roll(self._act_hist, -1, axis=1)
        self._act_hist[:, -1] = action.astype(np.float32)
        return action

    def get_index(self, obs_hist: np.ndarray, act_hist: np.ndarray) -> torch.Tensor:
        """Forward pass with grad for training: -> indices (N,)"""
        oh = torch.from_numpy(obs_hist[None]).float().to(self.device)
        ah = torch.from_numpy(act_hist[None]).float().to(self.device)
        z  = self._encode(oh, ah)           # (1, N, z_dim)
        return self.index_net(z)[0]         # (N,)

    def checkpoint_dict(self, epoch: int = 0, best_return: float = -float("inf")) -> Dict:
        return {
            "epoch": int(epoch), "best_return": float(best_return),
            "cfg": asdict(self.cfg),
            "encoder":   self.encoder.state_dict(),
            "index_net": self.index_net.state_dict(),
            "opt":       self.opt.state_dict(),
        }

    def load_checkpoint(self, path: str | Path) -> Tuple[int, float]:
        ckpt = torch.load(path, map_location=self.device)
        if "encoder" in ckpt:
            self.encoder.load_state_dict(ckpt["encoder"], strict=False)
        if "index_net" in ckpt:
            self.index_net.load_state_dict(ckpt["index_net"], strict=False)
        if "opt" in ckpt:
            try:
                self.opt.load_state_dict(ckpt["opt"])
            except ValueError:
                print("[resume] skipped incompatible optimizer state.")
        return int(ckpt.get("epoch", 0)), float(ckpt.get("best_return", -float("inf")))


# ---------------------------------------------------------------------------
# Episode collection (on-policy REINFORCE, single-arm, with history)
# ---------------------------------------------------------------------------

def _collect_training_episode(env: AdaptRMABEnv, agent: Agent) -> Dict:
    obs, _ = env.reset()
    N, L = env.cfg.N, agent.cfg.L

    obs_hist = np.zeros((N, L), dtype=np.float32)
    act_hist = np.zeros((N, L), dtype=np.float32)

    target_arm = int(env.rng.integers(N))
    ref_arm    = int(env.rng.integers(N))

    discounted_return = 0.0
    system_return = 0.0
    discount = 1.0
    logprob_terms: List[torch.Tensor] = []
    entropy_terms:  List[torch.Tensor] = []

    for _ in range(env.cfg.T):
        # Roll obs into history
        obs_hist = np.roll(obs_hist, -1, axis=1)
        obs_hist[:, -1] = obs

        indices = agent.get_index(obs_hist, act_hist)   # (N,) with grad

        logit = agent.cfg.sigmoid_scale * (indices[target_arm] - indices[ref_arm])
        p_act = torch.sigmoid(logit).clamp(1e-6, 1.0 - 1e-6)
        dist  = torch.distributions.Bernoulli(probs=p_act)
        a_t   = dist.sample()
        logprob_terms.append(dist.log_prob(a_t))
        entropy_terms.append(dist.entropy())

        action = np.zeros(N, dtype=np.int64)
        action[target_arm] = int(a_t.item())

        # Roll action into history
        act_hist = np.roll(act_hist, -1, axis=1)
        act_hist[:, -1] = action.astype(np.float32)

        obs, reward_vec, done, _ = env.step(action)
        system_return     += float(reward_vec.sum())
        discounted_return += discount * float(reward_vec[target_arm])
        discount          *= agent.cfg.gamma
        if done:
            break

    return {
        "target_return": float(discounted_return),
        "system_return": float(system_return),
        "logprob_sum": torch.stack(logprob_terms).sum(),
        "entropy": (torch.stack(entropy_terms).mean()
                    if entropy_terms else torch.tensor(0.0, device=agent.device)),
    }


def _save_training_plots(save_dir, epochs, ep_returns, avg10_returns, policy_losses):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].plot(epochs, ep_returns, alpha=0.4, label="ep_return")
    axes[0].plot(epochs, avg10_returns, label="avg10_return", linewidth=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Return")
    axes[0].set_title("NeurWIN+Belief Training Return (AdaptRMAB)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, policy_losses, color="tab:green", label="PolicyLoss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].set_title("NeurWIN+Belief Policy Loss")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "neurwin_training_curves.png")
    plt.savefig(path, dpi=150); plt.close(fig)
    print(f"[plot] saved {path}")


def train(env: AdaptRMABEnv, cfg: TrainConfig | None = None,
          checkpoint_dir: str | None = None) -> Agent:
    cfg = cfg or TrainConfig()
    cfg.N = env.cfg.N; cfg.K = env.cfg.K; cfg.T = env.cfg.T
    if checkpoint_dir is not None:
        cfg.save_dir = checkpoint_dir

    set_seed(cfg.seed)
    agent = Agent(N=cfg.N, K=cfg.K, cfg=cfg)

    best_return = -float("inf"); start_epoch = 0
    if cfg.resume_path:
        start_epoch, best_return = agent.load_checkpoint(cfg.resume_path)
        print(f"[resume] loaded {cfg.resume_path} at epoch {start_epoch}")

    os.makedirs(cfg.save_dir, exist_ok=True)
    metrics_path = os.path.join(cfg.save_dir, "neurwin_training_metrics.csv")
    fieldnames = ["epoch", "ep_return", "avg10_return", "best_return", "policy_loss", "entropy"]
    metrics_mode = "a" if (cfg.resume_path and Path(metrics_path).exists()) else "w"

    returns: List = []
    plot_epochs: List = []; plot_ep: List = []; plot_avg10: List = []; plot_pl: List = []

    with open(metrics_path, metrics_mode, newline="", encoding="utf-8") as mf:
        writer = csv.DictWriter(mf, fieldnames=fieldnames)
        if metrics_mode == "w":
            writer.writeheader(); mf.flush()

        for epoch in range(start_epoch, cfg.epochs):
            agent.encoder.train(); agent.index_net.train()

            batch = [_collect_training_episode(env, agent)
                     for _ in range(cfg.mini_batch_size)]

            ret_batch = torch.tensor([m["target_return"] for m in batch],
                                     device=agent.device, dtype=torch.float32)
            baseline  = ret_batch.mean()
            logprob   = torch.stack([m["logprob_sum"] for m in batch])
            entropy   = torch.stack([m["entropy"] for m in batch]).mean()

            policy_loss = -torch.mean((ret_batch - baseline).detach() * logprob)
            loss        = policy_loss - cfg.entropy_bonus * entropy

            agent.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(agent.encoder.parameters()) + list(agent.index_net.parameters()),
                cfg.grad_clip)
            agent.opt.step()

            avg_sys = float(np.mean([m["system_return"] for m in batch]))
            returns.append(avg_sys)
            avg10 = float(np.mean(returns[-10:]))

            plot_epochs.append(epoch + 1); plot_ep.append(avg_sys)
            plot_avg10.append(avg10); plot_pl.append(float(policy_loss.item()))

            if (epoch + 1) % 10 == 0:
                print(f"[NeurWIN] Epoch {epoch+1:04d} | avg10 {avg10:8.1f} | "
                      f"PolicyLoss {float(policy_loss.item()):8.4f} | "
                      f"Entropy {float(entropy.item()):7.4f}")

            if (epoch + 1) % cfg.save_every == 0:
                torch.save(agent.checkpoint_dict(epoch + 1, best_return),
                           os.path.join(cfg.save_dir, "latest.pth"))

            if avg10 > best_return:
                best_return = avg10
                best_path = os.path.join(cfg.save_dir, "best.pth")
                torch.save(agent.checkpoint_dict(epoch + 1, best_return), best_path)
                print(f"[NeurWIN] saved BEST {best_path} (avg10={avg10:.1f})")

            writer.writerow({
                "epoch": epoch + 1, "ep_return": avg_sys, "avg10_return": avg10,
                "best_return": best_return, "policy_loss": float(policy_loss.item()),
                "entropy": float(entropy.item()),
            })
            mf.flush()

    _save_training_plots(cfg.save_dir, plot_epochs, plot_ep, plot_avg10, plot_pl)
    return agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--L", type=int, default=40)
    parser.add_argument("--z_dim", type=int, default=64)
    parser.add_argument("--save_dir", type=str, default="checkpoints_neurwin")
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()

    cfg = TrainConfig(
        N=args.N, K=args.K, T=args.T, seed=args.seed, epochs=args.epochs,
        L=args.L, z_dim=args.z_dim, save_dir=args.save_dir,
        resume_path=args.resume,
    )
    env = AdaptRMABEnv(AdaptRMABConfig(N=cfg.N, K=cfg.K, T=cfg.T), seed=cfg.seed)
    train(env, cfg=cfg)


if __name__ == "__main__":
    main()
