from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from diffusion_DPMD_train import BeliefEncoder
from env import AdaptRMABConfig, AdaptRMABEnv


def _topk(scores: np.ndarray, K: int) -> np.ndarray:
    a = np.zeros(scores.shape[0], dtype=np.int32)
    a[np.argsort(-scores)[:K]] = 1
    return a


@dataclass
class LearnedRolloutConfig:
    N: int = 20
    K: int = 5
    T: int = 100
    L: int = 40
    z_dim: int = 64
    encoder_hidden: int = 128
    encoder_heads: int = 4
    encoder_layers: int = 3
    head_hidden: int = 128
    H: int = 10
    gamma: float = 0.99
    epochs: int = 50
    episodes_per_epoch: int = 10
    batch_size: int = 256
    lr: float = 3e-4
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "checkpoints_learned_rollout"


class NextObsHead(nn.Module):
    def __init__(self, z_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + 1, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        b, n, _ = z.shape
        x = torch.cat([z, action.unsqueeze(-1)], dim=-1)
        return self.net(x.reshape(b * n, -1)).reshape(b, n)


class LearnedRolloutAgent:
    def __init__(self, cfg: LearnedRolloutConfig):
        self.cfg = cfg
        self.N = cfg.N
        self.K = cfg.K
        self.device = torch.device(cfg.device)
        self.encoder = BeliefEncoder(
            z_dim=cfg.z_dim, hidden_dim=cfg.encoder_hidden,
            n_heads=cfg.encoder_heads, n_layers=cfg.encoder_layers, L=cfg.L,
        ).to(self.device)
        self.next_obs = NextObsHead(cfg.z_dim, cfg.head_hidden).to(self.device)
        self._obs_hist: np.ndarray | None = None
        self._act_hist: np.ndarray | None = None

    def reset_history(self):
        self._obs_hist = np.zeros((self.N, self.cfg.L), dtype=np.float32)
        self._act_hist = np.zeros((self.N, self.cfg.L), dtype=np.float32)

    @torch.no_grad()
    def _predict_next(self, obs_hist: np.ndarray, act_hist: np.ndarray,
                      action: np.ndarray) -> np.ndarray:
        oh = torch.from_numpy(obs_hist[None]).float().to(self.device)
        ah = torch.from_numpy(act_hist[None]).float().to(self.device)
        aa = torch.from_numpy(action[None].astype(np.float32)).to(self.device)
        z = self.encoder(oh, ah)
        return self.next_obs(z, aa)[0].cpu().numpy()

    @torch.no_grad()
    def act_hard(self, obs: np.ndarray, **_) -> np.ndarray:
        if self._obs_hist is None:
            self.reset_history()
        self._obs_hist = np.roll(self._obs_hist, -1, axis=1)
        self._obs_hist[:, -1] = obs.astype(np.float32)

        obs_a = self._obs_hist.copy()
        obs_p = self._obs_hist.copy()
        act_a = self._act_hist.copy()
        act_p = self._act_hist.copy()
        scores = np.zeros(self.N, dtype=np.float64)
        for h in range(self.cfg.H):
            a_now = np.ones(self.N, dtype=np.float32) if h == 0 else np.zeros(self.N, dtype=np.float32)
            p_now = np.zeros(self.N, dtype=np.float32)
            next_a = self._predict_next(obs_a, act_a, a_now)
            next_p = self._predict_next(obs_p, act_p, p_now)
            scores += (self.cfg.gamma ** (h + 1)) * (
                np.maximum(next_a, 0.0) - np.maximum(next_p, 0.0)
            )
            obs_a = np.roll(obs_a, -1, axis=1); obs_a[:, -1] = next_a
            obs_p = np.roll(obs_p, -1, axis=1); obs_p[:, -1] = next_p
            act_a = np.roll(act_a, -1, axis=1); act_a[:, -1] = a_now
            act_p = np.roll(act_p, -1, axis=1); act_p[:, -1] = p_now

        action = _topk(scores, self.K)
        self._act_hist = np.roll(self._act_hist, -1, axis=1)
        self._act_hist[:, -1] = action.astype(np.float32)
        return action

    def load_checkpoint(self, path: str | Path):
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"], strict=False)
        self.next_obs.load_state_dict(ckpt["next_obs"], strict=False)

    def checkpoint_dict(self):
        return {
            "cfg": asdict(self.cfg),
            "encoder": self.encoder.state_dict(),
            "next_obs": self.next_obs.state_dict(),
        }


def train(cfg: LearnedRolloutConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    env_cfg = AdaptRMABConfig(N=cfg.N, K=cfg.K, T=cfg.T)
    agent = LearnedRolloutAgent(cfg)
    opt = torch.optim.Adam(
        list(agent.encoder.parameters()) + list(agent.next_obs.parameters()),
        lr=cfg.lr,
    )
    rng = np.random.default_rng(cfg.seed)

    for epoch in range(1, cfg.epochs + 1):
        losses = []
        for ep in range(cfg.episodes_per_epoch):
            env = AdaptRMABEnv(env_cfg, seed=cfg.seed * 10000 + epoch * 100 + ep)
            obs, _ = env.reset()
            obs_hist = np.zeros((cfg.N, cfg.L), dtype=np.float32)
            act_hist = np.zeros((cfg.N, cfg.L), dtype=np.float32)
            samples = []
            for _ in range(cfg.T):
                obs_hist = np.roll(obs_hist, -1, axis=1)
                obs_hist[:, -1] = obs.astype(np.float32)
                action = np.zeros(cfg.N, dtype=np.int32)
                action[rng.choice(cfg.N, size=cfg.K, replace=False)] = 1
                next_obs, _, done, _ = env.step(action)
                samples.append((obs_hist.copy(), act_hist.copy(), action.astype(np.float32), next_obs.astype(np.float32)))
                act_hist = np.roll(act_hist, -1, axis=1)
                act_hist[:, -1] = action.astype(np.float32)
                obs = next_obs
                if done:
                    break
            rng.shuffle(samples)
            for start in range(0, len(samples), cfg.batch_size):
                batch = samples[start:start + cfg.batch_size]
                oh = torch.from_numpy(np.stack([b[0] for b in batch])).float().to(agent.device)
                ah = torch.from_numpy(np.stack([b[1] for b in batch])).float().to(agent.device)
                ac = torch.from_numpy(np.stack([b[2] for b in batch])).float().to(agent.device)
                target = torch.from_numpy(np.stack([b[3] for b in batch])).float().to(agent.device)
                pred = agent.next_obs(agent.encoder(oh, ah), ac)
                loss = F.mse_loss(pred, target)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(agent.encoder.parameters()) + list(agent.next_obs.parameters()), 1.0)
                opt.step()
                losses.append(float(loss.item()))
        if epoch == 1 or epoch % 10 == 0:
            print(f"[learned-rollout] epoch {epoch:04d}/{cfg.epochs} loss={np.mean(losses):.5f}")
        torch.save(agent.checkpoint_dict(), Path(cfg.save_dir) / "best.pth")
    print(f"[learned-rollout] saved {Path(cfg.save_dir) / 'best.pth'}")
    return agent


def load_learned_rollout(path: str, env_cfg: AdaptRMABConfig) -> LearnedRolloutAgent:
    ckpt = torch.load(path, map_location="cpu")
    saved = ckpt.get("cfg", {})
    cfg = LearnedRolloutConfig(N=env_cfg.N, K=env_cfg.K, T=env_cfg.T)
    for key, value in saved.items():
        if hasattr(cfg, key) and key not in {"N", "K", "T", "device"}:
            setattr(cfg, key, value)
    agent = LearnedRolloutAgent(cfg)
    agent.load_checkpoint(path)
    agent.encoder.eval()
    agent.next_obs.eval()
    return agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--episodes_per_epoch", type=int, default=10)
    parser.add_argument("--L", type=int, default=40)
    parser.add_argument("--z_dim", type=int, default=64)
    parser.add_argument("--H", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints_learned_rollout")
    args = parser.parse_args()
    cfg = LearnedRolloutConfig(
        N=args.N, K=args.K, T=args.T, epochs=args.epochs,
        episodes_per_epoch=args.episodes_per_epoch, L=args.L,
        z_dim=args.z_dim, H=args.H, seed=args.seed, save_dir=args.ckpt_dir,
    )
    train(cfg)


if __name__ == "__main__":
    main()
