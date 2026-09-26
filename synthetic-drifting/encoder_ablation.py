"""
2D encoder-history ablation for BIRD/DPMD on synthetic-drifting.

Compares Transformer, LSTM, vanilla RNN, and MLP encoders across multiple
history lengths while keeping the diffusion actor, critic, replay, and training
loop unchanged.

Outputs:
  encoder_ablation_N{N}_K{K}.npz
  encoder_ablation_N{N}_K{K}.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import diffusion_DPMD_train as dpmd
from env import AdaptRMABConfig, AdaptRMABEnv


OriginalTransformerEncoder = dpmd.BeliefEncoder


class LSTMEncoder(nn.Module):
    def __init__(self, z_dim: int, hidden_dim: int = 64,
                 n_heads: int = 4, n_layers: int = 2, L: int = 20):
        super().__init__()
        self.rnn = nn.LSTM(2, hidden_dim, num_layers=n_layers, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, z_dim)

    def forward(self, obs_hist: torch.Tensor, act_hist: torch.Tensor) -> torch.Tensor:
        batch, n_arms, hist_len = obs_hist.shape
        x = torch.stack([obs_hist, act_hist], dim=-1).reshape(batch * n_arms, hist_len, 2)
        y, _ = self.rnn(x)
        return self.out_proj(y[:, -1, :]).reshape(batch, n_arms, -1)


class RNNEncoder(nn.Module):
    def __init__(self, z_dim: int, hidden_dim: int = 64,
                 n_heads: int = 4, n_layers: int = 2, L: int = 20):
        super().__init__()
        self.rnn = nn.RNN(2, hidden_dim, num_layers=n_layers,
                          nonlinearity="tanh", batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, z_dim)

    def forward(self, obs_hist: torch.Tensor, act_hist: torch.Tensor) -> torch.Tensor:
        batch, n_arms, hist_len = obs_hist.shape
        x = torch.stack([obs_hist, act_hist], dim=-1).reshape(batch * n_arms, hist_len, 2)
        y, _ = self.rnn(x)
        return self.out_proj(y[:, -1, :]).reshape(batch, n_arms, -1)


class MLPEncoder(nn.Module):
    def __init__(self, z_dim: int, hidden_dim: int = 64,
                 n_heads: int = 4, n_layers: int = 2, L: int = 20):
        super().__init__()
        self.L = int(L)
        self.net = nn.Sequential(
            nn.Linear(2 * self.L, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, obs_hist: torch.Tensor, act_hist: torch.Tensor) -> torch.Tensor:
        batch, n_arms, hist_len = obs_hist.shape
        x = torch.cat([obs_hist, act_hist], dim=-1).reshape(batch * n_arms, 2 * hist_len)
        return self.net(x).reshape(batch, n_arms, -1)


ENCODERS = {
    "transformer": OriginalTransformerEncoder,
    "lstm": LSTMEncoder,
    "rnn": RNNEncoder,
    "mlp": MLPEncoder,
}


def _load_baseline_agent(ckpt_path: str, env_cfg: AdaptRMABConfig):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    saved = ckpt.get("cfg", {})
    cfg = dpmd.DPMDTrainConfig()
    cfg.N = env_cfg.N; cfg.K = env_cfg.K; cfg.T = env_cfg.T
    for key in ("z_dim", "encoder_hidden", "encoder_heads", "encoder_layers", "L",
                "actor_hidden", "actor_t_dim", "T_diff", "score_clip",
                "critic_hidden", "action_candidates", "target_action_candidates"):
        if key in saved:
            setattr(cfg, key, saved[key])
    agent = dpmd.DPMDAgent(N=cfg.N, K=cfg.K, cfg=cfg)
    agent.load_checkpoint(ckpt_path)
    return agent, saved


def _matches_baseline(saved: dict, args: argparse.Namespace, name: str, hist_len: int) -> bool:
    return (
        name == "transformer"
        and int(saved.get("N", args.N)) == args.N
        and int(saved.get("K", args.K)) == args.K
        and int(saved.get("T", args.T)) == args.T
        and int(saved.get("L", hist_len)) == hist_len
    )


def evaluate_agent(agent, env_cfg: AdaptRMABConfig, n_episodes: int, seed: int) -> np.ndarray:
    returns = []
    for ep in range(n_episodes):
        env = AdaptRMABEnv(env_cfg, seed=seed + ep)
        obs, _ = env.reset()
        agent.reset_history()
        ep_return = 0.0
        for _ in range(env_cfg.T):
            action = agent.act_hard(obs)
            obs, reward_vec, done, _ = env.step(action)
            ep_return += float(reward_vec.sum())
            if done:
                break
        returns.append(ep_return)
    return np.asarray(returns, dtype=np.float32)


def plot_grid(encoder_names: list[str], history_lengths: list[int],
              mean_returns: np.ndarray, std_returns: np.ndarray, out_path: str) -> None:
    colors = {
        "transformer": "#2ca02c",
        "lstm": "#1f77b4",
        "rnn": "#ff7f0e",
        "mlp": "#9467bd",
    }
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    x = np.asarray(history_lengths)
    for enc_idx, name in enumerate(encoder_names):
        y = mean_returns[enc_idx]
        err = std_returns[enc_idx]
        ax.plot(x, y, marker="o", linewidth=2, label=name, color=colors.get(name))
        ax.fill_between(x, y - err, y + err, color=colors.get(name), alpha=0.12)
    ax.set_xlabel("History length L")
    ax.set_ylabel("Episode return")
    ax.set_title("synthetic-drifting encoder/history ablation")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--train_seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--history_lengths", nargs="+", type=int, default=[1, 10, 20, 40, 80])
    parser.add_argument("--encoders", nargs="+", default=list(ENCODERS.keys()),
                        choices=list(ENCODERS.keys()))
    parser.add_argument("--baseline_ckpt", default="checkpoints_dpmd/best.pth")
    parser.add_argument("--force_retrain", action="store_true")
    parser.add_argument("--out_stem", default=None)
    parser.add_argument("--out_dir", default="experiment_outputs")
    args = parser.parse_args()

    env_cfg = AdaptRMABConfig(N=args.N, K=args.K, T=args.T)
    raw_returns = np.zeros(
        (len(args.encoders), len(args.history_lengths), len(args.train_seeds), args.n_episodes),
        dtype=np.float32,
    )

    try:
        for enc_idx, name in enumerate(args.encoders):
            dpmd.BeliefEncoder = ENCODERS[name]
            for len_idx, hist_len in enumerate(args.history_lengths):
                for seed_idx, seed in enumerate(args.train_seeds):
                    print("=" * 72)
                    ckpt_path = Path(args.baseline_ckpt)
                    use_baseline = False
                    variant_path = Path(f"checkpoints_dpmd_encoder_{name}_L{hist_len}_seed{seed}") / "best.pth"
                    if variant_path.exists() and not args.force_retrain:
                        ckpt_path = variant_path
                        use_baseline = True
                    if not use_baseline and name == "transformer" and ckpt_path.exists() and not args.force_retrain:
                        _, saved = _load_baseline_agent(str(ckpt_path), env_cfg)
                        use_baseline = _matches_baseline(saved, args, name, hist_len)
                    action = "Loading" if use_baseline else "Training"
                    print(f"{action} encoder={name} L={hist_len} seed={seed}")
                    print("=" * 72)
                    if use_baseline:
                        agent, _ = _load_baseline_agent(str(ckpt_path), env_cfg)
                    else:
                        cfg = dpmd.DPMDTrainConfig(
                            N=args.N, K=args.K, T=args.T, L=hist_len,
                            seed=seed, epochs=args.epochs,
                            save_dir=f"checkpoints_dpmd_encoder_{name}_L{hist_len}_seed{seed}",
                        )
                        env = AdaptRMABEnv(env_cfg, seed=seed)
                        agent = dpmd.train(env, cfg=cfg, checkpoint_dir=cfg.save_dir)
                    returns = evaluate_agent(agent, env_cfg, args.n_episodes, seed=seed + 1000)
                    raw_returns[enc_idx, len_idx, seed_idx] = returns
                    print(f"[{name} L={hist_len} seed={seed}] "
                          f"mean={returns.mean():.2f} std={returns.std():.2f}")
    finally:
        dpmd.BeliefEncoder = OriginalTransformerEncoder

    flat = raw_returns.reshape(len(args.encoders), len(args.history_lengths), -1)
    mean_returns = flat.mean(axis=-1)
    std_returns = flat.std(axis=-1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / (args.out_stem or f"encoder_ablation_N{args.N}_K{args.K}")
    np.savez(
        f"{stem}.npz",
        encoder_names=np.asarray(args.encoders),
        history_lengths=np.asarray(args.history_lengths, dtype=np.int32),
        train_seeds=np.asarray(args.train_seeds, dtype=np.int32),
        raw_returns=raw_returns,
        mean_returns=mean_returns,
        std_returns=std_returns,
    )
    plot_grid(args.encoders, args.history_lengths, mean_returns, std_returns, f"{stem}.png")
    print(f"[saved] {stem}.npz")
    print(f"[saved] {stem}.png")


if __name__ == "__main__":
    main()
