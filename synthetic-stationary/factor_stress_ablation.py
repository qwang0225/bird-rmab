"""
Factor stress ablation for synthetic stationary RMAB.

This isolates the two reviewer-mentioned factors:
  1. partial_obs_known_dyn: noisy observations, but fixed known homogeneous dynamics.
  2. full_obs_unknown_dyn: full observations, but hidden heterogeneous fixed dynamics.

Default methods are non-oracle: obs_greedy, activation_greedy, learned_rollout,
and BIRD/DPMD.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Callable

import numpy as np

from baselines import ActivationGreedyPolicy, GreedyObsPolicy, RandomPolicy
from env import AdaptRMABConfig, AdaptRMABEnv, TYPE_ALPHA_MEAN, TYPE_BETA_MEAN

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


METHODS = ("obs_greedy", "activation_greedy", "learned_rollout", "dpmd")
SETTINGS = ("partial_obs_known_dyn", "full_obs_unknown_dyn")
SETTING_LABELS = {
    "partial_obs_known_dyn": "Partial obs + known dyn",
    "full_obs_unknown_dyn": "Full obs + unknown dyn",
}
METHOD_LABELS = {
    "random": "Random",
    "obs_greedy": "Obs greedy",
    "activation_greedy": "Activation greedy",
    "learned_rollout": "Learned rollout",
    "dpmd": "BIRD",
}


class KnownDynamicsEnv(AdaptRMABEnv):
    """AdaptRMAB variant with fixed homogeneous dynamics and noisy observations."""

    def __init__(
        self,
        cfg: AdaptRMABConfig,
        seed: int = 0,
        alpha_known: float | None = None,
        beta_known: float | None = None,
    ):
        super().__init__(cfg, seed=seed)
        self.alpha_known = float(np.mean(TYPE_ALPHA_MEAN) if alpha_known is None else alpha_known)
        self.beta_known = float(np.mean(TYPE_BETA_MEAN) if beta_known is None else beta_known)

    def reset(self, seed: int | None = None):
        obs, _ = super().reset(seed=seed)
        del obs
        self.theta = np.zeros(self.cfg.N, dtype=np.int64)
        self.alpha = np.full(self.cfg.N, self.alpha_known, dtype=np.float32)
        self.beta = np.full(self.cfg.N, self.beta_known, dtype=np.float32)
        return self._observe(), self._make_info()

    def step(self, a: np.ndarray):
        a = np.asarray(a, dtype=np.int32).reshape(-1)
        cfg = self.cfg
        assert a.shape[0] == cfg.N
        assert int(a.sum()) <= cfg.K, f"budget violated: {int(a.sum())} > {cfg.K}"

        reward_vec = np.maximum(self.x, 0.0).astype(np.float32)
        w = (cfg.sigma_w * self.rng.standard_normal(cfg.N)).astype(np.float32)
        self.x = (
            self.alpha * self.x
            + self.beta * a.astype(np.float32)
            - cfg.drift
            + w
        ).astype(np.float32)

        self.t += 1
        done = self.t >= cfg.T
        info = self._make_info()
        info["reward_per_arm"] = reward_vec
        return self._observe(), reward_vec, done, info


def _make_setting(
    setting: str,
    args: argparse.Namespace,
) -> tuple[AdaptRMABConfig, Callable[[int], AdaptRMABEnv], dict]:
    if setting == "partial_obs_known_dyn":
        cfg = AdaptRMABConfig(
            N=args.N,
            K=args.K,
            T=args.T,
            sigma_v=args.partial_sigma_v,
            sigma_w=args.sigma_w,
            drift=args.drift,
        )
        alpha_known = args.known_alpha
        beta_known = args.known_beta

        def factory(seed: int) -> AdaptRMABEnv:
            return KnownDynamicsEnv(
                cfg,
                seed=seed,
                alpha_known=alpha_known,
                beta_known=beta_known,
            )

        meta = {
            "sigma_v": cfg.sigma_v,
            "sigma_w": cfg.sigma_w,
            "known_alpha": float(np.mean(TYPE_ALPHA_MEAN) if alpha_known is None else alpha_known),
            "known_beta": float(np.mean(TYPE_BETA_MEAN) if beta_known is None else beta_known),
            "unknown_dynamics": False,
        }
        return cfg, factory, meta

    if setting == "full_obs_unknown_dyn":
        cfg = AdaptRMABConfig(
            N=args.N,
            K=args.K,
            T=args.T,
            sigma_v=0.0,
            sigma_w=args.sigma_w,
            drift=args.drift,
        )

        def factory(seed: int) -> AdaptRMABEnv:
            return AdaptRMABEnv(cfg, seed=seed)

        meta = {
            "sigma_v": cfg.sigma_v,
            "sigma_w": cfg.sigma_w,
            "dynamics": "hidden stationary per-type alpha/beta",
            "unknown_dynamics": True,
        }
        return cfg, factory, meta

    raise ValueError(f"Unknown setting: {setting}")


def _read_training_time(save_dir: Path) -> float:
    path = save_dir / "training_time.json"
    if not path.exists():
        return float("nan")
    return float(json.loads(path.read_text(encoding="utf-8")).get("training_time_sec", float("nan")))


def _load_or_train_bird(
    args: argparse.Namespace,
    setting: str,
    env_cfg: AdaptRMABConfig,
    env_factory: Callable[[int], AdaptRMABEnv],
):
    import diffusion_DPMD_train as dpmd

    save_dir = Path(args.save_dir_template.format(
        setting=setting,
        N=env_cfg.N,
        K=env_cfg.K,
        seed=args.seed,
    ))
    ckpt_path = save_dir / "best.pth"
    latest_path = save_dir / "latest.pth"
    cfg = dpmd.DPMDTrainConfig(
        N=env_cfg.N,
        K=env_cfg.K,
        T=env_cfg.T,
        seed=args.seed,
        epochs=args.epochs,
        L=args.L,
        z_dim=args.z_dim,
        save_dir=str(save_dir),
    )
    if ckpt_path.exists() and not args.force_retrain:
        agent = dpmd.DPMDAgent(N=env_cfg.N, K=env_cfg.K, cfg=cfg)
        agent.load_checkpoint(ckpt_path)
        return agent, save_dir
    if args.resume_latest and latest_path.exists():
        cfg.resume_path = str(latest_path)

    env = env_factory(args.seed)
    agent = dpmd.train(env, cfg=cfg, checkpoint_dir=cfg.save_dir)
    return agent, save_dir


def _load_or_train_learned_rollout(
    args: argparse.Namespace,
    setting: str,
    env_cfg: AdaptRMABConfig,
    env_factory: Callable[[int], AdaptRMABEnv],
):
    import torch
    import torch.nn.functional as F
    from learned_rollout import LearnedRolloutAgent, LearnedRolloutConfig

    save_dir = Path(args.learned_save_dir_template.format(
        setting=setting,
        N=env_cfg.N,
        K=env_cfg.K,
        seed=args.seed,
    ))
    ckpt_path = save_dir / "best.pth"
    cfg = LearnedRolloutConfig(
        N=env_cfg.N,
        K=env_cfg.K,
        T=env_cfg.T,
        L=args.L,
        z_dim=args.z_dim,
        H=args.rollout_H,
        epochs=args.learned_epochs,
        episodes_per_epoch=args.learned_episodes_per_epoch,
        seed=args.seed,
        save_dir=str(save_dir),
    )
    agent = LearnedRolloutAgent(cfg)
    if ckpt_path.exists() and not args.force_retrain:
        agent.load_checkpoint(ckpt_path)
        agent.encoder.eval()
        agent.next_obs.eval()
        return agent, save_dir, _read_training_time(save_dir)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    save_dir.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.Adam(
        list(agent.encoder.parameters()) + list(agent.next_obs.parameters()),
        lr=cfg.lr,
    )
    rng = np.random.default_rng(cfg.seed)
    train_start = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        losses = []
        for ep in range(cfg.episodes_per_epoch):
            env = env_factory(cfg.seed * 10000 + epoch * 100 + ep)
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
                samples.append((
                    obs_hist.copy(),
                    act_hist.copy(),
                    action.astype(np.float32),
                    next_obs.astype(np.float32),
                ))
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
            print(f"[learned-rollout:{setting}] epoch {epoch:04d}/{cfg.epochs} "
                  f"loss={np.mean(losses):.5f}")
        torch.save(agent.checkpoint_dict(), ckpt_path)

    train_sec = time.perf_counter() - train_start
    (save_dir / "training_time.json").write_text(json.dumps({
        "method": "learned_rollout",
        "env_name": f"synthetic-stationary-{setting}",
        "training_time_sec": train_sec,
        "completed_epochs": cfg.epochs,
        "N": cfg.N,
        "K": cfg.K,
        "T": cfg.T,
        "L": cfg.L,
        "seed": cfg.seed,
    }, indent=2), encoding="utf-8")
    agent.encoder.eval()
    agent.next_obs.eval()
    return agent, save_dir, train_sec


def _evaluate_policy_on_factory(
    policy,
    env_cfg: AdaptRMABConfig,
    env_factory: Callable[[int], AdaptRMABEnv],
    n_episodes: int,
    seed_offset: int,
):
    returns = []
    action_time = 0.0
    action_count = 0
    for ep in range(n_episodes):
        env = env_factory(seed_offset + ep)
        obs, _ = env.reset()
        if hasattr(policy, "reset"):
            policy.reset()
        ep_return = 0.0
        for _ in range(env_cfg.T):
            t0 = time.perf_counter()
            action = policy.act(obs)
            action_time += time.perf_counter() - t0
            action_count += 1
            obs, reward_vec, done, _ = env.step(action)
            ep_return += float(reward_vec.sum())
            if done:
                break
        returns.append(ep_return)
    latency_ms = 1000.0 * action_time / max(action_count, 1)
    return np.asarray(returns, dtype=np.float32), latency_ms


def _evaluate_agent_on_factory(
    agent,
    env_cfg: AdaptRMABConfig,
    env_factory: Callable[[int], AdaptRMABEnv],
    n_episodes: int,
    seed_offset: int,
):
    returns = []
    action_time = 0.0
    action_count = 0
    for ep in range(n_episodes):
        env = env_factory(seed_offset + ep)
        obs, _ = env.reset()
        agent.reset_history()
        ep_return = 0.0
        for _ in range(env_cfg.T):
            t0 = time.perf_counter()
            action = agent.act_hard(obs)
            action_time += time.perf_counter() - t0
            action_count += 1
            obs, reward_vec, done, _ = env.step(action)
            ep_return += float(reward_vec.sum())
            if done:
                break
        returns.append(ep_return)
    latency_ms = 1000.0 * action_time / max(action_count, 1)
    return np.asarray(returns, dtype=np.float32), latency_ms


def _make_policy(method: str, cfg: AdaptRMABConfig, seed: int):
    if method == "random":
        return RandomPolicy(cfg, seed=seed)
    if method == "obs_greedy":
        return GreedyObsPolicy(cfg)
    if method == "activation_greedy":
        return ActivationGreedyPolicy(cfg)
    raise ValueError(f"No hand-coded policy for method: {method}")


def _plot_results(
    setting_names: list[str],
    method_names: list[str],
    mean_returns: np.ndarray,
    std_returns: np.ndarray,
    latency_ms: np.ndarray,
    out_path: Path,
) -> None:
    if plt is None:
        print(f"[plot skipped] matplotlib unavailable; retained numeric results in {out_path.with_suffix('.npz')}")
        return
    labels = [SETTING_LABELS.get(name, name) for name in setting_names]
    x = np.arange(len(labels))
    width = 0.8 / max(len(method_names), 1)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for m_idx, method in enumerate(method_names):
        offset = (m_idx - (len(method_names) - 1) / 2.0) * width
        display = METHOD_LABELS.get(method, method)
        axes[0].bar(
            x + offset,
            mean_returns[:, m_idx],
            yerr=std_returns[:, m_idx],
            width=width,
            capsize=3,
            label=display,
        )
        axes[1].bar(
            x + offset,
            latency_ms[:, m_idx],
            width=width,
            label=display,
        )

    axes[0].set_ylabel("Episode return")
    axes[1].set_ylabel("Latency (ms/decision)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=12, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close(fig)


def _save_npz(
    path: Path,
    args: argparse.Namespace,
    setting_names: list[str],
    method_names: list[str],
    setting_metadata: list[dict],
    raw_returns: np.ndarray,
    mean_returns: np.ndarray,
    std_returns: np.ndarray,
    latency_ms: np.ndarray,
    train_time_sec: np.ndarray,
    completed_mask: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        setting_names=np.asarray(setting_names),
        setting_labels=np.asarray([SETTING_LABELS.get(s, s) for s in setting_names]),
        method_names=np.asarray(method_names),
        method_labels=np.asarray([METHOD_LABELS.get(m, m) for m in method_names]),
        setting_metadata_json=np.asarray([json.dumps(m, sort_keys=True) for m in setting_metadata]),
        raw_returns=raw_returns,
        mean_returns=mean_returns,
        std_returns=std_returns,
        latency_ms_per_decision=latency_ms,
        train_time_sec=train_time_sec,
        train_time_min=train_time_sec / 60.0,
        completed_mask=completed_mask,
        N=np.asarray(args.N, dtype=np.int32),
        K=np.asarray(args.K, dtype=np.int32),
        T=np.asarray(args.T, dtype=np.int32),
        L=np.asarray(args.L, dtype=np.int32),
        n_episodes=np.asarray(args.n_episodes, dtype=np.int32),
        seed=np.asarray(args.seed, dtype=np.int32),
        env_name=np.asarray("synthetic-stationary-factor-stress"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=list(SETTINGS))
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--L", type=int, default=40)
    parser.add_argument("--z_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learned_epochs", type=int, default=50)
    parser.add_argument("--learned_episodes_per_epoch", type=int, default=10)
    parser.add_argument("--rollout_H", type=int, default=10)
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--partial_sigma_v", type=float, default=0.50)
    parser.add_argument("--sigma_w", type=float, default=0.40)
    parser.add_argument("--drift", type=float, default=0.30)
    parser.add_argument("--known_alpha", type=float, default=None)
    parser.add_argument("--known_beta", type=float, default=None)
    parser.add_argument("--force_retrain", action="store_true")
    parser.add_argument("--resume_latest", action="store_true")
    parser.add_argument(
        "--save_dir_template",
        default="checkpoints_dpmd_factor_{setting}_N{N}_K{K}_seed{seed}",
    )
    parser.add_argument(
        "--learned_save_dir_template",
        default="checkpoints_learned_rollout_factor_{setting}_N{N}_K{K}_seed{seed}",
    )
    parser.add_argument("--out_dir", default="experiment_outputs/factor_stress")
    parser.add_argument("--out_stem", default="factor_stress_ablation")
    args = parser.parse_args()

    setting_names = list(args.settings)
    method_names = list(args.methods)
    raw_returns = np.full(
        (len(setting_names), len(method_names), args.n_episodes),
        np.nan,
        dtype=np.float32,
    )
    mean_returns = np.full((len(setting_names), len(method_names)), np.nan, dtype=np.float32)
    std_returns = np.full((len(setting_names), len(method_names)), np.nan, dtype=np.float32)
    latency_ms = np.full((len(setting_names), len(method_names)), np.nan, dtype=np.float32)
    train_time_sec = np.full((len(setting_names), len(method_names)), np.nan, dtype=np.float32)
    completed_mask = np.zeros((len(setting_names), len(method_names)), dtype=bool)
    setting_metadata: list[dict] = []

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = out_dir / args.out_stem

    for setting_idx, setting in enumerate(setting_names):
        env_cfg, env_factory, meta = _make_setting(setting, args)
        setting_metadata.append(meta)
        bird_agent = None
        bird_save_dir = None
        learned_agent = None
        learned_save_dir = None
        learned_train_sec = float("nan")

        for method_idx, method in enumerate(method_names):
            print("=" * 72)
            print(f"Factor stress: setting={setting} method={method} seed={args.seed}")
            print("=" * 72)

            if method == "dpmd":
                if bird_agent is None:
                    bird_agent, bird_save_dir = _load_or_train_bird(
                        args, setting, env_cfg, env_factory)
                returns, latency = _evaluate_agent_on_factory(
                    bird_agent, env_cfg, env_factory, args.n_episodes, args.seed + 1000)
                train_sec = _read_training_time(bird_save_dir)
            elif method == "learned_rollout":
                if learned_agent is None:
                    learned_agent, learned_save_dir, learned_train_sec = _load_or_train_learned_rollout(
                        args, setting, env_cfg, env_factory)
                returns, latency = _evaluate_agent_on_factory(
                    learned_agent, env_cfg, env_factory, args.n_episodes, args.seed + 1000)
                train_sec = learned_train_sec
            else:
                policy = _make_policy(method, env_cfg, args.seed)
                returns, latency = _evaluate_policy_on_factory(
                    policy, env_cfg, env_factory, args.n_episodes, args.seed + 1000)
                train_sec = float("nan")

            raw_returns[setting_idx, method_idx] = returns
            mean_returns[setting_idx, method_idx] = returns.mean()
            std_returns[setting_idx, method_idx] = returns.std()
            latency_ms[setting_idx, method_idx] = latency
            train_time_sec[setting_idx, method_idx] = train_sec
            completed_mask[setting_idx, method_idx] = True
            print(
                f"[{setting} {method}] mean={returns.mean():.2f} std={returns.std():.2f} "
                f"latency={latency:.3f} ms/decision train_min={train_sec / 60.0:.2f}"
            )
            _save_npz(
                Path(f"{out_stem}.npz"),
                args,
                setting_names,
                method_names,
                setting_metadata,
                raw_returns,
                mean_returns,
                std_returns,
                latency_ms,
                train_time_sec,
                completed_mask,
            )
            print(f"[partial saved] {out_stem}.npz")

    _save_npz(
        Path(f"{out_stem}.npz"),
        args,
        setting_names,
        method_names,
        setting_metadata,
        raw_returns,
        mean_returns,
        std_returns,
        latency_ms,
        train_time_sec,
        completed_mask,
    )
    _plot_results(
        setting_names,
        method_names,
        mean_returns,
        std_returns,
        latency_ms,
        Path(f"{out_stem}.png"),
    )
    print(f"[saved] {out_stem}.npz")
    if Path(f"{out_stem}.png").exists():
        print(f"[saved] {out_stem}.png")


if __name__ == "__main__":
    main()
