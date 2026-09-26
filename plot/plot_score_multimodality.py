"""Diagnostic plot for stochastic BIRD score/rank distributions.

Run from the repository root after restoring the corresponding checkpoints, e.g.

  python plot_score_multimodality.py --env synthetic-drifting
  python plot_score_multimodality.py --env mimic-icu

The script fixes belief histories from an evaluation rollout, samples many score
vectors from the trained BIRD diffusion actor, and compares the resulting
rank/Top-K distribution against the deterministic MLP actor.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ENV_SPECS = {
    "synthetic-drifting": {
        "dir": "synthetic-drifting",
        "cfg": "AdaptRMABConfig",
        "env": "AdaptRMABEnv",
        "bird_ckpt": "checkpoints_dpmd/best.pth",
        "mlp_ckpt": "checkpoints_mlp_actor/best.pth",
        "seed_offset": 1000,
    },
    "mimic-icu": {
        "dir": "mimic-icu",
        "cfg": "MIMICRMABConfig",
        "env": "MIMICRMABEnv",
        "bird_ckpt": "checkpoints_dpmd/best.pth",
        "mlp_ckpt": "checkpoints_mlp_actor/best.pth",
        "seed_offset": 2000,
    },
}


def _clear_local_modules() -> None:
    for name in [
        "env", "diffusion_model", "diffusion_DPMD_train", "mlp_actor",
        "actor_ablation", "baselines",
    ]:
        sys.modules.pop(name, None)


def _import_from_env(env_dir: Path, module_name: str):
    _clear_local_modules()
    sys.path.insert(0, str(env_dir))
    try:
        return importlib.import_module(module_name)
    finally:
        try:
            sys.path.remove(str(env_dir))
        except ValueError:
            pass


def _load_agents(spec: dict, env_cfg, env_dir: Path, bird_ckpt: Path, mlp_ckpt: Path):
    if not bird_ckpt.exists():
        raise FileNotFoundError(f"Missing BIRD checkpoint: {bird_ckpt}")
    if not mlp_ckpt.exists():
        raise FileNotFoundError(f"Missing MLP checkpoint: {mlp_ckpt}")

    _clear_local_modules()
    sys.path.insert(0, str(env_dir))
    try:
        actor_ablation = importlib.import_module("actor_ablation")
        bird = actor_ablation._load_dpmd(str(bird_ckpt), env_cfg)
        mlp = actor_ablation._load_mlp_actor(str(mlp_ckpt), env_cfg)
    finally:
        try:
            sys.path.remove(str(env_dir))
        except ValueError:
            pass

    for agent in (bird, mlp):
        for attr in ("encoder", "actor", "critic"):
            if hasattr(agent, attr):
                getattr(agent, attr).eval()
    return bird, mlp


def _make_histories(env, bird, steps: list[int], seed: int):
    obs, _ = env.reset(seed=seed)
    bird.reset_history()

    obs_hist = bird._obs_hist.copy()
    act_hist = bird._act_hist.copy()
    saved = []
    max_step = max(steps)

    for t in range(max_step + 1):
        obs_hist = np.roll(obs_hist, -1, axis=1)
        if obs.ndim == 1:
            obs_hist[:, -1] = obs
        else:
            obs_hist[:, -1, :] = obs

        if t in steps:
            saved.append((t, obs_hist.copy(), act_hist.copy()))

        action, _ = bird.select_action(obs_hist, act_hist, explore=False)
        obs, _, done, _ = env.step(action)

        act_hist = np.roll(act_hist, -1, axis=1)
        act_hist[:, -1] = action.astype(np.float32)
        if done:
            break

    return saved


@torch.no_grad()
def _encode(agent, obs_hist: np.ndarray, act_hist: np.ndarray):
    oh = torch.from_numpy(obs_hist[None]).float().to(agent.device)
    ah = torch.from_numpy(act_hist[None]).float().to(agent.device)
    return agent._encode(oh, ah)


@torch.no_grad()
def _sample_bird_scores(agent, z, n_samples: int) -> np.ndarray:
    samples = agent.actor.sample(z, num_samples=n_samples)
    if samples.ndim == 2:
        samples = samples[:, None, :]
    return samples[0].detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def _mlp_scores(agent, z) -> np.ndarray:
    return agent.actor(z)[0].detach().cpu().numpy().astype(np.float32)


def _rank_positions(scores: np.ndarray) -> np.ndarray:
    # Rank 1 is highest score.
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order)
    row = np.arange(scores.shape[0])[:, None]
    ranks[row, order] = np.arange(1, scores.shape[1] + 1)
    return ranks


def _topk_activation(scores: np.ndarray, k: int) -> np.ndarray:
    idx = np.argsort(-scores, axis=1)[:, :k]
    active = np.zeros_like(scores, dtype=np.float32)
    active[np.arange(scores.shape[0])[:, None], idx] = 1.0
    return active


def _select_arms(activation_probs: np.ndarray, max_arms: int) -> np.ndarray:
    mean_p = activation_probs.mean(axis=0)
    uncertain = np.abs(mean_p - 0.5)
    return np.argsort(uncertain)[:max_arms]


def _plot(path: Path, env_name: str, steps, selected_arms, activation_probs,
          rank_samples, mlp_ranks, n_arms: int) -> None:
    fig = plt.figure(figsize=(10.5, 5.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15])

    ax0 = fig.add_subplot(gs[0, :])
    im = ax0.imshow(
        activation_probs[:, selected_arms],
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    ax0.set_title(f"{env_name}: BIRD empirical Top-K activation probability")
    ax0.set_ylabel("Fixed rollout step")
    ax0.set_yticks(np.arange(len(steps)))
    ax0.set_yticklabels([str(s) for s in steps])
    ax0.set_xticks(np.arange(len(selected_arms)))
    ax0.set_xticklabels([str(int(a)) for a in selected_arms])
    ax0.set_xlabel("Arm index")
    cbar = fig.colorbar(im, ax=ax0, fraction=0.025, pad=0.015)
    cbar.set_label("Pr(active)")

    first_state_ranks = rank_samples[0]
    for j, arm in enumerate(selected_arms[:3]):
        ax = fig.add_subplot(gs[1, j])
        bins = np.arange(0.5, n_arms + 1.5, 1.0)
        ax.hist(first_state_ranks[:, arm], bins=bins, color="#2ca02c", alpha=0.8)
        ax.axvline(mlp_ranks[0, arm], color="black", linestyle="--", linewidth=1.5,
                   label="MLP rank")
        ax.set_title(f"Arm {int(arm)} rank samples")
        ax.set_xlabel("Rank (1 = highest)")
        if j == 0:
            ax.set_ylabel("Count")
        ax.set_xlim(0.5, n_arms + 0.5)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run(args) -> None:
    spec = ENV_SPECS[args.env]
    ROOT = Path(__file__).resolve().parent.parent
    env_dir = root / spec["dir"]

    env_mod = _import_from_env(env_dir, "env")
    cfg_cls = getattr(env_mod, spec["cfg"])
    env_cls = getattr(env_mod, spec["env"])
    env_cfg = cfg_cls(N=args.N, K=args.K, T=args.T)
    env = env_cls(env_cfg, seed=args.seed)

    bird_ckpt = env_dir / args.bird_ckpt
    mlp_ckpt = env_dir / args.mlp_ckpt
    bird, mlp = _load_agents(spec, env_cfg, env_dir, bird_ckpt, mlp_ckpt)

    steps = [int(x) for x in args.steps.split(",") if x.strip()]
    histories = _make_histories(env, bird, steps, seed=spec["seed_offset"] + args.seed)
    if not histories:
        raise RuntimeError("No histories were collected; check --steps and --T.")

    all_scores = []
    all_ranks = []
    all_active_probs = []
    all_mlp_scores = []
    all_mlp_ranks = []
    actual_steps = []

    for step, obs_hist, act_hist in histories:
        z_bird = _encode(bird, obs_hist, act_hist)
        z_mlp = _encode(mlp, obs_hist, act_hist)
        bird_scores = _sample_bird_scores(bird, z_bird, args.samples)
        mlp_score = _mlp_scores(mlp, z_mlp)

        ranks = _rank_positions(bird_scores)
        active = _topk_activation(bird_scores, args.K)
        mlp_rank = _rank_positions(mlp_score[None])[0]

        actual_steps.append(step)
        all_scores.append(bird_scores)
        all_ranks.append(ranks)
        all_active_probs.append(active.mean(axis=0))
        all_mlp_scores.append(mlp_score)
        all_mlp_ranks.append(mlp_rank)

    score_arr = np.stack(all_scores, axis=0)
    rank_arr = np.stack(all_ranks, axis=0)
    active_probs = np.stack(all_active_probs, axis=0)
    mlp_score_arr = np.stack(all_mlp_scores, axis=0)
    mlp_rank_arr = np.stack(all_mlp_ranks, axis=0)
    selected_arms = _select_arms(active_probs, args.max_arms)

    out_prefix = Path(args.out_prefix or f"{args.env}_score_multimodality")
    out_png = out_prefix.with_suffix(".png")
    out_npz = out_prefix.with_suffix(".npz")

    _plot(out_png, args.env, actual_steps, selected_arms, active_probs,
          rank_arr, mlp_rank_arr, args.N)
    np.savez(
        out_npz,
        env=np.asarray(args.env),
        steps=np.asarray(actual_steps, dtype=np.int32),
        selected_arms=selected_arms.astype(np.int32),
        bird_scores=score_arr,
        bird_ranks=rank_arr,
        bird_activation_probs=active_probs,
        mlp_scores=mlp_score_arr,
        mlp_ranks=mlp_rank_arr,
        N=np.asarray(args.N, dtype=np.int32),
        K=np.asarray(args.K, dtype=np.int32),
        samples=np.asarray(args.samples, dtype=np.int32),
        seed=np.asarray(args.seed, dtype=np.int32),
    )
    print(f"[saved] {out_png}")
    print(f"[saved] {out_npz}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=sorted(ENV_SPECS), required=True)
    parser.add_argument("--N", type=int, default=20)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--steps", type=str, default="20,50,80")
    parser.add_argument("--max_arms", type=int, default=6)
    parser.add_argument("--bird_ckpt", type=str, default=None)
    parser.add_argument("--mlp_ckpt", type=str, default=None)
    parser.add_argument("--out_prefix", type=str, default=None)
    args = parser.parse_args()

    spec = ENV_SPECS[args.env]
    args.bird_ckpt = args.bird_ckpt or spec["bird_ckpt"]
    args.mlp_ckpt = args.mlp_ckpt or spec["mlp_ckpt"]
    run(args)


if __name__ == "__main__":
    main()
