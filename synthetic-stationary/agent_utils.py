"""
agent_utils.py  (synthetic-stationary)

Shared agent loaders and evaluation helper used by the stationary
comparison and actor-ablation scripts.
"""
from __future__ import annotations

import numpy as np
import torch

from env import AdaptRMABConfig, AdaptRMABEnv


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_neurwin(ckpt_path: str, env_cfg: AdaptRMABConfig) -> object:
    from neurwin import Agent, TrainConfig
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    saved = ckpt.get("cfg", {})
    cfg   = TrainConfig()
    cfg.N = env_cfg.N; cfg.K = env_cfg.K; cfg.T = env_cfg.T
    for key in ("z_dim", "encoder_hidden", "encoder_heads", "encoder_layers",
                "L", "index_hidden", "sigmoid_scale"):
        if key in saved:
            setattr(cfg, key, saved[key])
    agent = Agent(N=cfg.N, K=cfg.K, cfg=cfg)
    agent.load_checkpoint(ckpt_path)
    return agent


def load_dpmd(ckpt_path: str, env_cfg: AdaptRMABConfig) -> object:
    from diffusion_DPMD_train import DPMDAgent, DPMDTrainConfig
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    saved = ckpt.get("cfg", {})
    cfg   = DPMDTrainConfig()
    cfg.N = env_cfg.N; cfg.K = env_cfg.K; cfg.T = env_cfg.T
    for key in ("z_dim", "encoder_hidden", "encoder_heads", "encoder_layers", "L",
                "actor_hidden", "actor_t_dim", "T_diff", "score_clip",
                "critic_hidden", "action_candidates"):
        if key in saved:
            setattr(cfg, key, saved[key])
    agent = DPMDAgent(N=cfg.N, K=cfg.K, cfg=cfg)
    agent.load_checkpoint(ckpt_path)
    return agent


def load_ppo(ckpt_path: str, env_cfg: AdaptRMABConfig) -> object:
    from ppo import PPOAgent, PPOConfig
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    saved = ckpt.get("cfg", {})
    cfg   = PPOConfig()
    cfg.N = env_cfg.N; cfg.K = env_cfg.K; cfg.T = env_cfg.T
    for key in ("L", "arm_enc_hidden", "z_dim"):
        if key in saved:
            setattr(cfg, key, saved[key])
    agent = PPOAgent(N=cfg.N, K=cfg.K, cfg=cfg)
    agent.load_checkpoint(ckpt_path)
    return agent


def load_mlp_actor(ckpt_path: str, env_cfg: AdaptRMABConfig) -> object:
    from mlp_actor import MLPActorAgent, MLPActorConfig
    ckpt  = torch.load(ckpt_path, map_location="cpu")
    saved = ckpt.get("cfg", {})
    cfg   = MLPActorConfig()
    cfg.N = env_cfg.N; cfg.K = env_cfg.K; cfg.T = env_cfg.T
    for key in ("z_dim", "encoder_hidden", "encoder_heads", "encoder_layers",
                "L", "actor_hidden", "critic_hidden"):
        if key in saved:
            setattr(cfg, key, saved[key])
    agent = MLPActorAgent(N=cfg.N, K=cfg.K, cfg=cfg)
    agent.load_checkpoint(ckpt_path)
    return agent


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def eval_agent(agent, env_cfg: AdaptRMABConfig,
               n_episodes: int, seed: int) -> np.ndarray:
    """Roll out agent for n_episodes; return per-episode returns."""
    T = env_cfg.T
    returns = []
    for ep in range(n_episodes):
        env = AdaptRMABEnv(env_cfg, seed=seed + ep)
        obs, _ = env.reset()
        if hasattr(agent, "reset_history"):
            agent.reset_history()
        ep_return = 0.0
        for _ in range(T):
            action = agent.act_hard(obs)
            obs, reward_vec, done, _ = env.step(action)
            ep_return += float(reward_vec.sum())
            if done:
                break
        returns.append(ep_return)
    return np.array(returns, dtype=np.float32)
