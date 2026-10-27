"""Train PPO on the procedural EddyChaseEnv, and evaluate PPO vs random vs
direct-navigation baselines on both procedural episodes and replayed
real(-ish) Track-B sequences from the TDA detector+tracker+forecaster.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from .env import EddyChaseEnv, EddyChaseConfig, ReplayStep
from .baselines import random_policy, direct_navigation_policy, pursuit_guidance_policy


def make_env(config: EddyChaseConfig, seed: int):
    def _thunk():
        env = EddyChaseEnv(config=config, mode="procedural", seed=seed)
        return env
    return _thunk


def train_ppo(config: EddyChaseConfig = EddyChaseConfig(), total_timesteps: int = 200_000,
              n_envs: int = 8, seed: int = 0, verbose: int = 1) -> PPO:
    vec_env = DummyVecEnv([make_env(config, seed + i) for i in range(n_envs)])
    model = PPO(
        "MlpPolicy", vec_env,
        n_steps=256, batch_size=256, n_epochs=10,
        learning_rate=3e-4, gamma=0.995, gae_lambda=0.95,
        ent_coef=0.01, verbose=verbose, seed=seed,
        policy_kwargs=dict(net_arch=[64, 64]),
    )
    model.learn(total_timesteps=total_timesteps)
    return model


def _run_episode(env: EddyChaseEnv, policy_fn, rng: Optional[np.random.Generator] = None) -> dict:
    obs, info = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        if policy_fn is random_policy:
            action = policy_fn(obs, rng)
        elif policy_fn in (direct_navigation_policy, pursuit_guidance_policy):
            action = policy_fn(obs, step_km=env.cfg.step_km)
        else:
            action, _ = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
    return {
        "success": info["success"], "steps": info["t"],
        "travel_km": info["travel_km"], "final_distance_km": info["distance_km"],
        "reward": total_reward,
    }


def evaluate_policy_on_env_factory(policy_fn, env_factory, n_episodes: int = 100, seed: int = 123) -> dict:
    rng = np.random.default_rng(seed)
    results = []
    for i in range(n_episodes):
        env = env_factory()
        res = _run_episode(env, policy_fn, rng=rng)
        results.append(res)

    arr = lambda k: np.array([r[k] for r in results])
    return {
        "n_episodes": n_episodes,
        "success_rate": float(arr("success").mean()),
        "mean_steps": float(arr("steps").mean()),
        "mean_travel_km": float(arr("travel_km").mean()),
        "mean_final_distance_km": float(arr("final_distance_km").mean()),
        "mean_reward": float(arr("reward").mean()),
    }


def evaluate_all(model: PPO, config: EddyChaseConfig, n_episodes: int = 100, seed: int = 123,
                  replay_sequences: Optional[List[List[ReplayStep]]] = None) -> dict:
    out = {}

    def ppo_policy(obs):
        return model.predict(obs, deterministic=True)

    procedural_factory = lambda: EddyChaseEnv(config=config, mode="procedural", seed=None)

    out["procedural"] = {
        "ppo": evaluate_policy_on_env_factory(ppo_policy, procedural_factory, n_episodes, seed),
        "random": evaluate_policy_on_env_factory(random_policy, procedural_factory, n_episodes, seed),
        "direct": evaluate_policy_on_env_factory(direct_navigation_policy, procedural_factory, n_episodes, seed),
        "pursuit": evaluate_policy_on_env_factory(pursuit_guidance_policy, procedural_factory, n_episodes, seed),
    }

    if replay_sequences:
        def replay_factory_fn(seq):
            return lambda: EddyChaseEnv(config=config, mode="replay", replay_sequence=seq, seed=None)

        for name, policy_fn in (("ppo", ppo_policy), ("random", random_policy), ("direct", direct_navigation_policy), ("pursuit", pursuit_guidance_policy)):
            per_seq = []
            for seq in replay_sequences:
                rng = np.random.default_rng(seed)
                env = replay_factory_fn(seq)()
                per_seq.append(_run_episode(env, policy_fn, rng=rng))
            arr = lambda k: np.array([r[k] for r in per_seq])
            out.setdefault("replay", {})[name] = {
                "n_episodes": len(per_seq),
                "success_rate": float(arr("success").mean()),
                "mean_steps": float(arr("steps").mean()),
                "mean_travel_km": float(arr("travel_km").mean()),
                "mean_final_distance_km": float(arr("final_distance_km").mean()),
                "mean_reward": float(arr("reward").mean()),
            }

    return out
