"""Non-learned baseline policies PPO is compared against: random movement,
and a greedy direct-navigation controller that always steps toward the
eddy's currently predicted position.
"""

from __future__ import annotations

import numpy as np

from .env import ACTION_DELTAS


def random_policy(obs: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.integers(0, 5))


def direct_navigation_policy(obs: np.ndarray, step_km: float = 6.0) -> int:
    """Pick the single action whose resulting position is closest to the
    eddy's predicted position -- a simple, non-learned proportional
    navigator, i.e. "fly straight at where the eddy will be".
    """
    agent_xy = obs[0:2]
    pred_xy = obs[4:6]

    best_action = 0
    best_dist = float(np.linalg.norm(agent_xy - pred_xy))
    for action, (dx, dy) in ACTION_DELTAS.items():
        candidate = agent_xy + np.array([dx, dy]) * step_km
        d = float(np.linalg.norm(candidate - pred_xy))
        if d < best_dist:
            best_dist = d
            best_action = action
    return best_action
