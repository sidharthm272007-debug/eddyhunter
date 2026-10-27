"""Non-learned guidance policies: random movement, a greedy direct-navigation
controller, and the lead-pursuit controller this project actually ships.

Why classical guidance instead of PPO for single-eddy pursuit: on real
detected tracks, episodes run a median of 3 steps before the track ends
(SWOT's sparse revisit cadence), which is too short a horizon for anything
RL's long-horizon credit assignment is good at -- it collapses to a single
aim-and-go decision. PPO trained on this task tied or lost to the simple
`direct_navigation_policy` baseline (~27% vs ~27-33% success on real
replayed tracks) precisely because there's no sequential structure here for
it to exploit. A classical guidance law solves the actual problem exactly,
with no training, no drift risk, and a fully auditable "why did it pick that
heading" answer -- which matters more for an operational tool than a
black-box policy that's tied with a one-line heuristic anyway. (RL would
earn its keep on the *harder* version of this problem -- routing a vessel
across several eddies at once under a fuel/time budget, deciding which to
chase and when to wait for a fresher forecast -- a genuinely long-horizon
decision this project doesn't yet model.)
"""

from __future__ import annotations

import numpy as np

from .env import ACTION_DELTAS


def random_policy(obs: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.integers(0, 5))


def _best_action_toward(agent_xy: np.ndarray, target_xy: np.ndarray, step_km: float) -> int:
    best_action = 0
    best_dist = float(np.linalg.norm(agent_xy - target_xy))
    for action, (dx, dy) in ACTION_DELTAS.items():
        candidate = agent_xy + np.array([dx, dy]) * step_km
        d = float(np.linalg.norm(candidate - target_xy))
        if d < best_dist:
            best_dist = d
            best_action = action
    return best_action


def direct_navigation_policy(obs: np.ndarray, step_km: float) -> int:
    """Pure pursuit: aim at the eddy's currently forecast position. `step_km`
    must match the environment's own EddyChaseConfig.step_km -- passing the
    wrong value (this used to silently default to 6.0 km regardless of the
    real config's 35 km step) makes every candidate-move distance comparison
    wrong, which handicaps this baseline for no good reason.
    """
    return _best_action_toward(obs[0:2], obs[4:6], step_km)


def pursuit_guidance_policy(obs: np.ndarray, step_km: float) -> int:
    """Lead pursuit: aim one step ahead of the current forecast, using the
    eddy's own estimated drift velocity (obs[6:8]), instead of chasing where
    it already was. This is the standard fix for pure pursuit's classic
    failure mode -- always aiming at the target's *current* spot means you
    perpetually trail a moving target instead of intercepting it. This is
    the policy actually recommended for single-eddy pursuit in this project
    -- see the module docstring for why it replaces PPO here.
    """
    agent_xy = obs[0:2]
    pred_xy = obs[4:6]
    vel_xy = obs[6:8]
    if not np.all(np.isfinite(vel_xy)):
        return _best_action_toward(agent_xy, pred_xy, step_km)
    lead_xy = pred_xy + vel_xy  # one step of eddy drift ahead of the forecast point
    return _best_action_toward(agent_xy, lead_xy, step_km)
