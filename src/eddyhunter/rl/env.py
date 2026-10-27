"""PPO navigation environment: an agent chases a moving, forecasted eddy.

Two drive modes:
 - "procedural" (default, used for training): each episode samples a random
   eddy start position, a roughly-constant drift velocity (small random-walk
   perturbation added each step, like a real eddy wobbling around its mean
   propagation), and fixed per-episode persistence/confidence values drawn
   from realistic ranges. Cheap and diverse -> good for PPO training.
 - "replay": step through a literal sequence of Track-B records (e.g. from
   the real TDA detector+tracker+forecaster running on synthetic or real
   SWOT/OSCAR data), so a trained policy can be evaluated against the
   project's actual upstream output, not just the training distribution.

Observation (11-dim, all positions in km in a local tangent-plane frame
centered on the arena):
    [agent_x, agent_y, eddy_x, eddy_y, eddy_pred_x, eddy_pred_y,
     eddy_vel_u_kmstep, eddy_vel_v_kmstep, persistence_norm, confidence,
     dist_to_predicted_km]

Actions (Discrete(5)): stay, north, south, east, west -- each a fixed-size
step in km.

Reward: shaped by reduction in distance to the eddy's *predicted* position
(the thing the agent is actually supposed to be chasing), a small per-step
cost (discourage dithering), a boundary-violation penalty, and a success
bonus for reaching the predicted eddy within a tolerance radius.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import gymnasium as gym
from gymnasium import spaces


ACTION_DELTAS = {
    0: (0.0, 0.0),    # stay
    1: (0.0, 1.0),    # north
    2: (0.0, -1.0),   # south
    3: (1.0, 0.0),    # east
    4: (-1.0, 0.0),   # west
}
ACTION_NAMES = ["stay", "north", "south", "east", "west"]


@dataclass
class ReplayStep:
    eddy_x_km: float
    eddy_y_km: float
    pred_x_km: float
    pred_y_km: float
    vel_u_kmstep: float
    vel_v_kmstep: float
    persistence_norm: float
    confidence: float


@dataclass
class EddyChaseConfig:
    # One "step" = one day. Once a real 93-day daily OSCAR series was built
    # (outputs/real_scene_long.nc, 211 real tracks, 123 of them multi-point,
    # longest 17 consecutive days), real tracked-eddy displacement
    # (outputs/real_long_tracks.csv, 321 real day-to-day segments) gives a
    # much better-sampled distribution than the original 8-segment estimate:
    # median 14 km/day, mean 33.6, p90 112, max 149. step_km below matches a
    # realistic ocean glider/AUV's daily range, sized to the *mean* eddy
    # speed so pursuit is winnable on the typical case while still being a
    # real challenge against the faster tail.
    arena_half_km: float = 250.0
    # Default (None): agent starts uniformly anywhere in the arena -- fine
    # for PPO training, where the policy needs to generalize across every
    # possible relative position. For a *demo* this is actively misleading:
    # the agent can spawn 350 km from the eddy with no steps left to close
    # that gap, which has nothing to do with whether the guidance policy is
    # any good and makes identical runs look wildly inconsistent. Set this
    # (app.py's live "Run episode" does) to bound the agent's start to a
    # realistic patrol radius around the eddy's own position instead.
    agent_start_radius_km: Optional[float] = None
    step_km: float = 35.0
    max_steps: int = 60
    success_radius_km: float = 15.0
    step_penalty: float = 0.02
    boundary_penalty: float = 1.0
    success_bonus: float = 20.0
    timeout_penalty: float = 2.0
    distance_reward_scale: float = 1.0
    # procedural-mode sampling ranges
    eddy_speed_kmstep_range: tuple = (0.0, 35.0)
    eddy_wobble_kmstep: float = 15.0
    persistence_norm_range: tuple = (0.1, 1.0)
    confidence_range: tuple = (0.2, 1.0)
    strong_persistence_m: float = 30_000.0


class EddyChaseEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: EddyChaseConfig = EddyChaseConfig(),
                 mode: str = "procedural", replay_sequence: Optional[Sequence[ReplayStep]] = None,
                 seed: Optional[int] = None):
        super().__init__()
        self.cfg = config
        self.mode = mode
        self.replay_sequence = list(replay_sequence) if replay_sequence else None
        self._rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)

        self.agent_xy = np.zeros(2)
        self.eddy_xy = np.zeros(2)
        self.eddy_vel = np.zeros(2)
        self.eddy_pred_xy: Optional[np.ndarray] = None  # explicit forecast, replay mode only
        self.persistence_norm = 0.5
        self.confidence = 0.5
        self.t = 0
        self._travel_km = 0.0

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)

    def _predicted_xy(self):
        # Replay mode carries the TDA/forecast module's own prediction
        # (which may be a short-regression fit, not just constant-velocity);
        # procedural mode has no such external forecaster, so it falls back
        # to the same constant-velocity model the forecast module uses when
        # it only has two points of history.
        if self.eddy_pred_xy is not None:
            return self.eddy_pred_xy
        return self.eddy_xy + self.eddy_vel

    def _obs(self):
        pred = self._predicted_xy()
        dist = float(np.linalg.norm(self.agent_xy - pred))
        return np.array([
            self.agent_xy[0], self.agent_xy[1],
            self.eddy_xy[0], self.eddy_xy[1],
            pred[0], pred[1],
            self.eddy_vel[0], self.eddy_vel[1],
            self.persistence_norm, self.confidence,
            dist,
        ], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.t = 0
        self._travel_km = 0.0

        half = self.cfg.arena_half_km

        if self.mode == "replay" and self.replay_sequence:
            r0 = self.replay_sequence[0]
            self.eddy_xy = np.array([r0.eddy_x_km, r0.eddy_y_km])
            self.eddy_vel = np.array([r0.vel_u_kmstep, r0.vel_v_kmstep])
            self.eddy_pred_xy = np.array([r0.pred_x_km, r0.pred_y_km])
            self.persistence_norm = r0.persistence_norm
            self.confidence = r0.confidence
        else:
            self.eddy_xy = self._rng.uniform(-half * 0.6, half * 0.6, size=2)
            speed = self._rng.uniform(*self.cfg.eddy_speed_kmstep_range)
            angle = self._rng.uniform(0, 2 * np.pi)
            self.eddy_vel = speed * np.array([np.cos(angle), np.sin(angle)])
            self.eddy_pred_xy = None
            self.persistence_norm = self._rng.uniform(*self.cfg.persistence_norm_range)
            self.confidence = self._rng.uniform(*self.cfg.confidence_range)

        if self.cfg.agent_start_radius_km is not None:
            r = self.cfg.agent_start_radius_km
            offset = self._rng.uniform(-r, r, size=2)
            self.agent_xy = np.clip(self.eddy_xy + offset, -half, half)
        else:
            self.agent_xy = self._rng.uniform(-half, half, size=2)

        return self._obs(), {}

    def step(self, action: int):
        self.t += 1
        dx, dy = ACTION_DELTAS[int(action)]
        move = np.array([dx, dy]) * self.cfg.step_km
        prev_agent_xy = self.agent_xy.copy()
        new_agent_xy = self.agent_xy + move

        half = self.cfg.arena_half_km
        out_of_bounds = bool(np.any(np.abs(new_agent_xy) > half))
        self.agent_xy = np.clip(new_agent_xy, -half, half)
        self._travel_km += float(np.linalg.norm(move))

        pred_before = self._predicted_xy()
        prev_dist = float(np.linalg.norm(prev_agent_xy - pred_before))

        # advance eddy
        if self.mode == "replay" and self.replay_sequence and self.t < len(self.replay_sequence):
            r = self.replay_sequence[self.t]
            self.eddy_xy = np.array([r.eddy_x_km, r.eddy_y_km])
            self.eddy_vel = np.array([r.vel_u_kmstep, r.vel_v_kmstep])
            self.eddy_pred_xy = np.array([r.pred_x_km, r.pred_y_km])
            self.persistence_norm = r.persistence_norm
            self.confidence = r.confidence
        else:
            wobble = self._rng.normal(0, self.cfg.eddy_wobble_kmstep, size=2)
            self.eddy_vel = self.eddy_vel + wobble
            # Bound the eddy to the same monitored arena as the agent. At the
            # original 0-3 km/step procedural speeds this never mattered (a
            # 60-step random walk couldn't drift far enough to leave a
            # 150km-radius arena by chance), but at the real-world-calibrated
            # speeds this env now uses, an unbounded eddy routinely drifts
            # thousands of km past an agent confined to the arena -- an
            # impossible, not just hard, pursuit. Clamping treats "eddy left
            # the monitored region" the same way bounding the agent already
            # does: stop drifting further, don't vanish off observable space.
            self.eddy_xy = np.clip(self.eddy_xy + self.eddy_vel, -half, half)

        pred_after = self._predicted_xy()
        new_dist = float(np.linalg.norm(self.agent_xy - pred_after))

        reward = self.cfg.distance_reward_scale * (prev_dist - new_dist)
        reward -= self.cfg.step_penalty
        if out_of_bounds:
            reward -= self.cfg.boundary_penalty

        terminated = False
        truncated = False
        success = False

        if new_dist <= self.cfg.success_radius_km:
            reward += self.cfg.success_bonus
            terminated = True
            success = True
        elif self.mode == "replay" and self.replay_sequence and self.t >= len(self.replay_sequence) - 1:
            truncated = True
        elif self.t >= self.cfg.max_steps:
            reward -= self.cfg.timeout_penalty
            truncated = True

        info = {
            "success": success, "distance_km": new_dist,
            "travel_km": self._travel_km, "t": self.t,
        }
        return self._obs(), float(reward), terminated, truncated, info
