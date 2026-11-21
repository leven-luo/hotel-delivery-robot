"""Gymnasium environment for the active anti-spill tray.

Observation (only quantities the real robot can measure; no time or route index,
so the policy cannot memorize a route):
    acc           (3)    chassis IMU acceleration x, y, z, in units of g (noisy)
    chassis_tilt  (2)    chassis pitch/roll, rad (noisy)
    tray_angle    (2)    absolute tray pitch/roll from the tray IMU, rad (noisy)
    prev_action   (2)    last action
    acc_hist      (2*H)  previous horizontal accelerations, units of g
    preview       (2*P)  planner's commanded horizontal acceleration for the next P steps, units of g
    bump_dist     (1)    forward ToF: distance to next bump / sensor range, 1.0 if none in range

Action: 2 values in [-1, 1].
    absolute mode: servo command = action * max_tilt
    residual mode: servo command = gravity feedforward + action * residual_scale * max_tilt

Reward per step: 1 - (w_theta*|theta|^2 + w_theta_dot*|theta_dot|^2 + w_smooth*|delta action|^2),
capped below at -1; a spill ends the episode with -spill_penalty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .core import PhysicsParams, TraySim, ideal_tray_angle
from .dynamics import G
from .routes import fixed_route, random_route


@dataclass
class EnvConfig:
    dt: float = 0.01
    max_tilt_deg: float = 20.0
    history: int = 4
    preview_steps: int = 20  # 0.2 s; set 0 to disable (ablation)
    tof_range: float = 0.3  # m; set 0 to disable the bump sensor (ablation)
    residual: bool = False
    residual_scale: float = 0.5
    level: float = 0.0  # curriculum level: 0 = fixed route, 1 = fully random routes
    replay_fixed_prob: float = 0.2  # share of episodes on the fixed route, against forgetting
    randomize_physics: bool = True
    imu_noise: float = 0.05  # m/s^2
    angle_noise_deg: float = 0.3
    w_theta: float = 20.0
    w_theta_dot: float = 0.05
    w_smooth: float = 0.5
    spill_penalty: float = 100.0


class TrayEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None):
        super().__init__()
        self.config = config or EnvConfig()
        self.level = self.config.level  # changed by the curriculum callback
        c = self.config
        self.max_tilt = np.deg2rad(c.max_tilt_deg)

        layout = [
            ("acc", 3),
            ("chassis_tilt", 2),
            ("tray_angle", 2),
            ("prev_action", 2),
            ("acc_hist", 2 * c.history),
            ("preview", 2 * c.preview_steps),
            ("bump_dist", 1 if c.tof_range > 0 else 0),
        ]
        self.obs_index: dict[str, slice] = {}
        start = 0
        for name, size in layout:
            if size:
                self.obs_index[name] = slice(start, start + size)
                start += size
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(start,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self.sim: TraySim | None = None
        self._acc_hist = np.zeros((c.history, 2))
        self._prev_action = np.zeros(2)
        self._meas: dict[str, np.ndarray] = {}
        self._max_theta = 0.0

    # ------------------------------------------------------------------ episode
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        c, rng, options = self.config, self.np_random, options or {}

        route = options.get("route")
        if route is None:
            use_fixed = self.level <= 0.0 or rng.random() < c.replay_fixed_prob
            route = fixed_route(c.dt) if use_fixed else random_route(rng, self.level, c.dt)
        params = options.get("params")
        if params is None:
            params = PhysicsParams.sample(rng, c.max_tilt_deg) if c.randomize_physics else PhysicsParams()
        params.servo.max_angle_deg = c.max_tilt_deg

        self.sim = TraySim(route, params)
        self._acc_hist = np.zeros((c.history, 2))
        self._prev_action = np.zeros(2)
        self._max_theta = 0.0
        return self._observe(), self._info(spilled=False)

    def step(self, action):
        assert self.sim is not None, "call reset() first"
        c = self.config
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)

        theta = self.sim.step(self._servo_command(action))
        theta_dot = self.sim.slosh.theta_dot
        spilled = self.sim.slosh.spilled
        self._max_theta = max(self._max_theta, float(np.abs(theta).max()))

        if spilled:
            reward = -c.spill_penalty
        else:
            cost = (
                c.w_theta * float(np.sum(theta**2))
                + c.w_theta_dot * float(np.sum(theta_dot**2))
                + c.w_smooth * float(np.sum((action - self._prev_action) ** 2))
            )
            reward = 1.0 - min(cost, 2.0)
        self._prev_action = action

        terminated = spilled
        truncated = self.sim.done and not spilled
        return self._observe(), reward, terminated, truncated, self._info(spilled)

    # ------------------------------------------------------------------ helpers
    def feedforward_command(self) -> np.ndarray:
        """Gravity feedforward from the latest measurement: servo angle that aligns the cup with g - a."""
        acc, chassis = self._meas["acc"], self._meas["chassis_tilt"]
        return ideal_tray_angle(acc[:2], acc[2]) - chassis

    def _servo_command(self, action: np.ndarray) -> np.ndarray:
        c = self.config
        if c.residual:
            return self.feedforward_command() + action * c.residual_scale * self.max_tilt
        return action * self.max_tilt

    def _observe(self) -> np.ndarray:
        c, sim, rng = self.config, self.sim, self.np_random
        route = sim.route
        k = min(sim.k, len(route) - 1)
        angle_noise = np.deg2rad(c.angle_noise_deg)

        acc = np.array([*route.a_xy[k], route.a_z[k]]) + rng.normal(0.0, c.imu_noise, 3)
        chassis = sim.chassis_tilt(k) + rng.normal(0.0, angle_noise, 2)
        tray = sim.tray_abs(k) + rng.normal(0.0, angle_noise, 2)
        self._meas = {"acc": acc, "chassis_tilt": chassis}

        parts = {
            "acc": acc / G,
            "chassis_tilt": chassis,
            "tray_angle": tray,
            "prev_action": self._prev_action,
            "acc_hist": self._acc_hist.ravel() / G,
        }
        if c.preview_steps:
            idx = np.minimum(np.arange(k + 1, k + 1 + c.preview_steps), len(route) - 1)
            parts["preview"] = route.cmd_xy[idx].ravel() / G
        if c.tof_range > 0:
            parts["bump_dist"] = np.array([min(route.distance_to_next_bump(k), c.tof_range) / c.tof_range])

        if c.history:
            self._acc_hist = np.roll(self._acc_hist, 1, axis=0)
            self._acc_hist[0] = acc[:2]

        obs = np.concatenate([parts[name] for name in self.obs_index])
        return obs.astype(np.float32)

    def _info(self, spilled: bool) -> dict[str, Any]:
        sim = self.sim
        return {
            "theta_deg": np.degrees(sim.slosh.theta).copy(),
            "max_theta_deg": float(np.degrees(self._max_theta)),
            "spilled": bool(spilled),
            "level": float(self.level),
        }
