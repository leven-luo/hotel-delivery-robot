"""Classical baselines. Each controller maps a TrayEnv observation to an action in [-1, 1].

They all need the environment in absolute mode (config.residual = False), because
they output the full servo command.
"""

from __future__ import annotations

import numpy as np

from sim.routes import MOTOR_TAU


class _Base:
    name = "base"

    def __init__(self, env):
        assert not env.config.residual, "baselines output absolute servo commands; use residual=False"
        self.idx = env.obs_index
        self.max_tilt = env.max_tilt
        self.dt = env.config.dt

    def reset(self) -> None:
        pass

    def _get(self, obs: np.ndarray, name: str) -> np.ndarray:
        return np.asarray(obs[self.idx[name]], dtype=float)

    def _to_action(self, servo_command: np.ndarray) -> np.ndarray:
        return np.clip(servo_command / self.max_tilt, -1.0, 1.0).astype(np.float32)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.act(obs)


class FixedTray(_Base):
    """The tray never moves: what a normal delivery robot does."""

    name = "fixed"

    def act(self, obs):
        return np.zeros(2, dtype=np.float32)


class LevelPID(_Base):
    """Keep the tray level using the tray IMU (the intuitive, but physically wrong, target)."""

    name = "level_pid"

    def __init__(self, env, kp: float = 0.8, ki: float = 8.0, kd: float = 0.0):
        super().__init__(env)
        self.kp, self.ki, self.kd = kp, ki, kd
        self.reset()

    def reset(self):
        self.integral = np.zeros(2)
        self.prev_error = None

    def act(self, obs):
        error = self._get(obs, "tray_angle")  # target is 0 (level)
        self.integral += error * self.dt
        deriv = np.zeros(2) if self.prev_error is None else (error - self.prev_error) / self.dt
        self.prev_error = error
        return self._to_action(-(self.kp * error + self.ki * self.integral + self.kd * deriv))


class GravityFeedforward(_Base):
    """Tilt the tray to atan(a / g), compensating chassis tilt.

    lead_steps > 0 aims at the acceleration predicted lead_steps ahead from the planner's
    previewed commands, to compensate servo lag (needs preview in obs).
    """

    def __init__(self, env, lead_steps: int = 0, smoothing: float = 0.3):
        super().__init__(env)
        if lead_steps:
            assert "preview" in self.idx, "lead_steps needs preview_steps > 0 in the env config"
            preview = self.idx["preview"]
            assert lead_steps <= (preview.stop - preview.start) // 2, "lead_steps exceeds preview length"
        self.lead_steps = lead_steps
        self.smoothing = smoothing
        self.name = "feedforward" if lead_steps == 0 else f"feedforward_lead{lead_steps}"
        self.reset()

    def reset(self):
        self.acc_filtered = None

    def act(self, obs):
        acc = self._get(obs, "acc")  # units of g
        if self.acc_filtered is None:
            self.acc_filtered = acc[:2].copy()
        self.acc_filtered += self.smoothing * (acc[:2] - self.acc_filtered)
        a_xy = self.acc_filtered
        if self.lead_steps:
            # Predict the actual acceleration lead_steps ahead by passing the previewed
            # commands through the drive's first-order lag (a model-based prediction).
            preview = self._get(obs, "preview").reshape(-1, 2)
            alpha = self.dt / MOTOR_TAU
            a_xy = a_xy.copy()
            for cmd in preview[: self.lead_steps]:
                a_xy += alpha * (cmd - a_xy)
        target_abs = np.arctan2(a_xy, 1.0 + acc[2])  # = atan2(a, g + a_z) with accelerations in units of g
        return self._to_action(target_abs - self._get(obs, "chassis_tilt"))


def all_baselines(env) -> list[_Base]:
    return [FixedTray(env), LevelPID(env), GravityFeedforward(env), GravityFeedforward(env, lead_steps=12)]
