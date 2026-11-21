"""Physical models: liquid slosh and tray servos.

Both models are deliberately simple so that RL training is fast. Their
parameters are exactly what system identification on the real robot should
measure (see experiments/README.md), and what domain randomization varies.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

G = 9.81  # m/s^2


@dataclass
class SloshParams:
    freq_hz: float = 2.0  # first slosh mode natural frequency; depends on cup and fill level
    damping: float = 0.05  # damping ratio of that mode
    spill_angle_deg: float = 12.0  # surface tilt (relative to cup) at which water reaches the rim

    @property
    def omega(self) -> float:
        return 2.0 * np.pi * self.freq_hz

    @property
    def spill_angle(self) -> float:
        return np.deg2rad(self.spill_angle_deg)


class SloshModel:
    """Equivalent-pendulum model of the first slosh mode, one per horizontal axis.

    ``theta`` is the liquid surface tilt relative to the cup, for (pitch, roll).
    The surface relaxes toward the plane perpendicular to effective gravity:

        theta_eq = atan2(a_xy, g + a_z) - tray_abs
        theta''  = -2 zeta w theta' - w^2 (theta - theta_eq)

    Simplifications: tray angular acceleration and pitch/roll coupling are ignored.
    """

    def __init__(self, params: SloshParams):
        self.params = params
        self.theta = np.zeros(2)
        self.theta_dot = np.zeros(2)

    def reset(self) -> None:
        self.theta[:] = 0.0
        self.theta_dot[:] = 0.0

    @staticmethod
    def equilibrium(a_xy: np.ndarray, a_z: float, tray_abs: np.ndarray) -> np.ndarray:
        return np.arctan2(a_xy, G + a_z) - tray_abs

    def step(self, a_xy: np.ndarray, a_z: float, tray_abs: np.ndarray, dt: float) -> np.ndarray:
        w, zeta = self.params.omega, self.params.damping
        theta_eq = self.equilibrium(a_xy, a_z, tray_abs)
        theta_ddot = -2.0 * zeta * w * self.theta_dot - w**2 * (self.theta - theta_eq)
        # Semi-implicit Euler: stable for w * dt well below 1 (w ~ 20 rad/s, dt = 0.01 s).
        self.theta_dot = self.theta_dot + theta_ddot * dt
        self.theta = self.theta + self.theta_dot * dt
        return self.theta

    @property
    def spilled(self) -> bool:
        return bool(np.any(np.abs(self.theta) > self.params.spill_angle))


@dataclass
class ServoParams:
    time_constant: float = 0.05  # s, first-order lag toward the (delayed) command
    max_rate_deg_s: float = 300.0  # slew-rate limit
    delay_steps: int = 2  # pure delay in control steps (communication + PWM update)
    max_angle_deg: float = 20.0  # mechanical limit of the gimbal

    @property
    def max_rate(self) -> float:
        return np.deg2rad(self.max_rate_deg_s)

    @property
    def max_angle(self) -> float:
        return np.deg2rad(self.max_angle_deg)


class ServoModel:
    """Two hobby servos (pitch, roll): pure delay + first-order lag + rate and angle limits.

    The angle is relative to the chassis; the tray's absolute tilt is chassis tilt + angle.
    """

    def __init__(self, params: ServoParams, dt: float):
        self.params = params
        self.dt = dt
        self.angle = np.zeros(2)
        self._buffer: deque[np.ndarray] = deque()
        self.reset()

    def reset(self) -> None:
        self.angle = np.zeros(2)
        self._buffer = deque(np.zeros(2) for _ in range(self.params.delay_steps))

    def step(self, command: np.ndarray) -> np.ndarray:
        p = self.params
        command = np.clip(np.asarray(command, dtype=float), -p.max_angle, p.max_angle)
        self._buffer.append(command)
        delayed = self._buffer.popleft()
        rate = np.clip((delayed - self.angle) / p.time_constant, -p.max_rate, p.max_rate)
        self.angle = np.clip(self.angle + rate * self.dt, -p.max_angle, p.max_angle)
        return self.angle
