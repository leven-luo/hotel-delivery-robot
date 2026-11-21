"""TraySim: car + chassis tilt + tray servos + liquid, with no RL dependencies.

Kept separate from the Gymnasium wrapper so baselines, tests and (later) MPC
can use the same physics, and so the firmware team has one reference model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dynamics import G, ServoModel, ServoParams, SloshModel, SloshParams
from .routes import Route


@dataclass
class PhysicsParams:
    slosh: SloshParams = field(default_factory=SloshParams)
    servo: ServoParams = field(default_factory=ServoParams)
    susp_gain: float = 0.01  # rad of chassis tilt (squat / dive / body roll) per m/s^2

    @classmethod
    def sample(cls, rng: np.random.Generator, max_tilt_deg: float = 20.0) -> "PhysicsParams":
        """Domain randomization ranges; narrow them once system identification is done."""
        return cls(
            slosh=SloshParams(
                freq_hz=rng.uniform(1.5, 3.0),
                damping=rng.uniform(0.02, 0.10),
                spill_angle_deg=rng.uniform(9.0, 15.0),
            ),
            servo=ServoParams(
                time_constant=rng.uniform(0.03, 0.08),
                max_rate_deg_s=rng.uniform(200.0, 400.0),
                delay_steps=int(rng.integers(1, 4)),
                max_angle_deg=max_tilt_deg,
            ),
            susp_gain=rng.uniform(0.005, 0.015),
        )


def ideal_tray_angle(a_xy: np.ndarray, a_z: float = 0.0) -> np.ndarray:
    """Absolute tray tilt that keeps the liquid surface level relative to the cup."""
    return np.arctan2(a_xy, G + a_z)


class TraySim:
    def __init__(self, route: Route, params: PhysicsParams | None = None):
        self.route = route
        self.params = params or PhysicsParams()
        self.dt = route.dt
        self.slosh = SloshModel(self.params.slosh)
        self.servo = ServoModel(self.params.servo, self.dt)
        self.k = 0

    @property
    def done(self) -> bool:
        return self.k >= len(self.route)

    def _index(self, k: int | None) -> int:
        return min(self.k if k is None else k, len(self.route) - 1)

    def chassis_tilt(self, k: int | None = None) -> np.ndarray:
        k = self._index(k)
        tilt = -self.params.susp_gain * self.route.a_xy[k]  # squat on acceleration, roll outward in turns
        tilt[0] += self.route.bump_pitch[k]
        return tilt

    def tray_abs(self, k: int | None = None) -> np.ndarray:
        return self.chassis_tilt(k) + self.servo.angle

    def step(self, servo_command: np.ndarray) -> np.ndarray:
        """Advance one control step with the given servo command (rad, relative to chassis)."""
        k = self._index(None)
        self.servo.step(servo_command)
        self.slosh.step(self.route.a_xy[k], self.route.a_z[k], self.tray_abs(k), self.dt)
        self.k += 1
        return self.slosh.theta.copy()
