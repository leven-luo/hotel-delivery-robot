"""Driving routes: commanded and actual car motion, plus speed bumps.

A route is fully determined before an episode starts: the tray cannot influence
the car, so the whole trajectory (speed, accelerations, bump-induced vertical
motion and chassis pitch) is precomputed. ``cmd_xy`` holds the planner's
commanded acceleration, which the tray controller may read ahead of time
("preview"); the actual acceleration lags it through the drive motors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dynamics import G

WHEELBASE = 0.20  # m
V_MAX = 1.2  # m/s
MOTOR_TAU = 0.12  # s, lag between commanded and actual acceleration / yaw rate
SUSPENSION_HZ = 4.0  # vertical tire/suspension mode of the small chassis
SUSPENSION_DAMPING = 0.5


@dataclass(frozen=True)
class Segment:
    duration: float  # s
    accel: float = 0.0  # commanded longitudinal acceleration, m/s^2
    yaw_rate: float = 0.0  # commanded yaw rate, rad/s (positive = left turn)


@dataclass(frozen=True)
class Bump:
    position: float  # m along the path, where the bump starts
    height: float  # m
    length: float = 0.10  # m, a wooden strip with ramps


@dataclass
class Route:
    dt: float
    v: np.ndarray  # (N,) speed, m/s
    s: np.ndarray  # (N,) distance travelled by the car center, m
    a_xy: np.ndarray  # (N, 2) actual horizontal acceleration in the body frame
    a_z: np.ndarray  # (N,) vertical acceleration of the chassis
    bump_pitch: np.ndarray  # (N,) chassis pitch from bumps, rad (tray sign convention)
    cmd_xy: np.ndarray  # (N, 2) commanded horizontal acceleration (known to the planner)
    bumps: tuple[Bump, ...]

    def __len__(self) -> int:
        return len(self.v)

    @property
    def duration(self) -> float:
        return len(self) * self.dt

    def distance_to_next_bump(self, k: int) -> float:
        """Distance from the front axle to the start of the next bump ahead (inf if none)."""
        front = self.s[k] + WHEELBASE / 2
        ahead = [b.position - front for b in self.bumps if b.position - front >= 0.0]
        return min(ahead) if ahead else float("inf")


def _bump_height(x: np.ndarray, bumps: tuple[Bump, ...]) -> np.ndarray:
    z = np.zeros_like(x)
    for b in bumps:
        u = (x - b.position) / b.length
        inside = (u >= 0.0) & (u <= 1.0)
        z[inside] += b.height * np.sin(np.pi * u[inside])
    return z


def _suspension(z_ground: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Second-order filter from ground height to chassis height; returns (z, z_ddot)."""
    w = 2 * np.pi * SUSPENSION_HZ
    z = zd = 0.0
    zs, zdd = np.zeros_like(z_ground), np.zeros_like(z_ground)
    for k, zg in enumerate(z_ground):
        acc = w**2 * (zg - z) - 2 * SUSPENSION_DAMPING * w * zd
        zd += acc * dt
        z += zd * dt
        zs[k], zdd[k] = z, acc
    return zs, zdd


def build_route(segments: list[Segment], bumps: tuple[Bump, ...] = (), dt: float = 0.01) -> Route:
    acc_cmd = np.concatenate([np.full(max(1, round(seg.duration / dt)), seg.accel) for seg in segments])
    yaw_cmd = np.concatenate([np.full(max(1, round(seg.duration / dt)), seg.yaw_rate) for seg in segments])
    n = len(acc_cmd)

    v, s, a_long, yaw = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    alpha = dt / MOTOR_TAU
    vel = pos = a_act = yaw_act = 0.0
    for k in range(n):
        cmd = acc_cmd[k]
        if (vel >= V_MAX and cmd > 0) or (vel <= 0.0 and cmd < 0):
            cmd = 0.0  # the planner does not ask for impossible accelerations
        acc_cmd[k] = cmd
        a_act += alpha * (cmd - a_act)
        yaw_act += alpha * (yaw_cmd[k] - yaw_act)
        new_vel = float(np.clip(vel + a_act * dt, 0.0, V_MAX))
        v[k], s[k], yaw[k] = vel, pos, yaw_act
        a_long[k] = (new_vel - vel) / dt
        pos += 0.5 * (vel + new_vel) * dt
        vel = new_vel

    a_xy = np.column_stack([a_long, v * yaw])  # centripetal acceleration points left in a left turn
    cmd_xy = np.column_stack([acc_cmd, v * yaw_cmd])

    z_front = _bump_height(s + WHEELBASE / 2, bumps)
    z_rear = _bump_height(s - WHEELBASE / 2, bumps)
    _, a_z = _suspension(0.5 * (z_front + z_rear), dt)
    a_z = np.clip(a_z, -0.7 * G, 1.5 * G)
    pitch_ground = -(z_front - z_rear) / WHEELBASE  # nose up = front edge up = negative
    pitch, _ = _suspension(pitch_ground, dt)

    return Route(dt=dt, v=v, s=s, a_xy=a_xy, a_z=a_z, bump_pitch=pitch, cmd_xy=cmd_xy, bumps=tuple(bumps))


def fixed_route(dt: float = 0.01) -> Route:
    """The benchmark route: start, cruise over one bump, left turn, hard brake."""
    segments = [
        Segment(0.5),
        Segment(0.8, accel=1.5),
        Segment(1.0),
        Segment(1.5, yaw_rate=1.0),
        Segment(0.8),
        Segment(0.6, accel=-2.4),
        Segment(1.0),
    ]
    return build_route(segments, bumps=(Bump(position=1.0, height=0.01),), dt=dt)


def random_route(rng: np.random.Generator, level: float, dt: float = 0.01) -> Route:
    """Random route whose spread grows with the curriculum level (0 = fixed route, 1 = full)."""
    level = float(np.clip(level, 0.0, 1.0))
    if level == 0.0:
        return fixed_route(dt)

    amp = 0.5 + 2.0 * level  # largest accel/brake magnitude, m/s^2
    segments = [Segment(rng.uniform(0.2, 0.6)), Segment(rng.uniform(0.6, 1.2), accel=rng.uniform(0.8, 0.8 + 1.7 * level))]
    for _ in range(int(rng.integers(2, 3 + round(4 * level)))):
        kind = rng.choice(["cruise", "accel", "brake", "turn"])
        if kind == "cruise":
            segments.append(Segment(rng.uniform(0.4, 1.5)))
        elif kind == "accel":
            segments.append(Segment(rng.uniform(0.3, 1.0), accel=rng.uniform(0.3, amp)))
        elif kind == "brake":
            segments.append(Segment(rng.uniform(0.2, 0.6), accel=-rng.uniform(0.3, amp)))
        else:
            sign = rng.choice([-1.0, 1.0])
            segments.append(Segment(rng.uniform(0.6, 1.5), yaw_rate=sign * rng.uniform(0.3, 0.3 + 1.2 * level)))
    segments += [Segment(1.5, accel=-rng.uniform(1.0, 1.0 + 1.5 * level)), Segment(0.5)]

    path_length = build_route(segments, dt=dt).s[-1]
    n_bumps = int(rng.integers(0, 2 + round(2 * level)))
    bumps = tuple(
        Bump(position=float(rng.uniform(0.3, max(0.4, path_length - 0.3))), height=float(rng.uniform(0.004, 0.004 + 0.012 * level)))
        for _ in range(n_bumps)
    )
    return build_route(segments, bumps=bumps, dt=dt)
