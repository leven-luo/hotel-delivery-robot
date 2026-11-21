"""Simulation package. The numpy core imports here; the Gymnasium env lives in sim.tray_env."""

from .core import PhysicsParams, TraySim, ideal_tray_angle
from .dynamics import G, ServoParams, SloshParams
from .routes import Bump, Route, Segment, build_route, fixed_route, random_route

__all__ = [
    "G", "Bump", "PhysicsParams", "Route", "Segment", "ServoParams", "SloshParams", "TraySim",
    "build_route", "fixed_route", "ideal_tray_angle", "random_route",
]
