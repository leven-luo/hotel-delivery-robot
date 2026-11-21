"""Physics and route sanity checks. Needs numpy only."""

import unittest

import numpy as np

from sim import G, PhysicsParams, SloshParams, TraySim, fixed_route, ideal_tray_angle, random_route
from sim.dynamics import ServoModel, ServoParams, SloshModel


class SloshTest(unittest.TestCase):
    def run_slosh(self, tray_abs, accel=2.0, seconds=6.0, dt=0.01):
        model = SloshModel(SloshParams(freq_hz=2.0, damping=0.3))
        for _ in range(int(seconds / dt)):
            model.step(np.array([accel, 0.0]), 0.0, tray_abs, dt)
        return model.theta

    def test_fixed_tray_settles_at_effective_gravity_angle(self):
        theta = self.run_slosh(np.zeros(2))
        self.assertAlmostEqual(theta[0], np.arctan2(2.0, G), places=3)
        self.assertAlmostEqual(theta[1], 0.0, places=6)

    def test_aligned_tray_keeps_surface_level(self):
        theta = self.run_slosh(ideal_tray_angle(np.array([2.0, 0.0])))
        np.testing.assert_allclose(theta, 0.0, atol=1e-9)


class ServoTest(unittest.TestCase):
    def test_delay_lag_and_rate_limit(self):
        params = ServoParams(time_constant=0.05, max_rate_deg_s=300, delay_steps=2)
        servo = ServoModel(params, dt=0.01)
        target = np.deg2rad([10.0, -10.0])
        angles = [servo.step(target).copy() for _ in range(60)]
        np.testing.assert_allclose(angles[1], 0.0)  # still inside the 2-step delay
        steps = np.abs(np.diff(np.array(angles), axis=0))
        self.assertTrue(np.all(steps <= params.max_rate * 0.01 + 1e-12))
        np.testing.assert_allclose(angles[-1], target, atol=np.deg2rad(0.1))


class RouteTest(unittest.TestCase):
    def test_fixed_route_is_deterministic_and_physical(self):
        a, b = fixed_route(), fixed_route()
        np.testing.assert_array_equal(a.a_xy, b.a_xy)
        self.assertGreaterEqual(a.v.min(), 0.0)
        self.assertLessEqual(a.v.max(), 1.2 + 1e-9)
        self.assertLess(a.v[-1], 0.05)  # ends stopped
        self.assertEqual(a.cmd_xy.shape, a.a_xy.shape)
        self.assertGreater(np.abs(a.a_z).max(), 0.1)  # the bump shakes the chassis

    def test_random_routes_depend_on_seed(self):
        r1 = random_route(np.random.default_rng(1), 1.0)
        r1b = random_route(np.random.default_rng(1), 1.0)
        r2 = random_route(np.random.default_rng(2), 1.0)
        np.testing.assert_array_equal(r1.a_xy, r1b.a_xy)
        self.assertFalse(len(r1) == len(r2) and np.allclose(r1.a_xy, r2.a_xy))

    def test_level_zero_is_the_fixed_route(self):
        np.testing.assert_array_equal(random_route(np.random.default_rng(3), 0.0).a_xy, fixed_route().a_xy)


class TraySimTest(unittest.TestCase):
    def max_theta(self, policy):
        sim = TraySim(fixed_route(), PhysicsParams())
        worst = 0.0
        while not sim.done:
            worst = max(worst, float(np.abs(sim.step(policy(sim))).max()))
        return np.degrees(worst)

    def test_feedforward_beats_fixed_tray(self):
        fixed = self.max_theta(lambda sim: np.zeros(2))

        def feedforward(sim):
            k = sim._index(None)
            return ideal_tray_angle(sim.route.a_xy[k], sim.route.a_z[k]) - sim.chassis_tilt(k)

        ff = self.max_theta(feedforward)
        self.assertGreater(fixed, 12.0)  # a fixed tray spills on the benchmark route
        self.assertLess(ff, fixed / 2)


if __name__ == "__main__":
    unittest.main()
