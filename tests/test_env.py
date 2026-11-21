"""Gymnasium environment and baseline checks. Skipped when gymnasium is not installed."""

import importlib.util
import unittest

import numpy as np

HAS_GYM = importlib.util.find_spec("gymnasium") is not None


@unittest.skipUnless(HAS_GYM, "gymnasium not installed")
class TrayEnvTest(unittest.TestCase):
    def setUp(self):
        from sim.tray_env import EnvConfig, TrayEnv

        self.EnvConfig, self.TrayEnv = EnvConfig, TrayEnv

    def test_passes_gymnasium_checker(self):
        from gymnasium.utils.env_checker import check_env

        check_env(self.TrayEnv(), skip_render_check=True)

    def test_observation_layout_matches_space(self):
        env = self.TrayEnv(self.EnvConfig(history=3, preview_steps=5, tof_range=0.0))
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertNotIn("bump_dist", env.obs_index)
        self.assertEqual(env.obs_index["preview"].stop - env.obs_index["preview"].start, 10)

    def test_residual_zero_action_equals_feedforward(self):
        env = self.TrayEnv(self.EnvConfig(residual=True, randomize_physics=False))
        env.reset(seed=0)
        np.testing.assert_allclose(env._servo_command(np.zeros(2)), env.feedforward_command())

    def test_curriculum_level_changes_routes(self):
        env = self.TrayEnv(self.EnvConfig(level=1.0, replay_fixed_prob=0.0))
        lengths = set()
        for seed in range(5):
            env.reset(seed=seed)
            lengths.add(len(env.sim.route))
        self.assertGreater(len(lengths), 1)

    def test_baselines_rank_as_expected_on_benchmark(self):
        from control.baselines import FixedTray, GravityFeedforward
        from rl.evaluate import benchmark_set, run_episode

        env = self.TrayEnv(self.EnvConfig(residual=False))
        _, route, params = benchmark_set(n_heldout=0)[0]
        fixed = run_episode(env, FixedTray(env), route, params)
        ff = run_episode(env, GravityFeedforward(env), route, params)
        self.assertTrue(fixed["spilled"])
        self.assertFalse(ff["spilled"])


if __name__ == "__main__":
    unittest.main()
