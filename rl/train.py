"""Train SAC or PPO on TrayEnv, optionally with an automatic curriculum.

The curriculum starts on the fixed benchmark route (level 0) and raises the
randomization level by --level-step whenever the deterministic policy delivers
without spilling in at least --threshold of evaluation episodes.

Examples:
    python -m rl.train --algo sac --residual --curriculum --timesteps 300000
    python -m rl.train --algo sac --level 1.0 --timesteps 300000        # no curriculum (ablation)
    python -m rl.train --algo sac --residual --preview-steps 0          # no preview (ablation)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

from sim.tray_env import EnvConfig, TrayEnv

try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError as exc:  # pragma: no cover
    raise SystemExit("stable-baselines3 is not installed: run `pip install -e .` first") from exc


def success_rate(model, cfg: EnvConfig, level: float, n_episodes: int, seed: int) -> float:
    """Share of episodes delivered without spilling, at the given curriculum level."""
    env = TrayEnv(replace(cfg, level=level, replay_fixed_prob=0.0))
    delivered = 0
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        delivered += not info["spilled"]
    return delivered / n_episodes


class CurriculumCallback(BaseCallback):
    def __init__(self, envs: list[TrayEnv], cfg: EnvConfig, check_freq: int, n_episodes: int,
                 threshold: float, level_step: float, seed: int, verbose: int = 1):
        super().__init__(verbose)
        self.envs, self.cfg = envs, cfg
        self.check_freq, self.n_episodes = check_freq, n_episodes
        self.threshold, self.level_step, self.seed = threshold, level_step, seed

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq:
            return True
        level = self.envs[0].level
        rate = success_rate(self.model, self.cfg, level, self.n_episodes, self.seed + self.n_calls)
        self.logger.record("curriculum/level", level)
        self.logger.record("curriculum/success_rate", rate)
        if rate >= self.threshold and level < 1.0:
            new_level = round(min(1.0, level + self.level_step), 3)
            for env in self.envs:  # set on the raw envs: wrappers do not forward attribute writes
                env.level = new_level
            if self.verbose:
                print(f"[curriculum] step {self.num_timesteps}: success {rate:.0%} at level {level} -> {new_level}")
        elif self.verbose:
            print(f"[curriculum] step {self.num_timesteps}: success {rate:.0%} at level {level}")
        return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=["sac", "ppo"], default="sac")
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--residual", action="store_true", help="learn a correction on top of gravity feedforward")
    p.add_argument("--curriculum", action="store_true", help="start at the fixed route and raise the level automatically")
    p.add_argument("--level", type=float, default=0.0, help="starting (or fixed) randomization level in [0, 1]")
    p.add_argument("--level-step", type=float, default=0.1)
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--check-freq", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--preview-steps", type=int, default=20)
    p.add_argument("--history", type=int, default=4)
    p.add_argument("--no-tof", action="store_true", help="disable the forward bump sensor")
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs")
    p.add_argument("--name", default=None)
    args = p.parse_args()

    cfg = EnvConfig(
        residual=args.residual,
        level=args.level,
        preview_steps=args.preview_steps,
        history=args.history,
        tof_range=0.0 if args.no_tof else EnvConfig.tof_range,
    )
    raw_envs = [TrayEnv(cfg) for _ in range(args.n_envs)]
    vec_env = DummyVecEnv([lambda e=e: Monitor(e) for e in raw_envs])

    algo = SAC if args.algo == "sac" else PPO
    # Small network so the policy fits and runs fast on the ESP32.
    model = algo("MlpPolicy", vec_env, seed=args.seed, verbose=0, policy_kwargs={"net_arch": [64, 64]})

    callback = None
    if args.curriculum:
        callback = CurriculumCallback(raw_envs, cfg, args.check_freq, args.eval_episodes,
                                      args.threshold, args.level_step, seed=args.seed * 1000)

    name = args.name or f"{args.algo}{'_residual' if args.residual else ''}{'_curriculum' if args.curriculum else ''}_{time.strftime('%m%d_%H%M')}"
    run_dir = Path(args.out) / name
    run_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=False)
    model.save(run_dir / "model.zip")

    final_level = raw_envs[0].level
    meta = {
        "algo": args.algo,
        "env": asdict(replace(cfg, level=final_level)),
        "timesteps": args.timesteps,
        "seed": args.seed,
        "final_level": final_level,
        "minutes": round((time.time() - start) / 60, 1),
        "final_success_rate": success_rate(model, cfg, final_level, args.eval_episodes, seed=99_000),
    }
    (run_dir / "config.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {run_dir}  (final level {final_level}, success {meta['final_success_rate']:.0%})")


if __name__ == "__main__":
    main()
