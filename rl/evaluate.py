"""Compare controllers on the benchmark route and on held-out random routes.

Usage:
    python -m rl.evaluate                                  # classical baselines only
    python -m rl.evaluate --model runs/sac_residual_...    # add trained policies (repeatable)
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from control.baselines import all_baselines
from sim.core import PhysicsParams
from sim.routes import fixed_route, random_route
from sim.tray_env import EnvConfig, TrayEnv

HELDOUT_SEED = 10_000  # routes generated from these seeds are reserved for testing


def benchmark_set(n_heldout: int = 10, dt: float = 0.01):
    """The fixed benchmark route (nominal physics) plus held-out random routes (sampled physics)."""
    cases = [("fixed", fixed_route(dt), PhysicsParams())]
    for i in range(n_heldout):
        rng = np.random.default_rng(HELDOUT_SEED + i)
        cases.append((f"heldout_{i}", random_route(rng, 1.0, dt), PhysicsParams.sample(rng)))
    return cases


def run_episode(env: TrayEnv, policy, route, params, seed: int = 0) -> dict:
    obs, _ = env.reset(seed=seed, options={"route": route, "params": params})
    if hasattr(policy, "reset"):
        policy.reset()
    thetas = []
    while True:
        obs, _, terminated, truncated, info = env.step(policy(obs))
        thetas.append(info["theta_deg"])
        if terminated or truncated:
            break
    theta = np.abs(np.array(thetas))
    return {
        "spilled": info["spilled"],
        "max_theta_deg": info["max_theta_deg"],
        "rms_theta_deg": float(np.sqrt(np.mean(theta**2))),
        "steps": len(thetas),
    }


def load_trained(run_dir: str | Path):
    """Return (name, env, policy) for a run saved by rl.train."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "config.json").read_text())
    if meta["algo"] == "sac":
        from stable_baselines3 import SAC as Algo
    else:
        from stable_baselines3 import PPO as Algo
    model = Algo.load(run_dir / "model.zip")
    env = TrayEnv(EnvConfig(**meta["env"]))

    def policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    return run_dir.name, env, policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", default=[], help="run directory from rl.train (repeatable)")
    parser.add_argument("--n-heldout", type=int, default=10)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    base_env = TrayEnv(EnvConfig(residual=False))
    entries = [(b.name, base_env, b) for b in all_baselines(base_env)]
    entries += [load_trained(path) for path in args.model]
    cases = benchmark_set(args.n_heldout)

    rows = []
    for name, env, policy in entries:
        for case, route, params in cases:
            rows.append({"controller": name, "case": case, **run_episode(env, policy, route, params)})

    print(f"{'controller':28s} {'fixed max':>10s} {'held-out max':>13s} {'held-out rms':>13s} {'spill rate':>11s}")
    for name, *_ in entries:
        mine = [r for r in rows if r["controller"] == name]
        fixed = next(r for r in mine if r["case"] == "fixed")
        held = [r for r in mine if r["case"] != "fixed"]
        print(
            f"{name:28s} {fixed['max_theta_deg']:9.1f}° {np.mean([r['max_theta_deg'] for r in held]):12.1f}° "
            f"{np.mean([r['rms_theta_deg'] for r in held]):12.2f}° {np.mean([r['spilled'] for r in held]):10.0%}"
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-episode results: {path}")


if __name__ == "__main__":
    main()
