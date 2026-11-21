# Spill-Free Delivery Robot (OC/RL course project)

A four-wheeled delivery robot carries a cup of water on an actively tilting tray.
The car drives its route as usual; a controller decides, every 10 ms, how far to
tilt the tray (pitch and roll) so the liquid does not slosh over the rim.

We compare classical controllers (fixed tray, PID level-keeping, gravity
feedforward, MPC) against reinforcement learning (SAC, residual RL with a physics
feedforward), first in simulation and then on real hardware.

## Key idea

During acceleration the liquid surface settles perpendicular to the effective
gravity `g - a`, not to the cup. Tilting the tray to

```
pitch = atan(a_x / g),   roll = atan(a_y / g)
```

keeps the surface level relative to the cup. Servo lag, sensor noise, unknown
fill level (slosh frequency) and speed bumps make this formula insufficient,
which is where RL comes in: it can act on previewed speed commands before the
car decelerates and learns to avoid exciting slosh resonance.

## Repository layout

```
sim/            Simulation (numpy core + Gymnasium env)
  dynamics.py     Liquid slosh (equivalent pendulum) and servo models
  routes.py       Fixed benchmark route and random route generator (bumps, turns, braking)
  core.py         TraySim: steps car + chassis tilt + servo + liquid, no RL dependencies
  tray_env.py     TrayEnv (Gymnasium): observations, reward, curriculum level, residual mode
control/        Classical baselines: fixed tray, PID level-keeping, gravity feedforward (+ preview)
rl/
  train.py        Train SAC/PPO with an automatic curriculum (fixed -> random routes)
  evaluate.py     Run every controller on benchmark + held-out routes, write a CSV
firmware/       ESP32 code for the tray (to be added)
experiments/    Real-robot trial protocol and data log template
tests/          Unit tests (stdlib unittest; env tests skip if gymnasium is missing)
```

## Setup

Python 3.11 or 3.12 is recommended (PyTorch wheels lag behind the newest Python).

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

Or with plain pip: `python -m venv .venv && source .venv/bin/activate && pip install -e .`

## Quick start

```bash
python -m unittest discover -s tests -t . -v     # physics + env sanity checks
python -m rl.evaluate                            # compare baselines on benchmark routes
python -m rl.train --algo sac --residual --curriculum --timesteps 300000
python -m rl.evaluate --model runs/<run-name>    # add a trained policy to the comparison
```

## Baseline results (simulation, v0)

Benchmark route with nominal physics, plus 10 held-out random routes (level 1.0,
randomized physics). Max slosh = largest liquid surface tilt relative to the cup.

| Controller | Benchmark max slosh | Held-out mean max slosh | Held-out spill rate |
| --- | --- | --- | --- |
| Fixed tray | 12.2° (spills) | 12.0° | 100% |
| PID level-keeping | 12.0° (spills) | 11.9° | 100% |
| Gravity feedforward | 8.5° | 10.6° | 60% |
| Gravity feedforward + 0.12 s preview | 5.7° | 7.5° | 0% |

Spilled episodes stop at the spill angle, so their max is capped near it. The
preview feedforward is the strong model-based baseline RL has to beat (on slosh
magnitude, and without preview in the ablation).

## Conventions

- Body frame: x forward, y left, z up. Units: SI (m, s, rad).
- Tray angle sign: positive pitch/roll leans the tray *into* the acceleration
  (front/left edge down), so the ideal tray angle has the same sign as `a_x`/`a_y`.
- `theta` = liquid surface tilt relative to the cup; spill when `|theta|` exceeds
  the fill-dependent spill angle.

## Roadmap

| Deadline | Deliverable | Content |
| --- | --- | --- |
| Nov 19 | Preliminary report + this repo | Simulation v0, three baselines, first SAC learning curve |
| Nov 26 | Internal checkpoint | Tray hardware tracks commanded angles; system identification fed back into sim |
| Dec 3 | Pre-report | Residual RL, curriculum to random routes, first real-robot runs |
| Dec 10 | Presentation | Full comparison on 5 benchmark + 2 held-out layouts, demo video |
| Dec 17 | Final report | Analysis, ablations (preview length, curriculum), limitations |

## Team roles

| Role | Owns |
| --- | --- |
| Hardware | Car and tray assembly, sensors, system identification, test track |
| Modeling and simulation | `sim/`, route generator, calibration from measured data |
| Classical control | `control/`, MPC baseline |
| Reinforcement learning | `rl/train.py`, reward design, curriculum, preview |
| Deployment and experiments | `firmware/`, `experiments/`, data, videos, report integration |
