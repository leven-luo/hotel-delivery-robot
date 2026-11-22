# Hotel Delivery Robot with an Active Two-Axis Tray

NYU course project (Reinforcement Learning and Optimal Control for Autonomous
Systems I), Project 5.

A delivery robot carries a tall cup on an actively tilting two-axis tray and must
reach a guest room from the hotel lobby as fast as possible without tipping the
cup or colliding with people. The policy jointly controls **forward speed, tray
pitch and tray roll**. The hotel map (corridors, doors, turns, intersections) is
known, but whether a person steps out of a door or a blind corner is not: events
occur with location-dependent probabilities and the policy only sees what its
sensors could observe at that moment.

We compare optimal-control baselines (fixed tray, gravity feedforward with
preview, LQR, conservative MPC) against reinforcement learning (SAC/PPO,
recurrent policies for partial observability, residual RL on top of the
feedforward), first in simulation and then on real hardware.

## Status

| Part | State |
| --- | --- |
| Simulator v0 (this code) | Tray + servo + chassis-tilt dynamics, liquid-slosh cup model, route generator with bumps and curriculum, Gymnasium env, baselines, training/evaluation scripts, tests |
| In progress | Cup-tipping model (tall cup, IMU-based tip detection), speed as an action, hotel map with stochastic pedestrians at doors and blind corners |
| Hardware (planned) | Waveshare UGV02 chassis; 3D-printed gimbal (fixed bracket / outer frame / food tray), DS3218 servo + 625ZZ bearing per axis; ESP32 tray controller; IMUs on tray and inside the cup over ESP-NOW |

The failure metric moved from spilled water to **cup tipping**: weighing during
motion is corrupted by bumps and braking, so a tall cup carries an IMU and a tip
is declared when its tilt relative to the tray IMU exceeds a calibrated threshold
for a minimum duration.

## Key idea

During acceleration the liquid surface settles perpendicular to the effective
gravity `g - a`, not to the cup. Tilting the tray to

```
pitch = atan(a_x / g),   roll = atan(a_y / g)
```

keeps the cup aligned with the effective gravity, so the robot can brake and turn
harder before the cup tips. Servo lag, sensor noise and bumps make this formula
insufficient on its own, and it says nothing about *how fast* to drive. RL
decides both: it keeps speed on straight, visible corridors and slows down before
doors and blind corners, without being told whether someone is behind them.

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
| Nov 19 | Preliminary report + this repo | Simulator v0, baselines, project plan |
| Nov 26 | Internal checkpoint | Cup-tipping model, speed action, hotel map in sim; gimbal assembled and tracking commands |
| Dec 3 | Pre-report | RL and MPC trained/implemented, first hardware runs |
| Dec 10 | Presentation | Success / tip / collision rates and delivery time for all methods, demo video |
| Dec 17 | Final report | Analysis, ablations (preview, recurrent policy, curriculum), limitations |

Metrics: success rate, tip rate, collision rate, and mean time of successful
episodes (so that early failures do not look fast).

## Team roles

| Role | Owns |
| --- | --- |
| Hardware | Car and tray assembly, sensors, system identification, test track |
| Modeling and simulation | `sim/`, route generator, calibration from measured data |
| Classical control | `control/`, MPC baseline |
| Reinforcement learning | `rl/train.py`, reward design, curriculum, preview |
| Deployment and experiments | `firmware/`, `experiments/`, data, videos, report integration |
