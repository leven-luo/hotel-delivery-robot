# Hotel Delivery Robot with an Active Two-Axis Tray

NYU course project (Reinforcement Learning and Optimal Control for Autonomous
Systems I), Project 5.

**Research question.** Along a known route, can a delivery robot jointly control
its speed and two tray angles (pitch and roll) to shorten delivery time while
keeping a freestanding cup upright and avoiding collisions?

The payload is a tall cup of about 330 mL standing freely on a tilting tray. The
robot knows where corridors, doorways and intersections are, but it can detect a
pedestrian only after the person becomes visible. A separate path follower
handles steering; the policy chooses the target speed and the two tray angles.
Whether the policy learns to slow down before occluded doorways is a hypothesis
we test, so no doorway speed limit is imposed on it.

Methods: reference controllers (preset speed profile with a fixed or
gravity-feedforward tray) and SAC as the learning method. MPC and a second
learning algorithm (PPO) are extensions if time permits.

## Status

| Part | State |
| --- | --- |
| Simulator v0 (this code) | Servo and chassis-tilt models, route generator (acceleration, braking, turns, bumps, curriculum level), gravity-feedforward controllers with command preview, Gymnasium environment, SAC/PPO training and evaluation scripts, unit tests and CI. Speed profiles are prescribed and the payload is an earlier liquid-slosh model. |
| Next (in progress) | Closed-loop speed: policy target speed -> vehicle speed within acceleration limits -> position along the route -> doorway distance, pedestrian visibility, collision and arrival. First scenario: one corridor with one blind doorway. Cup tipping/sliding model. |
| Hardware (planned) | IMU, battery and wireless module fixed inside the cup and a second IMU on the tray; two-axis 3D-printed gimbal. Chassis and servo models (candidates: Waveshare UGV02, DS3218) and purchase status to be confirmed by the hardware lead. |

**Payload criterion: cup tip-over.** Each episode records a tip label (0/1), the
time of the first confirmed tip, collisions and the delivery time. A tray tilted
on purpose with the cup still upright counts as not tipped. On hardware, tipping
is judged from the angle between the cup axis and the tray normal and its
duration, calibrated on labeled real tip-overs and bump-recovery recordings. If
the simulator only implements a quasi-static criterion, those results are
reported as "risk threshold exceeded".

## Key idea

During acceleration the effective gravity on the payload is `g - a`. Tilting the
tray to

```
pitch = atan(a_x / g),   roll = atan(a_y / g)
```

keeps the cup aligned with it, so the robot can brake and turn harder before the
cup is at risk. This formula says nothing about *how fast* to drive; the learned
policy chooses speed and tray angles together.

## Repository layout

```
sim/            Simulation (numpy core + Gymnasium env)
  dynamics.py     Liquid-slosh (equivalent pendulum, v0 payload) and servo models
  routes.py       Fixed benchmark route and random route generator (bumps, turns, braking)
  core.py         TraySim: steps car + chassis tilt + servo + payload, no RL dependencies
  tray_env.py     TrayEnv (Gymnasium): observations, reward, curriculum level, residual mode
control/        Reference controllers: fixed tray, PID level-keeping, gravity feedforward (+ preview)
rl/
  train.py        Train SAC/PPO with an automatic curriculum (fixed -> random routes)
  evaluate.py     Run every controller on benchmark + held-out routes, write a CSV
docs/           Preliminary report source
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
python -m unittest discover -s tests -t . -v     # model and env sanity checks
python -m rl.evaluate                            # compare reference controllers (v0)
python -m rl.train --algo sac --residual --curriculum --timesteps 300000
python -m rl.evaluate --model runs/<run-name>    # add a trained policy to the comparison
```

## Planned experiments

1. **Tray control at a preset speed:** fixed vs. feedforward tray under the same
   braking, turning and bump conditions; tip-over counts.
2. **Core simulation, one corridor with one blind doorway:**

   | Group | Speed and tray control | Purpose |
   | --- | --- | --- |
   | A | Preset speed profile + feedforward tray | Reference |
   | B | Learned speed + the same feedforward tray | Effect of learned speed (A vs. B) |
   | C | Jointly learned speed, pitch and roll | Added value of joint control (B vs. C) |

   Same held-out routes, pedestrian settings, visible information, control limits
   and external braking rule for all groups. Report trials, success, tip-over and
   collision rates, and delivery time of successful episodes. On no-pedestrian
   trials, compare approach speed at the blind doorway with a matched open
   corridor segment.

## Conventions

- Body frame: x forward, y left, z up. Units: SI (m, s, rad).
- Tray angle sign: positive pitch/roll leans the tray *into* the acceleration
  (front/left edge down), so the ideal tray angle has the same sign as `a_x`/`a_y`.
- v0 environment: 100 Hz control, ±20° tray limit (hardware values will be set
  after system identification).

## Roadmap

| Dates | Work |
| --- | --- |
| Nov 19–21 | Preliminary report and this repository |
| Nov 22–26 | Closed-loop speed, corridor with one blind doorway and cup model in simulation; first SAC runs; confirm and order parts |
| Nov 27–Dec 3 | Experiment 1 in simulation; training for groups B and C; tray assembly; pre-report |
| Dec 4–9 | Experiment 2 results and the no-pedestrian approach-speed comparison; hardware trials if ready |
| Dec 10 / Dec 17 | Presentation / final report |

MPC and PPO are optional extensions after the core simulation.

## Team roles

| Role | Owns |
| --- | --- |
| Hardware | Cup and tray electronics, gimbal assembly, system identification, parts status |
| Modeling and simulation | `sim/`: closed-loop speed, corridor and doorway scenario, cup model |
| Reference controllers | `control/`: feedforward and preset speed profiles; optional MPC |
| Reinforcement learning | `rl/`: SAC training for groups B and C, reward and curriculum |
| Experiments and reporting | `experiments/`, evaluation runs, plots, report |
