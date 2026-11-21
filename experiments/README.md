# Real-robot experiments

## System identification (feed results back into `sim/`)

| Quantity | How to measure | Where it goes |
| --- | --- | --- |
| Servo delay, time constant, max rate | Step command, log tray IMU angle | `ServoParams` |
| Slosh frequency and damping | Tap the cup, film the surface at 60+ fps, count oscillations | `SloshParams.freq_hz`, `damping` |
| Spill angle for each fill level | Tilt the cup slowly until water reaches the rim | `SloshParams.spill_angle_deg` |
| Car acceleration lag | Step speed command, log chassis IMU | `routes.MOTOR_TAU` |
| Chassis squat/roll per m/s^2 | Chassis IMU tilt vs. acceleration | `PhysicsParams.susp_gain` |

## Trial protocol

1. Fill the cup to the mark and weigh it (0.1 g scale).
2. Run the layout with one controller; record video of the cup.
3. Weigh again; spilled grams = before - after.
4. Log the trial in `trial_log_template.csv` (copy it to `trials.csv`).

Use the same 5 benchmark layouts for every controller; keep 2 layouts unseen until the final test.
