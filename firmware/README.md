# Firmware (ESP32)

Planned contents:

- `tray_controller/`: 100 Hz loop on the ESP32 that reads both IMUs and the ToF sensor,
  runs the policy, and writes the two servo commands.
- Policy inference: the trained actor (2 x 64 MLP) exported to a C header of weights
  (exporter script to be added in `rl/export_policy.py`).
- Logging: stream measurements and commands over serial/Wi-Fi so real runs can be
  compared with simulation.

Keep the observation layout identical to `sim/tray_env.py` (same order, units and scaling).
