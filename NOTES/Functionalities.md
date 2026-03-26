# Functionalities

## Core Runtime Files

- `main_constant.py` stores simulation scale, speed calibration, reward values, sensor setup, curriculum obstacles, and tester obstacles.
- `main_environment.py` handles movement, decision-interval speed ramps, steering transitions, ray sensors, reward calculation, finish logic, and runtime obstacle management.
- `main_dqn_agent.py` contains the neural network, replay buffer, epsilon-greedy action selection, target-network update flow, and model save/load behavior.
- `main_train.py` drives curriculum learning, tester validation, logging, checkpoint naming, cycle-based training, and memory-size tuning mode.
- `main_visualize.py` runs the pygame viewer, manual mode, all-stage mode, tester mode, experiment mode, and speed test mode.
- `func_cleanmodel.py` replays tester validation over saved `.pth` files and can delete failed checkpoints.

## Training Flow

- Training starts from stage 0 and uses `OBSTACLES` from `main_constant.py`.
- Actions are held for one `DECISION_INTERVAL`, so steering and speed follow decision-based control instead of changing every frame.
- Final-stage stopping is tied to valid tester success, not only raw episode count.
- Tester validation uses `TEST_OBSTACLES` and records fail counts in `tester_stage.csv`.
- Continuous cycle mode labels outputs with `A`, `B`, `C`, and so on.

## Visualization Modes

- Standard model playback loads a `.pth` file and runs the learned policy.
- `--manual` lets you drive with keyboard input.
- `--allstage` evaluates progress across every curriculum stage.
- `--tester` swaps normal obstacles for tester-stage layouts.
- `--experiment` opens obstacle-spawn controls for sandbox testing.
- `--speedtest` runs an endless speed-control visualization mode.

## Outputs And Logs

- Training checkpoints and CSV logs are written to `models/`.
- Visualizer run summaries are written to `visualize_logs/`.
- Fine-tuning sweeps are written to `tuning/`.
- Archived comparison results in `ALL_MODELS/` stay available as reference data.

## Production-Curated Tests

- `test_environment_core.py` checks sensor/state payloads, speed-interval behavior, finish-line tracking, and runtime obstacle management.
- `test_training_core.py` checks stop helpers, reward-format helpers, tester-stage CSV handling, and tester validation bookkeeping.
- `test_visualize_core.py` checks visualize CSV helpers, experiment planner behavior, and warning-close counting.

## What To Change First

- Tuning simulation behavior: edit `main_constant.py`.
- Changing training logic: edit `main_train.py` and `main_dqn_agent.py`.
- Changing environment/reward behavior: edit `main_environment.py`.
- Changing viewer behavior or evaluation flow: edit `main_visualize.py`.
