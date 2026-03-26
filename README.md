# DQN Car Navigation

2D Deep Q-Network car-navigation simulation with curriculum learning, tester-stage validation, real-time visualization, and checkpoint utilities.

Created and developed by **Richky Abednego**.

## Overview

This project trains a car agent to drive through a three-lane road using sensor input, decision-interval control, and staged obstacle layouts. The codebase separates the runtime modules, visualization flow, and a small curated test suite so the repository stays easier to maintain in production.

## Main Features

- DQN training with replay buffer, target network, and configurable memory size.
- Curriculum-based obstacle stages for progressive learning.
- Tester-stage validation using `TEST_OBSTACLES` before accepting model progress.
- Real-time pygame visualization with manual mode, all-stage mode, experiment mode, and speed test mode.
- Model-cleaning utility to re-check saved `.pth` files and remove failed checkpoints.
- Centralized tuning and simulation constants in [`main_constant.py`](./main_constant.py).

## Repository Layout

- `main_constant.py`: simulation constants, reward tuning, sensor settings, and obstacle stages.
- `main_environment.py`: environment dynamics, sensors, reward logic, and done conditions.
- `main_dqn_agent.py`: DQN network, replay buffer, action selection, and train step logic.
- `main_train.py`: training entrypoint, curriculum loop, tester validation, CSV logging, and tuning mode.
- `main_visualize.py`: pygame visualizer and evaluation runner.
- `main_visualization.py`: compatibility wrapper for `main_visualize.py`.
- `func_cleanmodel.py`: utility for validating and pruning saved model files.
- `ALL_MODELS/`: archived training outputs and comparison assets.
- `NOTES/`: short practical usage guides and copy-ready command examples.
- `test_*.py`: curated regression checks for environment, training helpers, and visualization helpers.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+ is recommended.

## Quick Start

Train a new model:

```bash
python main_train.py --episodes 500
```

Train with live visualization:

```bash
python main_train.py --episodes 200 --visualize --render-every 5
```

Continue training from a checkpoint:

```bash
python main_train.py --loadmodel models\\model_stage1A.pth --episodes 300
```

Visualize a trained model:

```bash
python main_visualize.py --model models\\model_stage1A.pth --episodes 10
```

Run the curated tests:

```bash
python -m unittest discover -p "test_*.py" -v
```

## CLI Highlights

### Training

```bash
python main_train.py --episodes 500
python main_train.py --episodes 500 --visualize --render-every 5
python main_train.py --episodes 500 --multiply
python main_train.py --episodes 500 --multivalid
python main_train.py --episodes 500 --seed 123 --memory-size 200000
python main_train.py --loadmodel models\\model_stage1A.pth --episodes 300
python main_train.py --episodes 1000 --continuous 5
python main_train.py --fine-tune-memory --tune-episodes 200 --tune-runs 3
```

### Visualization

```bash
python main_visualize.py --model models\\model_stage1A.pth --episodes 10
python main_visualize.py --model models\\model_stage1A.pth --episodes 10 --tester
python main_visualize.py --allstage --episodes 20
python main_visualize.py --manual --episodes 3
python main_visualize.py --experiment
python main_visualize.py --speedtest
python main_visualization.py --episodes 5
```

### Model Utility

```bash
python func_cleanmodel.py --models-dir models
python func_cleanmodel.py --models-dir models --max-steps 4000
python func_cleanmodel.py --models-dir models --dry-run
```

## Outputs

- New training runs create checkpoints and CSV logs under `models/` unless you change the save target in code.
- Visualization runs create CSV logs under `visualize_logs/`.
- Fine-tuning runs create result folders under `tuning/`.
- Archived experiment outputs already stored in `ALL_MODELS/` are kept as reference material.

## Notes

- Main tuning and scenario changes should be done from [`main_constant.py`](./main_constant.py).
- `main_visualization.py` is only a wrapper alias for `main_visualize.py`.
- The curated tests intentionally focus on core behavior instead of keeping many overlapping test files.
- More short usage docs are available in [`NOTES/HowToTrain.md`](./NOTES/HowToTrain.md), [`NOTES/HowToUseModel.md`](./NOTES/HowToUseModel.md), [`NOTES/Functionalities.md`](./NOTES/Functionalities.md), and [`NOTES/quickcopy.txt`](./NOTES/quickcopy.txt).

## Credit

Creator and primary developer: **Richky Abednego**
