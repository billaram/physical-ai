# Physical AI Hello World

Five progressive examples that go from "see a robot in simulation" to "run a VLA model" — all on your Mac.

| # | Example | What it does | Time |
|---|---------|-------------|------|
| 01 | `hello_mujoco` | Load a robot arm in MuJoCo, move joints randomly | 30 sec |
| 02 | `fetch_robot` | Fetch arm pick-and-place environment with scripted controller | 2 min |
| 03 | `train_pusht` | Train a Diffusion Policy on PushT task | 30-60 min |
| 04 | `eval_pretrained` | Evaluate a pretrained policy with visualization | 5 min |
| 05 | `smolvla_inference` | Run SmolVLA (450M) vision-language-action model | 5 min |

## Prerequisites

- macOS on Apple Silicon
- [uv](https://docs.astral.sh/uv/) (`brew install uv`)
- [ffmpeg](https://ffmpeg.org/) for video recording (`brew install ffmpeg`)

## Setup

```bash
cd physical-ai

# Examples 01-02: MuJoCo + Gymnasium-Robotics
uv sync

# Example 03-04: + LeRobot with PushT simulation
uv sync --extra lerobot

# Example 05: + SmolVLA (downloads ~1GB model)
uv sync --extra vla
```

## Running

```bash
# 01 — See a robot arm in MuJoCo (close viewer window to exit)
uv run python examples/01_hello_mujoco.py

# 02 — Fetch robot pick-and-place environment
uv run python examples/02_fetch_robot.py

# 03 — Train a Diffusion Policy on PushT (takes 30-60 min)
uv run python examples/03_train_pusht.py

# 04 — Evaluate the pretrained policy
uv run python examples/04_eval_pretrained.py

# 05 — SmolVLA inference (VLA model)
uv run python examples/05_smolvla_inference.py
```

## Concepts

- **MuJoCo**: Physics simulator for robotics (from Google DeepMind)
- **Gymnasium**: Standard API for reinforcement learning environments
- **Diffusion Policy**: Imitation learning method that models action distributions as a diffusion process
- **LeRobot**: HuggingFace library for real-world robotics with imitation/reinforcement learning
- **VLA (Vision-Language-Action)**: Model that takes image + language instruction and outputs robot actions
- **SmolVLA**: 450M-parameter VLA model, small enough to run on a laptop

## Troubleshooting

**MuJoCo viewer doesn't open on macOS**: Some scripts need the `mjpython` wrapper:
```bash
uv run mjpython examples/01_hello_mujoco.py
```

**PyTorch MPS errors**: Some ops aren't yet on Metal. Set the fallback:
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python examples/03_train_pusht.py
```

**LeRobot training is slow on CPU**: Training defaults to CPU if MPS isn't detected. Pass `--device mps` or set in the script.
