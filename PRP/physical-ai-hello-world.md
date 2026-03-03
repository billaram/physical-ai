# PRP: Physical AI Hello World Codebase

**Status:** Complete
**Date:** 2026-03-03

---

## Goal

Set up a progressive "hello world" Physical AI codebase on macOS (Apple Silicon) that goes from basic robotics simulation to running a VLA model — for someone with Python/CV/LLM experience but new to robotics.

## Problem

Physical AI is a fast-moving field with fragmented tooling. A newcomer needs a clear, progressive path from "see a robot in a simulator" to "run a vision-language-action model" — all working locally on a Mac with minimal setup friction.

## Solution

Five self-contained Python examples, each building on the last, managed entirely via `uv` with no conda/pip/setup scripts.

### Architecture

```
physical-ai/
├── pyproject.toml                # uv-managed deps with optional groups
├── .python-version               # Python 3.12
├── README.md                     # Setup + usage instructions
├── examples/
│   ├── 01_hello_mujoco.py        # See a robot arm in MuJoCo
│   ├── 02_fetch_robot.py         # Gymnasium-Robotics pick-and-place
│   ├── 03_train_pusht.py         # Train a Diffusion Policy (LeRobot)
│   ├── 04_eval_pretrained.py     # Evaluate a pretrained policy
│   └── 05_smolvla_inference.py   # SmolVLA (450M) VLA inference
```

### Dependency Groups

| Group | Command | What it adds |
|-------|---------|-------------|
| Core | `uv sync` | mujoco, gymnasium, gymnasium-robotics |
| LeRobot | `uv sync --extra lerobot` | lerobot[pusht] (~94 packages) |
| VLA | `uv sync --extra vla` | lerobot[smolvla] |

## Examples — What Each One Teaches

### 01 — Hello MuJoCo ("See a Robot")
- **Time:** 30 seconds
- **Deps:** mujoco only
- **What:** Loads a 6-DOF robot arm defined in inline MJCF XML, renders in interactive viewer, moves joints with smooth sinusoidal targets
- **Teaches:** MuJoCo physics simulation, MJCF robot definition format, joints/actuators/bodies
- **Key detail:** Inline XML (no external model files needed). Includes a table, objects, and a realistic-looking arm with 6 joints and a gripper

### 02 — Fetch Robot ("Robot + Task")
- **Time:** 2 minutes
- **Deps:** gymnasium-robotics
- **What:** Creates FetchPickAndPlace-v4 environment, runs random actions (robot flails), then runs a hand-coded scripted controller
- **Teaches:** RL environments, observation/action/reward structure, GoalEnv API, why learned policies beat scripted ones
- **Key detail:** Updated to v4 (v3 is deprecated in gymnasium-robotics 1.4.2). Prints env info so user understands obs/action spaces

### 03 — Train PushT ("Train Your First Policy")
- **Time:** 30–60 min
- **Deps:** lerobot[pusht]
- **What:** Trains a Diffusion Policy on PushT (push T-block to target pose) for 5000 steps via LeRobot CLI
- **Teaches:** Imitation learning, diffusion policy, training loops, checkpointing
- **Key detail:** Uses `subprocess` to call `lerobot.scripts.train`. Auto-detects MPS/CUDA/CPU. Sets `PYTORCH_ENABLE_MPS_FALLBACK=1` for Metal compatibility

### 04 — Eval Pretrained ("See a Trained Policy Work")
- **Time:** 5 minutes
- **Deps:** lerobot[pusht]
- **What:** Downloads `lerobot/diffusion_pusht` (fully trained, ~95% overlap) and evaluates in PushT sim, saving videos
- **Teaches:** Model loading from Hub, evaluation, what "success" looks like
- **Key detail:** Uses `lerobot.scripts.eval` CLI. Pretrained model achieves ~95% overlap / ~65% success rate

### 05 — SmolVLA Inference ("VLA Model")
- **Time:** 5 minutes (+ ~1GB model download on first run)
- **Deps:** lerobot[smolvla]
- **What:** Loads SmolVLA (450M params), feeds it a sample image + language instruction from the libero dataset, gets predicted robot actions
- **Teaches:** VLA architecture (SigLIP → SmolLM2 → Action Expert), vision-language-action pipeline
- **Key detail:** Uses `SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")` + `make_pre_post_processors` API

## Technical Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Package manager | uv only | No conda/pip complexity, lockfile for reproducibility |
| Python version | 3.12 | Latest well-supported, compatible with all deps |
| Example 01 robot | Inline MJCF XML | Zero extra dependencies, educational (shows XML format) |
| Fetch env version | v4 (not v3) | v3 deprecated in gymnasium-robotics 1.4.2 |
| Training CLI | subprocess to lerobot scripts | Most robust, matches official docs |
| VLA model | SmolVLA 450M | Smallest VLA that runs on a laptop |
| Build backend | hatchling | Standard, needed `packages = ["examples"]` since no importable package |

## Verified

| Check | Result |
|-------|--------|
| `uv sync` installs core deps | mujoco 3.5.0, gymnasium 1.2.3, gymnasium-robotics 1.4.2 |
| Example 01 XML loads | 8 joints, 6 actuators, 11 bodies — steps correctly |
| Example 02 env creates | FetchPickAndPlace-v4 resets, steps, returns correct obs shape |
| `uv sync --extra lerobot` resolves | 94 additional packages including lerobot, diffusers, torch |
| `uv sync --extra vla` resolves | Adds SmolVLA dependencies |

## Known Considerations

- **MuJoCo viewer on macOS:** Some scripts may need `mjpython` wrapper instead of `python`
- **MPS fallback:** Set `PYTORCH_ENABLE_MPS_FALLBACK=1` for PyTorch ops not yet on Metal
- **ffmpeg:** Needed for video recording in example 04 — install via `brew install ffmpeg`
- **Training time:** 5000 steps is enough to see loss decrease but not enough for good performance. Full training (200k steps) takes hours
- **SmolVLA first run:** Downloads ~1GB model from HuggingFace Hub

## Files Created

1. `.python-version` — pins Python 3.12
2. `pyproject.toml` — uv project config with `lerobot` and `vla` optional dependency groups
3. `README.md` — setup instructions, example descriptions, troubleshooting
4. `examples/01_hello_mujoco.py` — 164 lines, inline 6-DOF arm with MuJoCo viewer
5. `examples/02_fetch_robot.py` — 192 lines, FetchPickAndPlace-v4 with scripted controller
6. `examples/03_train_pusht.py` — 117 lines, LeRobot Diffusion Policy training wrapper
7. `examples/04_eval_pretrained.py` — 104 lines, pretrained policy evaluation wrapper
8. `examples/05_smolvla_inference.py` — 138 lines, SmolVLA inference demo
