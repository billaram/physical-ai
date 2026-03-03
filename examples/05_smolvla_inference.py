"""
05 — SmolVLA Inference: Vision-Language-Action Model

Loads SmolVLA (450M params), the smallest VLA model that runs on a laptop.

A VLA model takes:
  - An image (what the robot camera sees)
  - A language instruction ("pick up the red block")
And outputs:
  - Robot actions (joint velocities, gripper commands, etc.)

This is the frontier of Physical AI — connecting vision, language understanding,
and robot control in a single model.

Dependencies: `lerobot[smolvla]` (installed with `uv sync --extra vla`).

Usage:
    uv run python examples/05_smolvla_inference.py
    # Or with MPS fallback:
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python examples/05_smolvla_inference.py
"""

import os
import time

# MPS fallback for ops not yet on Metal
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch


def get_device():
    """Detect the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    device = get_device()
    model_id = "lerobot/smolvla_base"

    print("=" * 60)
    print("SMOLVLA: Vision-Language-Action Model Inference")
    print("=" * 60)
    print()
    print(f"  Model:  {model_id} (450M params)")
    print(f"  Device: {device}")
    print()
    print("SmolVLA architecture:")
    print("  Image  ──→ SigLIP vision encoder ──→ visual tokens")
    print("  Text   ──→ SmolLM2 tokenizer     ──→ text tokens")
    print("  [visual + text tokens] ──→ SmolLM2 backbone ──→ latent")
    print("  latent ──→ Action Expert (flow matching) ──→ robot actions")
    print()

    # ── Load model ──
    print("Loading SmolVLA model (first run downloads ~1GB)...")
    t0 = time.time()

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = SmolVLAPolicy.from_pretrained(model_id)
    policy = policy.to(device).eval()
    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.1f}s")
    print()

    # ── Load a sample observation from a dataset ──
    print("Loading a sample observation from the libero dataset...")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("lerobot/libero")
    sample = dict(dataset[0])

    # Show what the observation contains
    print("  Sample observation keys:")
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            print(f"    {key:40s} shape={str(list(val.shape)):20s} dtype={val.dtype}")
        else:
            print(f"    {key:40s} = {val}")
    print()

    # ── Set up pre/post processors ──
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    # ── Run inference ──
    print("Running inference...")
    print("  Input: camera image + language instruction from dataset")
    print("  Output: predicted robot actions")
    print()

    t0 = time.time()
    with torch.inference_mode():
        batch = preprocess(sample)
        pred_action = policy.select_action(batch)
        pred_action = postprocess(pred_action)
    inference_time = time.time() - t0

    print(f"  Predicted action tensor:")
    if isinstance(pred_action, dict):
        for key, val in pred_action.items():
            if isinstance(val, torch.Tensor):
                print(f"    {key}: shape={list(val.shape)}, values={val.flatten()[:6].tolist()}")
    elif isinstance(pred_action, torch.Tensor):
        print(f"    shape: {list(pred_action.shape)}")
        print(f"    values (first 6): {pred_action.flatten()[:6].tolist()}")
    print()
    print(f"  Inference time: {inference_time:.3f}s")
    print()

    # ── Summary ──
    print("=" * 60)
    print("WHAT JUST HAPPENED:")
    print("  1. SmolVLA looked at a camera image from a real robot dataset")
    print("  2. It read a language instruction ('pick up the object')")
    print("  3. It predicted the next robot actions to execute")
    print()
    print("This is the VLA pipeline: Vision + Language → Action")
    print()
    print("In a real deployment, you'd run this in a loop:")
    print("  while task_not_done:")
    print("    image = camera.capture()")
    print("    action = vla.predict(image, instruction)")
    print("    robot.execute(action)")
    print("=" * 60)


if __name__ == "__main__":
    main()
