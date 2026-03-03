"""
03 — Train Your First Policy: Diffusion Policy on PushT

Trains a Diffusion Policy using LeRobot on the PushT task.
PushT = push a T-shaped block to a target pose in 2D. Simple, fast to train.

Diffusion Policy models the distribution of good actions as a denoising
diffusion process — same idea as image generation (Stable Diffusion), but
for robot actions instead of pixels.

This trains for a small number of steps (~5000) so you can see loss decrease.
A full training run (200k steps) takes several hours but reaches ~95% success.

Dependencies: `lerobot[pusht]` (installed with `uv sync --extra lerobot`).

Usage:
    uv run python examples/03_train_pusht.py
    # Or with MPS fallback for unsupported ops:
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python examples/03_train_pusht.py
"""

import os
import subprocess
import sys


def get_device():
    """Detect the best available device."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def main():
    device = get_device()
    output_dir = "outputs/train/diffusion_pusht"
    training_steps = 5000
    eval_freq = 2500
    save_freq = 2500
    batch_size = 64

    print("=" * 60)
    print("TRAINING: Diffusion Policy on PushT")
    print("=" * 60)
    print()
    print(f"  Device:         {device}")
    print(f"  Training steps: {training_steps}")
    print(f"  Batch size:     {batch_size}")
    print(f"  Eval every:     {eval_freq} steps")
    print(f"  Output:         {output_dir}")
    print()
    print("This will:")
    print("  1. Download the PushT dataset (~150MB of human demonstrations)")
    print("  2. Train a Diffusion Policy to imitate those demonstrations")
    print("  3. Periodically evaluate the policy in simulation")
    print("  4. Save checkpoints to the output directory")
    print()
    print("What is Diffusion Policy?")
    print("  Instead of predicting a single action, it learns to DENOISE")
    print("  random noise into good actions — capturing the full distribution")
    print("  of possible good behaviors. Same math as image generation.")
    print()

    # Build the training command
    cmd = [
        sys.executable, "-m", "lerobot.scripts.train",
        f"--policy.type=diffusion",
        f"--dataset.repo_id=lerobot/pusht",
        f"--env.type=pusht",
        f"--output_dir={output_dir}",
        f"--batch_size={batch_size}",
        f"--steps={training_steps}",
        f"--eval_freq={eval_freq}",
        f"--save_freq={save_freq}",
        f"--policy.device={device}",
        f"--seed=42",
    ]

    # Set environment for MPS compatibility
    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)

    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with exit code {e.returncode}")
        print("\nTroubleshooting:")
        print("  1. Make sure you ran: uv sync --extra lerobot")
        print("  2. Try: PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python examples/03_train_pusht.py")
        print("  3. Try with CPU: edit the script and set device='cpu'")
        sys.exit(1)
    except FileNotFoundError:
        print("\nError: lerobot not found. Install with:")
        print("  uv sync --extra lerobot")
        sys.exit(1)

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Checkpoint saved to: {output_dir}")
    print()
    print("Next: evaluate the pretrained (fully-trained) policy:")
    print("  uv run python examples/04_eval_pretrained.py")


if __name__ == "__main__":
    main()
