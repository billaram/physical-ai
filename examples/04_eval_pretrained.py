"""
04 — Evaluate a Pretrained Policy

Downloads a fully-trained Diffusion Policy from HuggingFace Hub
(lerobot/diffusion_pusht) and evaluates it in the PushT simulation.

This shows what a "working" policy looks like — the robot reliably pushes
the T-block to the target pose, achieving ~95% overlap and ~65% success rate.

Dependencies: `lerobot[pusht]` (installed with `uv sync --extra lerobot`).

Usage:
    uv run python examples/04_eval_pretrained.py
    # Or with MPS fallback:
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python examples/04_eval_pretrained.py
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
    model_id = "lerobot/diffusion_pusht"
    n_episodes = 10
    output_dir = "outputs/eval/diffusion_pusht"

    print("=" * 60)
    print("EVALUATING: Pretrained Diffusion Policy on PushT")
    print("=" * 60)
    print()
    print(f"  Model:    {model_id}")
    print(f"  Device:   {device}")
    print(f"  Episodes: {n_episodes}")
    print(f"  Output:   {output_dir}")
    print()
    print("This will:")
    print("  1. Download the pretrained model from HuggingFace (~300MB)")
    print("  2. Run it in the PushT simulation for several episodes")
    print("  3. Report success rate and save evaluation videos")
    print()

    # Build the evaluation command
    cmd = [
        sys.executable, "-m", "lerobot.scripts.eval",
        f"--policy.path={model_id}",
        f"--env.type=pusht",
        f"--eval.n_episodes={n_episodes}",
        f"--eval.batch_size=1",
        f"--output_dir={output_dir}",
        f"--policy.device={device}",
        "--policy.use_amp=false",
    ]

    # Set environment for MPS compatibility
    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)

    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nEvaluation failed with exit code {e.returncode}")
        print("\nTroubleshooting:")
        print("  1. Make sure you ran: uv sync --extra lerobot")
        print("  2. Make sure ffmpeg is installed: brew install ffmpeg")
        print("  3. Try with CPU: edit the script and set device='cpu'")
        sys.exit(1)
    except FileNotFoundError:
        print("\nError: lerobot not found. Install with:")
        print("  uv sync --extra lerobot")
        sys.exit(1)

    print()
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Videos saved to: {output_dir}")
    print()
    print(f"The pretrained model achieves ~95% avg overlap, ~65% success rate.")
    print(f"Compare that to your short training run from example 03!")
    print()
    print("Next: try a Vision-Language-Action model:")
    print("  uv run python examples/05_smolvla_inference.py")


if __name__ == "__main__":
    main()
