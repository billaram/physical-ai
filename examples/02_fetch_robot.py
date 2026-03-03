"""
02 — Fetch Robot Pick-and-Place

Uses Gymnasium-Robotics to load a Fetch robot arm in a pick-and-place task.
Shows what a robotics RL *environment* looks like: observations, actions, rewards.

First runs random actions (robot flails), then runs a simple scripted controller
that actually picks up and moves the block.

Dependencies: `mujoco`, `gymnasium`, `gymnasium-robotics` (installed with `uv sync`).
"""

import gymnasium as gym
import gymnasium_robotics
import numpy as np

# Register gymnasium-robotics environments
gym.register_envs(gymnasium_robotics)


def print_env_info(env):
    """Print what the environment gives us to work with."""
    obs, info = env.reset()

    print("=" * 60)
    print("ENVIRONMENT: FetchPickAndPlace-v4")
    print("=" * 60)
    print()
    print("Observation space (what the robot sees):")
    for key, val in obs.items():
        print(f"  {key:20s} shape={str(val.shape):10s} dtype={val.dtype}")
    print()
    print(f"  observation: robot joint positions/velocities + gripper state")
    print(f"  achieved_goal: current position of the block (x, y, z)")
    print(f"  desired_goal: target position for the block (x, y, z)")
    print()
    print(f"Action space: {env.action_space}")
    print(f"  actions[0:3] = gripper displacement (dx, dy, dz)")
    print(f"  actions[3]   = gripper open/close (-1=close, +1=open)")
    print()
    print(f"Reward: {env.reward_range}")
    print(f"  -1 if block is NOT at goal, 0 if it IS (sparse reward)")
    print(f"  This is why RL for robotics is hard — reward is almost always -1.")
    print()


def run_random_actions(env, n_steps=100):
    """Run random actions to show the robot flailing around."""
    print("-" * 60)
    print("Phase 1: RANDOM ACTIONS (robot flails)")
    print("-" * 60)
    obs, info = env.reset()
    total_reward = 0

    for step in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            obs, info = env.reset()

    print(f"  Ran {n_steps} random steps. Total reward: {total_reward}")
    print(f"  (Reward is almost always -1 — random actions don't solve tasks!)")
    print()


def run_scripted_controller(env, n_episodes=3):
    """
    A simple scripted pick-and-place controller.

    This is NOT learned — it's hand-coded. It shows what a "solution" looks like
    and why we want to learn policies instead of scripting them.
    """
    print("-" * 60)
    print("Phase 2: SCRIPTED CONTROLLER (hand-coded heuristic)")
    print("-" * 60)

    successes = 0

    for ep in range(n_episodes):
        obs, info = env.reset()
        grip_pos = obs["observation"][:3]     # gripper position
        object_pos = obs["achieved_goal"]      # block position
        goal_pos = obs["desired_goal"]         # target position

        phase = "reach"  # reach → descend → grasp → lift → move → place
        grasp_steps = 0

        for step in range(100):
            grip_pos = obs["observation"][:3]
            object_pos = obs["achieved_goal"]
            goal_pos = obs["desired_goal"]

            action = np.zeros(4)

            if phase == "reach":
                # Move above the object
                target = object_pos.copy()
                target[2] += 0.05  # hover above
                diff = target - grip_pos
                if np.linalg.norm(diff) < 0.02:
                    phase = "descend"
                action[:3] = diff * 6.0
                action[3] = 1.0  # gripper open

            elif phase == "descend":
                # Lower onto the object
                diff = object_pos - grip_pos
                if np.linalg.norm(diff) < 0.01:
                    phase = "grasp"
                action[:3] = diff * 6.0
                action[3] = 1.0  # gripper open

            elif phase == "grasp":
                # Close gripper
                action[3] = -1.0  # gripper close
                grasp_steps += 1
                if grasp_steps > 10:
                    phase = "lift"

            elif phase == "lift":
                # Lift up
                target = grip_pos.copy()
                target[2] = goal_pos[2] + 0.1
                diff = target - grip_pos
                if grip_pos[2] > goal_pos[2] + 0.05:
                    phase = "move"
                action[:3] = diff * 6.0
                action[3] = -1.0  # gripper closed

            elif phase == "move":
                # Move above goal
                target = goal_pos.copy()
                target[2] = grip_pos[2]
                diff = target - grip_pos
                if np.linalg.norm(diff[:2]) < 0.02:
                    phase = "place"
                action[:3] = diff * 6.0
                action[3] = -1.0  # gripper closed

            elif phase == "place":
                # Lower to goal and release
                diff = goal_pos - grip_pos
                action[:3] = diff * 6.0
                action[3] = 1.0  # gripper open

            action[:3] = np.clip(action[:3], -1, 1)
            obs, reward, terminated, truncated, info = env.step(action)

            if info.get("is_success", False):
                successes += 1
                print(f"  Episode {ep+1}: SUCCESS at step {step+1}!")
                break

            if terminated or truncated:
                break
        else:
            # Check final distance
            final_dist = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
            print(f"  Episode {ep+1}: finished, final distance to goal: {final_dist:.3f}")

    print()
    print(f"  Scripted controller: {successes}/{n_episodes} successes")
    print(f"  (A learned policy would generalize better than this hard-coded approach)")
    print()


def main():
    # Create environment with rendering
    print("Creating FetchPickAndPlace-v4 environment...")
    print("(A MuJoCo viewer window will open)")
    print()

    env = gym.make("FetchPickAndPlace-v4", render_mode="human", max_episode_steps=100)

    print_env_info(env)

    run_random_actions(env, n_steps=100)
    run_scripted_controller(env, n_episodes=3)

    env.close()

    print("=" * 60)
    print("KEY TAKEAWAY:")
    print("  The scripted controller is brittle and task-specific.")
    print("  Learned policies (examples 03-05) generalize across tasks.")
    print("=" * 60)


if __name__ == "__main__":
    main()
