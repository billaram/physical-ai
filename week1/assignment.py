"""
Week 1 Final Assignment: Reach the Target

Build a controller that moves a robot arm's end-effector toward a target cube.

This combines everything from Week 1:
  - Day 1: MuJoCo simulation
  - Day 2: Reading sensor data (perception)
  - Day 3: Forward kinematics (where is the tip?)
  - Day 4: PID control (move toward the target)

YOUR TASKS:
  1. Implement `get_tip_position()` — read the end-effector position from MuJoCo
  2. Implement `simple_reach_controller()` — a basic controller that reduces the
     distance between the tip and the target
  3. (Bonus) Track and print the error over time

Run: uv run week1/assignment.py
     (or: uv run mjpython week1/assignment.py)

Estimated time: 2-3 hours
"""

import time

import mujoco
import mujoco.viewer
import numpy as np

# A simple arm with a visible target cube
SCENE_XML = """
<mujoco model="reach_task">
  <option gravity="0 0 -9.81" timestep="0.002"/>

  <visual>
    <global offwidth="1280" offheight="720"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.4 0.6 0.8" rgb2="0 0 0"
             width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1=".8 .8 .8" rgb2=".6 .6 .6"
             width="512" height="512"/>
    <material name="grid_mat" texture="grid" texrepeat="4 4" reflectance="0.1"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0.3 -1" diffuse="0.8 0.8 0.8"/>
    <light pos="1 -1 2" dir="-0.5 0.5 -1" diffuse="0.4 0.4 0.5"/>

    <!-- Ground -->
    <geom type="plane" size="2 2 0.01" material="grid_mat"/>

    <!-- Table -->
    <body name="table" pos="0 0 0.35">
      <geom type="box" size="0.4 0.4 0.02" rgba="0.4 0.25 0.15 1" mass="100"/>
      <geom type="cylinder" size="0.03 0.17" pos=" 0.35  0.35 -0.17" rgba="0.3 0.3 0.3 1"/>
      <geom type="cylinder" size="0.03 0.17" pos="-0.35  0.35 -0.17" rgba="0.3 0.3 0.3 1"/>
      <geom type="cylinder" size="0.03 0.17" pos=" 0.35 -0.35 -0.17" rgba="0.3 0.3 0.3 1"/>
      <geom type="cylinder" size="0.03 0.17" pos="-0.35 -0.35 -0.17" rgba="0.3 0.3 0.3 1"/>
    </body>

    <!-- Robot arm on table -->
    <body name="base" pos="0 0 0.37">
      <geom type="cylinder" size="0.06 0.03" rgba="0.3 0.3 0.3 1" mass="5"/>

      <!-- Joint 1: base rotation (yaw) -->
      <body name="link1" pos="0 0 0.03">
        <joint name="j1" type="hinge" axis="0 0 1" range="-170 170" damping="5"/>
        <geom type="sphere" size="0.04" rgba="0.2 0.2 0.2 1"/>

        <!-- Joint 2: shoulder pitch -->
        <body name="link2" pos="0 0 0.04">
          <joint name="j2" type="hinge" axis="0 1 0" range="-90 90" damping="5"/>
          <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.25" rgba="0.9 0.5 0.1 1"/>

          <!-- Joint 3: elbow pitch -->
          <body name="link3" pos="0 0 0.25">
            <joint name="j3" type="hinge" axis="0 1 0" range="-120 120" damping="3"/>
            <geom type="sphere" size="0.035" rgba="0.2 0.2 0.2 1"/>
            <geom type="capsule" size="0.025" fromto="0 0 0 0 0 0.22" rgba="0.2 0.6 1.0 1"/>

            <!-- End-effector (tip) -->
            <body name="tip" pos="0 0 0.22">
              <geom name="tip_geom" type="sphere" size="0.03" rgba="1 0.3 0.3 1"/>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- Target cube (the goal — we want the tip to reach this) -->
    <body name="target" pos="0.15 0.1 0.6">
      <geom name="target_geom" type="box" size="0.025 0.025 0.025"
            rgba="0 0.8 0.2 0.7" contype="0" conaffinity="0" mass="0"/>
    </body>
  </worldbody>

  <actuator>
    <motor joint="j1" gear="80" ctrlrange="-1 1" ctrllimited="true"/>
    <motor joint="j2" gear="80" ctrlrange="-1 1" ctrllimited="true"/>
    <motor joint="j3" gear="60" ctrlrange="-1 1" ctrllimited="true"/>
  </actuator>
</mujoco>
"""


def get_tip_position(model, data) -> np.ndarray:
    """
    Get the current 3D position of the robot's tip (end-effector).

    Args:
        model: MuJoCo model
        data: MuJoCo data (current state)

    Returns:
        numpy array of shape (3,) with [x, y, z] position

    TODO: Use MuJoCo's API to get the position of the "tip" body.

    Hints:
        - First, get the body ID: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tip")
        - Then read position: data.xpos[body_id]  (this is a 3D vector)
        - Return a copy: data.xpos[body_id].copy()
    """
    # ── YOUR CODE HERE ──────────────────────────────────
    return np.zeros(3)  # TODO: replace with actual tip position
    # ── END YOUR CODE ───────────────────────────────────


def get_target_position(model, data) -> np.ndarray:
    """
    Get the 3D position of the target cube.

    TODO: Same approach as get_tip_position, but for the "target" body.
    """
    # ── YOUR CODE HERE ──────────────────────────────────
    return np.zeros(3)  # TODO: replace with actual target position
    # ── END YOUR CODE ───────────────────────────────────


def simple_reach_controller(
    tip_pos: np.ndarray,
    target_pos: np.ndarray,
    joint_angles: np.ndarray,
) -> np.ndarray:
    """
    A simple controller that moves the arm toward the target.

    This is NOT a proper IK solution — it's a hacky but educational approach.
    We'll use the insight that:
      - Joint 1 (yaw) controls left/right direction
      - Joint 2 (pitch) controls up/down
      - Joint 3 (pitch) controls extension/reach

    Args:
        tip_pos: current [x, y, z] of the end-effector
        target_pos: desired [x, y, z] position
        joint_angles: current [j1, j2, j3] joint angles in radians

    Returns:
        numpy array of shape (3,) with control signals for [j1, j2, j3]
        Each value should be in [-1, 1]

    TODO: Implement a simple proportional controller.

    Strategy (think about it!):
      - Compute the error: target_pos - tip_pos → [dx, dy, dz]
      - Joint 1 (yaw, rotates around z-axis):
          If target is to the right (+x when facing forward), rotate right
          Use: dx and dy to decide direction. A simple approach:
          ctrl[0] = Kp * (some function of dx, dy)
      - Joint 2 (shoulder pitch):
          If target is higher (+z), pitch backward (negative control)
          Use: ctrl[1] = -Kp * dz
      - Joint 3 (elbow pitch):
          If target is far away, extend the arm
          Use: ctrl[2] = Kp * (horizontal_distance)

    A very simple starting point (feel free to improve!):
      error = target_pos - tip_pos
      ctrl = np.zeros(3)
      Kp = 2.0
      ctrl[0] = Kp * error[1]    # yaw responds to y-error
      ctrl[1] = -Kp * error[2]   # pitch responds to z-error
      ctrl[2] = Kp * error[0]    # elbow responds to x-error (reach)
      ctrl = np.clip(ctrl, -1, 1)
    """
    # ── YOUR CODE HERE ──────────────────────────────────
    ctrl = np.zeros(3)
    # TODO: implement your controller
    return ctrl
    # ── END YOUR CODE ───────────────────────────────────


def main():
    print("=" * 55)
    print("  WEEK 1 ASSIGNMENT: Reach the Target")
    print("=" * 55)
    print()
    print("Goal: Make the RED tip reach the GREEN target cube.")
    print("You need to implement 3 functions in this file.")
    print()

    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    data = mujoco.MjData(model)

    print(f"Robot: 3-joint arm (yaw + 2x pitch)")
    print(f"Joints: {model.njnt}, Actuators: {model.nu}")

    # Quick check: are your functions implemented?
    tip = get_tip_position(model, data)
    target = get_target_position(model, data)
    if np.allclose(tip, 0) and np.allclose(target, 0):
        print("\n  WARNING: get_tip_position() and get_target_position()")
        print("  are returning zeros. Did you implement them?")
        print("  The arm will not move until you fill in the TODOs.\n")

    print(f"\nInitial tip position:    {tip}")
    print(f"Target position:         {target}")
    print(f"Initial distance:        {np.linalg.norm(target - tip):.4f}")
    print("\nLaunching viewer — close the window to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        step_count = 0
        min_distance = float("inf")
        last_print = 0

        while viewer.is_running():
            t = time.time() - start

            # Read current state
            tip_pos = get_tip_position(model, data)
            target_pos = get_target_position(model, data)
            joint_angles = data.qpos[:3].copy()

            # Compute distance
            distance = np.linalg.norm(target_pos - tip_pos)
            min_distance = min(min_distance, distance)

            # Your controller
            ctrl = simple_reach_controller(tip_pos, target_pos, joint_angles)
            data.ctrl[:] = ctrl

            # Step simulation
            mujoco.mj_step(model, data)
            viewer.sync()

            # Print progress every second
            step_count += 1
            if t - last_print >= 1.0:
                status = "CLOSE!" if distance < 0.05 else ""
                print(f"  [{t:.1f}s] distance={distance:.4f}  "
                      f"min={min_distance:.4f}  "
                      f"ctrl=[{ctrl[0]:.2f}, {ctrl[1]:.2f}, {ctrl[2]:.2f}]  "
                      f"{status}")
                last_print = t

    print(f"\n{'=' * 55}")
    print(f"  Session complete!")
    print(f"  Minimum distance achieved: {min_distance:.4f}")
    if min_distance < 0.05:
        print(f"  GREAT JOB! The tip got within 5cm of the target!")
    elif min_distance < 0.1:
        print(f"  Good progress! Try tuning your controller gains.")
    else:
        print(f"  The arm didn't get very close. Check your implementation.")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
