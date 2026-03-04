"""
Day 4 Exercise: PID Controller in MuJoCo

Control a single joint of a robot arm to reach a target angle using PID.

The robot arm has 6 joints, but we'll focus on controlling just joint 2
(the shoulder pitch) to reach a target angle.

YOUR TASK:
1. Implement the PID controller in the `pid_control()` function
2. Run the script to see the arm move to the target
3. Experiment with different Kp, Ki, Kd values

Run: uv run week1/day4_pid_control.py
"""

import time

import mujoco
import mujoco.viewer
import numpy as np

# Simple 2-joint arm for clarity
ARM_XML = """
<mujoco model="pid_arm">
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

    <!-- Ground -->
    <geom type="plane" size="2 2 0.01" material="grid_mat"/>

    <!-- Robot base (fixed to ground) -->
    <body name="base" pos="0 0 0.5">
      <geom type="cylinder" size="0.08 0.04" rgba="0.3 0.3 0.3 1" mass="10"/>

      <!-- Joint 1: shoulder (we'll control this one) -->
      <body name="upper_arm" pos="0 0 0.04">
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-90 90"
               damping="2"/>
        <geom type="capsule" size="0.04" fromto="0 0 0 0 0 0.4"
              rgba="0.9 0.5 0.1 1"/>

        <!-- Joint 2: elbow (we'll keep this still) -->
        <body name="forearm" pos="0 0 0.4">
          <joint name="elbow" type="hinge" axis="0 1 0" range="-120 120"
                 damping="1"/>
          <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.35"
                rgba="0.2 0.6 1.0 1"/>

          <!-- Tip marker -->
          <body name="tip" pos="0 0 0.35">
            <geom type="sphere" size="0.04" rgba="1 0.2 0.2 1"/>
          </body>
        </body>
      </body>
    </body>

    <!-- Target indicator (visual only) -->
    <body name="target_vis" pos="0.3 0 0.8" mocap="true">
      <geom type="sphere" size="0.03" rgba="0 1 0 0.5" contype="0" conaffinity="0"/>
    </body>
  </worldbody>

  <actuator>
    <motor joint="shoulder" gear="100" ctrlrange="-1 1" ctrllimited="true"/>
    <motor joint="elbow" gear="60" ctrlrange="-1 1" ctrllimited="true"/>
  </actuator>
</mujoco>
"""


class PIDController:
    """
    A PID controller.

    YOUR TASK: Implement the `compute()` method.
    """

    def __init__(self, kp: float, ki: float, kd: float, output_limit: float = 1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit

        # Internal state
        self.integral = 0.0
        self.prev_error = None

    def compute(self, error: float, dt: float) -> float:
        """
        Compute the PID control signal.

        Args:
            error: current_target - current_value (how far off we are)
            dt: time step in seconds

        Returns:
            control signal (clamped to [-output_limit, output_limit])

        TODO: Implement PID control.

        The formula is:
            P = Kp * error
            I = Ki * (accumulated error over time)
            D = Kd * (rate of change of error)
            output = P + I + D

        Hints:
            - self.integral accumulates error: self.integral += error * dt
            - For D term: if self.prev_error is not None, derivative = (error - self.prev_error) / dt
            - Don't forget to update self.prev_error = error at the end
            - Clamp output to [-self.output_limit, self.output_limit]
        """
        # ── YOUR CODE HERE ──────────────────────────────────
        # Step 1: Proportional term
        p_term = 0.0  # TODO

        # Step 2: Integral term (accumulate error)
        i_term = 0.0  # TODO

        # Step 3: Derivative term (rate of change)
        d_term = 0.0  # TODO

        # Step 4: Combine
        output = 0.0  # TODO: p_term + i_term + d_term

        # Step 5: Remember current error for next iteration
        # TODO: self.prev_error = error

        # Step 6: Clamp to output limits
        # TODO: output = max(-self.output_limit, min(self.output_limit, output))

        # ── END YOUR CODE ───────────────────────────────────

        return output

    def reset(self):
        """Reset the controller state."""
        self.integral = 0.0
        self.prev_error = None


def run_pid_demo():
    """Run the PID control demo in MuJoCo."""
    print("Loading MuJoCo model...")
    model = mujoco.MjModel.from_xml_string(ARM_XML)
    data = mujoco.MjData(model)

    # ── EXPERIMENT: Try different PID gains! ──
    # Start with these, then modify to see what happens:
    #   - High Kp, no D:  PIDController(kp=5.0, ki=0.0, kd=0.0)  → oscillates!
    #   - Add damping:    PIDController(kp=5.0, ki=0.0, kd=1.0)  → smooth
    #   - Add integral:   PIDController(kp=5.0, ki=0.5, kd=1.0)  → removes steady-state error
    pid = PIDController(kp=5.0, ki=0.1, kd=1.0)

    # Target angle for the shoulder joint (in radians)
    # 0 = straight up, positive = leaning forward
    target_angles = [0.5, -0.3, 0.8, 0.0]  # We'll cycle through these
    target_idx = 0
    target_angle = target_angles[0]
    last_switch_time = 0
    switch_interval = 3.0  # Switch target every 3 seconds

    # For tracking performance
    errors_over_time = []

    print(f"\nPID gains: Kp={pid.kp}, Ki={pid.ki}, Kd={pid.kd}")
    print(f"Target angle: {target_angle:.2f} rad ({np.degrees(target_angle):.1f}°)")
    print("\nLaunching viewer — close the window to exit.")
    print("Watch the shoulder joint (orange link) try to reach the target angle.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        step_count = 0

        while viewer.is_running():
            t = time.time() - start

            # Switch targets periodically
            if t - last_switch_time > switch_interval:
                target_idx = (target_idx + 1) % len(target_angles)
                target_angle = target_angles[target_idx]
                pid.reset()  # Reset integral when target changes
                last_switch_time = t
                print(f"\n  [{t:.1f}s] New target: {target_angle:.2f} rad"
                      f" ({np.degrees(target_angle):.1f}°)")

            # Read current shoulder angle
            shoulder_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                                   "shoulder")
            current_angle = data.qpos[shoulder_joint_id]

            # Compute error
            error = target_angle - current_angle

            # Compute control signal using YOUR PID controller
            control = pid.compute(error, model.opt.timestep)

            # Apply control to shoulder actuator (index 0)
            data.ctrl[0] = control
            data.ctrl[1] = 0.0  # Keep elbow still

            # Step simulation
            mujoco.mj_step(model, data)
            viewer.sync()

            # Log every 100 steps
            step_count += 1
            if step_count % 500 == 0:
                print(f"  [{t:.1f}s] target={np.degrees(target_angle):.1f}° "
                      f"current={np.degrees(current_angle):.1f}° "
                      f"error={np.degrees(error):.1f}° "
                      f"control={control:.3f}")


def test_pid():
    """Quick unit test for the PID controller (no MuJoCo needed)."""
    print("=" * 50)
    print("Testing PID Controller...")
    print("=" * 50)

    pid = PIDController(kp=2.0, ki=0.5, kd=0.1)

    # Simulate a simple system: value approaches target
    value = 0.0
    target = 1.0
    dt = 0.01
    history = []

    for step in range(500):
        error = target - value
        control = pid.compute(error, dt)
        # Simple integrator: value changes proportionally to control
        value += control * dt
        history.append((step * dt, value, error))

    final_error = abs(target - value)
    print(f"\n  Target: {target}")
    print(f"  Final value: {value:.4f}")
    print(f"  Final error: {final_error:.4f}")

    if final_error < 0.05:
        print("\n  PASS: PID controller converges to target!")
    else:
        print("\n  FAIL: PID controller did not converge.")
        print("  Make sure you implemented all three terms (P, I, D)")
        print("  and remembered to update self.prev_error")

    # Check that P-only has some steady-state error
    pid_p_only = PIDController(kp=2.0, ki=0.0, kd=0.0)
    value_p = 0.0
    for _ in range(500):
        error = target - value_p
        control = pid_p_only.compute(error, dt)
        value_p += control * dt

    print(f"\n  P-only final value: {value_p:.4f} (notice it's close but not exact)")
    print(f"  P+I final value:   {value:.4f} (integral term fixes this!)")
    print("=" * 50)


if __name__ == "__main__":
    # First run the simple test (no viewer needed)
    test_pid()

    # Then run the visual demo
    print("\n\nStarting MuJoCo visual demo...")
    print("(If the viewer doesn't open, try: uv run mjpython week1/day4_pid_control.py)")
    run_pid_demo()
