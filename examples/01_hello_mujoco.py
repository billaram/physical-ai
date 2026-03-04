"""
01 — Hello MuJoCo: See a Robot Arm

Loads a simple 6-DOF robot arm defined in inline XML, drops it into
MuJoCo's physics simulator, and moves the joints randomly.

Close the viewer window to exit.

Dependencies: just `mujoco` (installed with `uv sync`).
"""

import time

import mujoco
import mujoco.viewer
import numpy as np

# ── Robot definition in MuJoCo's MJCF XML format ──
# This defines a 6-joint articulated arm on a table.
# Every real robot starts as one of these XML files.
ARM_XML = """
<mujoco model="hello_arm">
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
    <material name="base_mat" rgba="0.3 0.3 0.3 1"/>
    <material name="link_mat" rgba="0.9 0.5 0.1 1"/>
    <material name="joint_mat" rgba="0.2 0.2 0.2 1"/>
    <material name="hand_mat" rgba="0.2 0.6 1.0 1"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0.3 -1" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3"/>
    <light pos="1 -1 2" dir="-0.5 0.5 -1" diffuse="0.4 0.4 0.5"/>

    <!-- Ground plane -->
    <geom type="plane" size="2 2 0.01" material="grid_mat"/>

    <!-- Table -->
    <body name="table" pos="0 0 0.4">
      <geom type="box" size="0.3 0.3 0.02" rgba="0.4 0.25 0.15 1" mass="100"/>
      <geom type="cylinder" size="0.03 0.2" pos=" 0.25  0.25 -0.2" rgba="0.3 0.3 0.3 1"/>
      <geom type="cylinder" size="0.03 0.2" pos="-0.25  0.25 -0.2" rgba="0.3 0.3 0.3 1"/>
      <geom type="cylinder" size="0.03 0.2" pos=" 0.25 -0.25 -0.2" rgba="0.3 0.3 0.3 1"/>
      <geom type="cylinder" size="0.03 0.2" pos="-0.25 -0.25 -0.2" rgba="0.3 0.3 0.3 1"/>
    </body>

    <!-- Robot arm (mounted on table) -->
    <body name="base" pos="0 0 0.42">
      <geom type="cylinder" size="0.06 0.03" material="base_mat" mass="5"/>

      <!-- Joint 1: base rotation (yaw) -->
      <body name="shoulder" pos="0 0 0.03">
        <joint name="j1" type="hinge" axis="0 0 1" range="-170 170"
               damping="5" frictionloss="0.5"/>
        <geom type="sphere" size="0.045" material="joint_mat"/>

        <!-- Joint 2: shoulder pitch -->
        <body name="upper_arm" pos="0 0 0.05">
          <joint name="j2" type="hinge" axis="0 1 0" range="-120 120"
                 damping="5" frictionloss="0.5"/>
          <geom type="capsule" size="0.035" fromto="0 0 0 0 0 0.28" material="link_mat"/>

          <!-- Joint 3: elbow pitch -->
          <body name="forearm" pos="0 0 0.28">
            <joint name="j3" type="hinge" axis="0 1 0" range="-140 140"
                   damping="3" frictionloss="0.3"/>
            <geom type="sphere" size="0.04" material="joint_mat"/>
            <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.24" material="link_mat"/>

            <!-- Joint 4: forearm rotation (roll) -->
            <body name="wrist_base" pos="0 0 0.24">
              <joint name="j4" type="hinge" axis="0 0 1" range="-170 170"
                     damping="1" frictionloss="0.1"/>
              <geom type="sphere" size="0.035" material="joint_mat"/>

              <!-- Joint 5: wrist pitch -->
              <body name="wrist" pos="0 0 0.02">
                <joint name="j5" type="hinge" axis="0 1 0" range="-120 120"
                       damping="1" frictionloss="0.1"/>
                <geom type="capsule" size="0.025" fromto="0 0 0 0 0 0.12" material="hand_mat"/>

                <!-- Joint 6: wrist roll -->
                <body name="hand" pos="0 0 0.12">
                  <joint name="j6" type="hinge" axis="0 0 1" range="-170 170"
                         damping="0.5" frictionloss="0.05"/>
                  <geom type="box" size="0.03 0.06 0.01" material="hand_mat"/>

                  <!-- Fingertips (cosmetic) -->
                  <geom type="box" size="0.008 0.008 0.03" pos="0  0.04 0.03" rgba="0.5 0.5 0.5 1"/>
                  <geom type="box" size="0.008 0.008 0.03" pos="0 -0.04 0.03" rgba="0.5 0.5 0.5 1"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- Some objects on the table to make the scene interesting -->
    <body name="red_cube" pos="0.2 0.15 0.45">
      <freejoint/>
      <geom type="box" size="0.025 0.025 0.025" rgba="0.9 0.2 0.2 1" mass="0.1"/>
    </body>
    <body name="green_sphere" pos="-0.15 0.2 0.45">
      <freejoint/>
      <geom type="sphere" size="0.03" rgba="0.2 0.8 0.3 1" mass="0.05"/>
    </body>
  </worldbody>

  <actuator>
    <motor joint="j1" gear="80" ctrlrange="-1 1" ctrllimited="true"/>
    <motor joint="j2" gear="80" ctrlrange="-1 1" ctrllimited="true"/>
    <motor joint="j3" gear="60" ctrlrange="-1 1" ctrllimited="true"/>
    <motor joint="j4" gear="40" ctrlrange="-1 1" ctrllimited="true"/>
    <motor joint="j5" gear="30" ctrlrange="-1 1" ctrllimited="true"/>
    <motor joint="j6" gear="20" ctrlrange="-1 1" ctrllimited="true"/>
  </actuator>
</mujoco>
"""


def main():
    print("Loading MuJoCo model...")
    model = mujoco.MjModel.from_xml_string(ARM_XML)
    data = mujoco.MjData(model)

    print(f"  Joints:    {model.njnt}")
    print(f"  Actuators: {model.nu}")
    print(f"  Bodies:    {model.nbody}")
    print(f"  Timestep:  {model.opt.timestep}s")
    print()
    print("Launching viewer — close the window to exit.")
    print("  - Left-click + drag to rotate")
    print("  - Right-click + drag to pan")
    print("  - Scroll to zoom")

    # Generate smooth random joint targets using sine waves
    rng = np.random.default_rng(42)
    frequencies = rng.uniform(0.3, 0.8, size=model.nu)
    phases = rng.uniform(0, 2 * np.pi, size=model.nu)
    amplitudes = rng.uniform(0.3, 0.8, size=model.nu)

    # Set up a key callback so the arm moves with sinusoidal targets
    # during simulation (the Simulate GUI handles its own physics loop)
    import threading

    def control_loop():
        """Background thread that updates control targets."""
        start = time.time()
        while True:
            t = time.time() - start
            data.ctrl[:] = amplitudes * np.sin(2 * np.pi * frequencies * t + phases)
            time.sleep(0.01)

    controller = threading.Thread(target=control_loop, daemon=True)
    controller.start()

    # launch() works on macOS without mjpython (unlike launch_passive)
    # It opens the full Simulate GUI — press Space to play/pause physics
    print("\n  TIP: Press SPACE in the viewer to start/stop physics simulation.")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
