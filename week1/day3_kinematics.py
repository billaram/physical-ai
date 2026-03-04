"""
Day 3 Exercise: 2D Forward Kinematics

A 2-link planar robot arm — the simplest possible robot.
Your job: compute where the tip (end-effector) ends up.

         (tip)
          o
         /
        / L2    ← link 2
       /
      o  ← joint 2 (elbow)
      |
      | L1     ← link 1
      |
      o  ← joint 1 (shoulder, fixed to ground)

Joint 1 angle (θ1): measured from the positive x-axis
Joint 2 angle (θ2): measured relative to link 1

Run: uv run week1/day3_kinematics.py
"""

import math


def forward_kinematics(theta1_deg: float, theta2_deg: float, L1: float, L2: float):
    """
    Compute the (x, y) position of the tip of a 2-link planar arm.

    Args:
        theta1_deg: angle of joint 1 in degrees (from x-axis)
        theta2_deg: angle of joint 2 in degrees (relative to link 1)
        L1: length of link 1
        L2: length of link 2

    Returns:
        (elbow_x, elbow_y, tip_x, tip_y)

    TODO: Fill in the math below.

    Hints:
        - Convert degrees to radians: math.radians(angle)
        - Elbow position: (L1 * cos(θ1), L1 * sin(θ1))
        - Tip position depends on the TOTAL angle (θ1 + θ2), not just θ2
        - x_tip = x_elbow + L2 * cos(θ1 + θ2)
        - y_tip = y_elbow + L2 * sin(θ1 + θ2)
    """
    theta1 = math.radians(theta1_deg)
    theta2 = math.radians(theta2_deg)

    # ── YOUR CODE HERE ──────────────────────────────────
    # Step 1: Compute elbow position
    elbow_x = 0.0  # TODO: replace with L1 * cos(θ1)
    elbow_y = 0.0  # TODO: replace with L1 * sin(θ1)

    # Step 2: Compute tip position
    tip_x = 0.0  # TODO: replace with elbow_x + L2 * cos(θ1 + θ2)
    tip_y = 0.0  # TODO: replace with elbow_y + L2 * sin(θ1 + θ2)
    # ── END YOUR CODE ───────────────────────────────────

    return elbow_x, elbow_y, tip_x, tip_y


def draw_arm(theta1_deg: float, theta2_deg: float, L1: float, L2: float):
    """Draw the arm configuration as ASCII art (simple version)."""
    elbow_x, elbow_y, tip_x, tip_y = forward_kinematics(
        theta1_deg, theta2_deg, L1, L2
    )

    print(f"\n  Configuration: θ1={theta1_deg}°, θ2={theta2_deg}°")
    print(f"  Link lengths:  L1={L1}, L2={L2}")
    print(f"  Elbow at:      ({elbow_x:.3f}, {elbow_y:.3f})")
    print(f"  Tip at:        ({tip_x:.3f}, {tip_y:.3f})")
    print(f"  Reach:         {math.sqrt(tip_x**2 + tip_y**2):.3f}")
    print(f"  Max reach:     {L1 + L2:.3f}")


def test_your_solution():
    """Test cases to verify your implementation."""
    print("=" * 50)
    print("Testing your forward kinematics...")
    print("=" * 50)

    test_cases = [
        # (θ1, θ2, L1, L2, expected_tip_x, expected_tip_y, description)
        (0, 0, 1.0, 1.0, 2.0, 0.0, "Arm fully extended along x-axis"),
        (90, 0, 1.0, 1.0, 0.0, 2.0, "Arm fully extended along y-axis"),
        (0, 90, 1.0, 1.0, 1.0, 1.0, "Elbow bent 90°"),
        (0, 180, 1.0, 1.0, 0.0, 0.0, "Arm folded back on itself"),
        (45, 0, 1.0, 0.5, 1.5 * math.cos(math.radians(45)),
         1.5 * math.sin(math.radians(45)), "45° with different link lengths"),
    ]

    all_passed = True
    for theta1, theta2, L1, L2, expected_x, expected_y, desc in test_cases:
        _, _, tip_x, tip_y = forward_kinematics(theta1, theta2, L1, L2)
        x_ok = abs(tip_x - expected_x) < 0.001
        y_ok = abs(tip_y - expected_y) < 0.001
        status = "PASS" if (x_ok and y_ok) else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"\n  [{status}] {desc}")
        print(f"    θ1={theta1}°, θ2={theta2}°, L1={L1}, L2={L2}")
        print(f"    Expected: ({expected_x:.3f}, {expected_y:.3f})")
        print(f"    Got:      ({tip_x:.3f}, {tip_y:.3f})")

    print("\n" + "=" * 50)
    if all_passed:
        print("All tests passed! You understand forward kinematics!")
    else:
        print("Some tests failed. Check your math and try again.")
        print("Remember: tip_x = L1*cos(θ1) + L2*cos(θ1+θ2)")
    print("=" * 50)


def explore_workspace():
    """Visualize what the arm can reach by sweeping joint angles."""
    print("\n\nExploring the workspace (all positions the tip can reach):")
    print("Each '.' is a reachable position.\n")

    L1, L2 = 1.0, 0.7
    grid_size = 40
    grid = [[" " for _ in range(grid_size)] for _ in range(grid_size)]

    for theta1 in range(0, 360, 5):
        for theta2 in range(-180, 180, 5):
            _, _, tip_x, tip_y = forward_kinematics(theta1, theta2, L1, L2)
            # Map to grid coordinates
            gx = int((tip_x + 2.0) / 4.0 * grid_size)
            gy = int((tip_y + 2.0) / 4.0 * grid_size)
            if 0 <= gx < grid_size and 0 <= gy < grid_size:
                grid[grid_size - 1 - gy][gx] = "."

    # Mark the origin (shoulder)
    ox = int(2.0 / 4.0 * grid_size)
    oy = int(2.0 / 4.0 * grid_size)
    if 0 <= ox < grid_size and 0 <= oy < grid_size:
        grid[grid_size - 1 - oy][ox] = "O"

    for row in grid:
        print("  " + "".join(row))

    print(f"\n  O = shoulder (origin)")
    print(f"  L1={L1}, L2={L2}")
    print(f"  Max reach = {L1 + L2}")
    print(f"  Min reach = {abs(L1 - L2)}")
    print(f"  The workspace is a ring (annulus)!")


if __name__ == "__main__":
    # Step 1: Test your solution
    test_your_solution()

    # Step 2: Explore different configurations
    print("\n\n--- Exploring Configurations ---")
    draw_arm(0, 0, 1.0, 1.0)      # Fully extended right
    draw_arm(90, 0, 1.0, 1.0)     # Fully extended up
    draw_arm(45, -45, 1.0, 0.8)   # Bent configuration
    draw_arm(0, 180, 1.0, 1.0)    # Folded back

    # Step 3: Visualize workspace (only works if FK is implemented)
    explore_workspace()
