# Week 1: Physical AI Foundations — Baby Steps

> Goal: Build intuition for how robots "see, think, and act" — no prior robotics knowledge needed.

---

## Schedule Overview

| Day | Theme | Time | Deliverable |
|-----|-------|------|-------------|
| 1 | What is a robot, really? | 1-2 hrs | Run MuJoCo sim, answer concept questions |
| 2 | How robots see (Perception) | 1-2 hrs | Read about sensors, complete quiz |
| 3 | How robots move (Kinematics 101) | 1.5-2 hrs | Forward kinematics coding exercise |
| 4 | How robots decide (Control basics) | 1-2 hrs | Implement a PID controller |
| 5 | How robots learn from demos | 1-2 hrs | Read Imitation Learning basics |
| 6 | The big picture: VLAs | 1-1.5 hrs | Read VLA overview, complete final quiz |
| 7 | Week 1 Project | 2-3 hrs | Programming assignment |

---

## Day 1: What is a Robot, Really?

### Core Concept
A robot is: **sensors** (eyes) + **actuators** (muscles) + **controller** (brain), all operating in a physics loop.

```
          ┌─────────────────────────┐
          │      ENVIRONMENT        │
          │    (physics, objects)    │
          └────┬───────────────▲────┘
               │               │
         observe           act
               │               │
          ┌────▼───────────────┴────┐
          │         ROBOT           │
          │  sensors → brain → motors│
          └─────────────────────────┘
```

Every robot runs this loop hundreds of times per second:
1. **Sense** — read cameras, joint encoders, force sensors
2. **Think** — decide what to do next
3. **Act** — send commands to motors
4. **Repeat**

### Reading (30 min)
- [ ] Read this excellent 10-min intro: [MuJoCo Documentation — Overview](https://mujoco.readthedocs.io/en/stable/overview.html) (just the overview page)
- [ ] Skim the MJCF XML in `examples/01_hello_mujoco.py` — try to understand each section (don't worry about memorizing)

### Hands-On (30 min)
- [ ] Run `uv run examples/01_hello_mujoco.py` (or `uv run mjpython examples/01_hello_mujoco.py` if viewer doesn't open)
- [ ] Play with the viewer: rotate, zoom, watch the arm move
- [ ] Try modifying the XML in the script:
  - Change a link length (e.g., `fromto="0 0 0 0 0 0.28"` → `"0 0 0 0 0 0.4"`)
  - Change gravity (e.g., `gravity="0 0 -1.0"` for moon-like)
  - Add another cube object
  - Re-run and observe the difference

### Key Vocabulary
| Term | Meaning | Analogy |
|------|---------|---------|
| **Joint** | Connection between two rigid parts that allows movement | Your elbow or shoulder |
| **Link** | Rigid body segment between joints | Your forearm bone |
| **Actuator** | Motor that applies force/torque at a joint | Your muscle |
| **DOF** (Degrees of Freedom) | Number of independent ways a robot can move | A door has 1 DOF (hinge), your arm has 7 |
| **End-effector** | The "hand" or tool at the tip | Your fingertips |
| **State** | All joint positions + velocities at one instant | A snapshot of the robot's pose |
| **Timestep** | How often the physics simulation updates | Like frames in a video game |

### Day 1 Questions (answer in your notes)
1. How many joints does the arm in example 01 have?
2. What does `damping` do on a joint? (Hint: what happens to a door with vs without a door closer?)
3. If `timestep=0.002`, how many physics steps happen per second?
4. What is the difference between a `hinge` joint and a `freejoint`?

---

## Day 2: How Robots See (Perception)

### Core Concept
Robots need to understand their environment. They use two types of sensing:

**Proprioception** (internal sensing) — knowing where your own body is
- Joint angles (encoders)
- Joint velocities
- Forces/torques at joints

**Exteroception** (external sensing) — perceiving the world
- RGB cameras → images
- Depth cameras → 3D shape of the scene
- LiDAR → 3D point cloud (sparse but accurate)

```
Observations in robotics:
┌──────────────────────────────┐
│  Proprioception              │
│  - joint_positions: [7]      │  ← "Where are my joints?"
│  - joint_velocities: [7]     │  ← "How fast are they moving?"
│  - gripper_state: [1]        │  ← "Am I gripping?"
├──────────────────────────────┤
│  Exteroception               │
│  - rgb_image: [H, W, 3]     │  ← "What do I see?"
│  - depth_image: [H, W]      │  ← "How far away is everything?"
│  - point_cloud: [N, 3]      │  ← "3D shape of the world"
├──────────────────────────────┤
│  Task Info                   │
│  - goal_position: [3]        │  ← "Where should I go?"
│  - language_instruction: str │  ← "What should I do?"
└──────────────────────────────┘
```

### Reading (45 min)
- [ ] Read sections 1-3 of: [Gymnasium-Robotics docs — Fetch environments](https://robotics.farama.org/envs/fetch/) (15 min)
- [ ] Watch: [3Blue1Brown — "But what is a convolution?"](https://www.youtube.com/watch?v=KuXjwB4LzSA) if you haven't — this is the foundation of robot vision (20 min)
- [ ] Read this short explainer on coordinate frames: [MuJoCo docs — Coordinate Frames](https://mujoco.readthedocs.io/en/stable/overview.html#coordinate-frames) (10 min)

### Hands-On (30 min)
- [ ] Run `uv run examples/02_fetch_robot.py`
- [ ] Look at the printed observation space — what does each part mean?
- [ ] Notice how the scripted controller uses the `desired_goal` and `achieved_goal` — this is goal-conditioned control

### Day 2 Questions
1. What's the difference between `observation`, `desired_goal`, and `achieved_goal` in a GoalEnv?
2. Why do robots need depth information, not just RGB? (Think: can you pick up a cup from a photo?)
3. What shape is the observation space of FetchPickAndPlace? What does each dimension represent?
4. A robot arm has joint encoders that read angles. If joint 3 reads 0.5 radians, what does that physically mean?

---

## Day 3: How Robots Move (Kinematics 101)

### Core Concept
**Kinematics** = the math of motion (ignoring forces).

Two key problems:
- **Forward Kinematics (FK)**: Given joint angles → where is the hand? (Easy, just geometry)
- **Inverse Kinematics (IK)**: Given desired hand position → what joint angles? (Hard, often multiple solutions)

```
Forward Kinematics:
  joint angles [θ1, θ2, θ3] ──→ hand position [x, y, z]
  "I know my joints are at these angles, where is my hand?"

Inverse Kinematics:
  desired hand position [x, y, z] ──→ joint angles [θ1, θ2, θ3]
  "I want my hand HERE, what should my joints be?"
```

### Why This Matters for Physical AI
Every robot policy — whether hand-coded, RL-trained, or a VLA model — ultimately outputs one of:
- **Joint angles** (direct joint control)
- **End-effector pose** (then IK solves for joints)
- **Joint velocities** or **torques**

Understanding this chain is essential.

### Reading (30 min)
- [ ] Read: [MIT Robotic Manipulation — Chapter 1: Introduction](https://manipulation.csail.mit.edu/intro.html) (just the intro, ~20 min)
- [ ] Watch: [Angela Sodemann — "Robot Forward Kinematics"](https://www.youtube.com/watch?v=VjsuBT4Npvk) (10 min YouTube video)

### Hands-On: 2D Forward Kinematics (see `week1/day3_kinematics.py`)
- [ ] Complete the programming exercise in `week1/day3_kinematics.py`
- [ ] This is a simple 2-link planar arm — compute where the tip ends up given joint angles
- [ ] Visualize different joint configurations

### Day 3 Questions
1. A 2-link arm has link lengths L1=1.0 and L2=0.5. Joint 1 is at 90° and joint 2 is at 0°. Where is the tip? (Draw it on paper!)
2. Why is inverse kinematics harder than forward kinematics?
3. Can two different sets of joint angles give the same end-effector position? (Yes/No and why?)
4. What is a "workspace"? Why can't a robot arm reach every point in space?

---

## Day 4: How Robots Decide (Control Basics)

### Core Concept
**Control** = making the robot do what you want, despite physics fighting you.

The simplest and most important controller: **PID**

```
error = desired_position - current_position

control_signal = Kp * error           ← Proportional (spring-like)
               + Ki * ∫error dt       ← Integral (removes steady-state error)
               + Kd * d(error)/dt     ← Derivative (damping, prevents overshoot)
```

PID is everywhere: thermostats, cruise control, drone stabilization, and yes — robot joints.

### Reading (30 min)
- [ ] Watch: [Brian Douglas — "PID Control" (MATLAB)](https://www.youtube.com/watch?v=wkfEjMlAUFA) (15 min, excellent visual explanation)
- [ ] Read: [MIT Robotic Manipulation — Chapter 3: Basic Pick and Place](https://manipulation.csail.mit.edu/pick.html) (skim, 15 min)

### Hands-On: PID Controller in MuJoCo (see `week1/day4_pid_control.py`)
- [ ] Complete the PID controller exercise
- [ ] Tune the gains: what happens with too much Kp? Too little Kd?
- [ ] Try to make the arm reach a specific joint configuration smoothly

### Day 4 Questions
1. What happens if Kp is very large? (Hint: think of a very stiff spring)
2. What does the D term prevent?
3. Why might a P-only controller never reach exactly the target position?
4. In the Fetch example (Day 2), the scripted controller uses `grip_ctrl` proportional to distance. Is that a P controller?

---

## Day 5: How Robots Learn from Demonstrations

### Core Concept
Instead of programming every motion by hand, we can **show** the robot what to do.

**Imitation Learning** = learning a policy from human demonstrations

```
Human demos:                        Learned policy:
(obs₁, action₁)
(obs₂, action₂)     ──train──→     π(observation) → action
(obs₃, action₃)
...                                 "Given what I see, do this"
```

The simplest form: **Behavior Cloning** = supervised learning on (observation, action) pairs.

Recent advances:
- **ACT** (Action Chunking Transformer): Predicts chunks of future actions, not one at a time
- **Diffusion Policy**: Uses diffusion models (like image generators!) to generate action trajectories

### Reading (1 hour)
- [ ] Read: [Lilian Weng — "Learning from Demonstrations"](https://lilianweng.github.io/posts/2018-04-07-imitation-learning/) (30 min — excellent overview)
- [ ] Read the **abstract + introduction** of the Diffusion Policy paper: [arxiv.org/abs/2303.04137](https://arxiv.org/abs/2303.04137) (15 min — just pages 1-3)
- [ ] Read the **abstract + introduction** of the ACT paper: [arxiv.org/abs/2304.13705](https://arxiv.org/abs/2304.13705) (15 min — just pages 1-3)

### Key Insight
This is where YOUR existing skills become superpowers:
- Behavior Cloning is just **supervised learning** (you know this!)
- Diffusion Policy is literally **image diffusion applied to actions** (you know diffusion!)
- ACT uses **transformers** (you definitely know transformers!)

### Day 5 Questions
1. What is the difference between Behavior Cloning and Reinforcement Learning?
2. Why might Behavior Cloning fail at a task where there are two equally good ways to do it? (Hint: averaging)
3. How does Diffusion Policy solve the "averaging" problem?
4. What does "action chunking" mean and why does it help?

---

## Day 6: The Big Picture — Vision-Language-Action Models

### Core Concept
**VLA = VLM + Actions**. Take a Vision-Language Model (like GPT-4V or Claude's vision) and add the ability to output robot actions.

```
┌─────────────┐   ┌──────────────────┐   ┌─────────────┐
│  Camera      │   │  "Pick up the    │   │             │
│  Image       │──→│   red cup"       │──→│  VLA Model  │──→ [actions]
│  [H,W,3]    │   │  (language)      │   │             │   [Δx,Δy,Δz,
└─────────────┘   └──────────────────┘   └─────────────┘    Δrx,Δry,Δrz,
                                                             gripper]
```

This is the **current frontier** of Physical AI. Models like:
- **RT-2** (55B) — Google DeepMind
- **OpenVLA** (7B) — Stanford/Berkeley, open source
- **pi0** (3B) — Physical Intelligence, current SOTA
- **SmolVLA** (450M) — HuggingFace, runs on a laptop

### Reading (45 min)
- [ ] Read the **abstract + intro + section 3** of the OpenVLA paper: [arxiv.org/abs/2406.09246](https://arxiv.org/abs/2406.09246) (20 min)
- [ ] Read this blog post: [HuggingFace — "SmolVLA: A Small Vision-Language-Action Model"](https://huggingface.co/blog/smolvla) (15 min)
- [ ] Revisit the VLA architecture diagram in `PHYSICAL_AI_GUIDE.md` Section 3.1 (10 min)

### Day 6 Questions
1. What are the three inputs to a VLA model?
2. How does OpenVLA represent robot actions? (Hint: discretized tokens)
3. Why is pi0's "flow matching" approach better than tokenized actions for dexterous tasks?
4. What does "cross-embodiment" mean? Why is it exciting?

---

## Day 7: Week 1 Programming Assignment

### Assignment: Build a "Reach Target" Controller from Scratch

See `week1/assignment.py` — a scaffolded exercise where you:

1. Load a robot arm in MuJoCo
2. Implement forward kinematics (compute end-effector position from joint angles)
3. Implement a simple controller that moves the arm toward a colored target cube
4. Track and plot the error over time

This brings together Days 1-4: simulation + perception + kinematics + control.

**Time estimate:** 2-3 hours
**Difficulty:** Beginner-friendly with scaffolding, but requires thinking

---

## Week 1 Comprehensive Quiz

Answer these after completing all 7 days. Write your answers in a notebook — explaining in your own words is the best way to learn.

### Fundamentals (Day 1-2)
1. What is the sense-think-act loop? Draw it.
2. Name 3 types of robot sensors and what information each provides.
3. What is a "degree of freedom"? How many DOF does a typical robot arm have?
4. What's the difference between proprioception and exteroception?

### Kinematics & Control (Day 3-4)
5. Explain forward kinematics vs inverse kinematics in one sentence each.
6. For a 2-link planar arm (L1=1, L2=1): what is the maximum reach? Can it reach a point at distance 3?
7. What are the three terms of a PID controller and what does each do?
8. You set Kp=1000 and Kd=0. What likely happens? Why?

### Learning & AI (Day 5-6)
9. What is Behavior Cloning? What are its limitations?
10. How is Diffusion Policy related to Stable Diffusion? What is different?
11. What does VLA stand for? What are the inputs and outputs?
12. Why is Physical AI harder than chatbot AI? List 3 reasons.

### Synthesis
13. Trace the path from "pick up the red cup" (human instruction) to the robot actually doing it. What are all the steps?
14. Why is simulation important for robot learning? List 2 advantages and 1 limitation.
15. Looking at the evolution RT-1 → RT-2 → Octo → OpenVLA → pi0, what is the key trend?

---

## What's Next (Week 2 Preview)
- Reinforcement Learning basics (reward, policy, value function)
- Training your first RL agent in MuJoCo
- Deeper dive into Diffusion Policy
- Start the LeRobot framework
