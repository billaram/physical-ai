# Physical AI: The Complete Guide

> From CV + LLM Agents to Building Physical AI Systems

---

## Table of Contents

1. [What is Physical AI?](#1-what-is-physical-ai)
2. [Key Research Areas](#2-key-research-areas)
3. [SOTA Architectures (2024-2026)](#3-sota-architectures)
4. [Open Source Ecosystem](#4-open-source-ecosystem)
5. [Your Skills Transfer Map](#5-your-skills-transfer-map)
6. [Learning Path (6-Month Plan)](#6-learning-path)
7. [Portfolio Projects](#7-portfolio-projects)
8. [Papers Reading List](#8-papers-reading-list)
9. [Companies & Labs to Follow](#9-companies--labs)

---

## 1. What is Physical AI?

Physical AI = AI systems that **perceive, reason about, and act upon the physical world**.

Unlike chatbots or image generators, Physical AI must contend with gravity, friction, deformable objects, real-time dynamics, and embodiment in hardware.

### How the Industry Defines It

| Player | Definition |
|--------|-----------|
| **NVIDIA** (Jensen Huang) | "Everything that moves will one day be autonomous." Three pillars: World Foundation Models, Simulation Platforms (Omniverse/Isaac), Robot Foundation Models |
| **Google DeepMind** | "Embodied AI" -- foundation models that operate across multiple physical tasks and embodiments |
| **Physical Intelligence** | "General-purpose robot foundation models" -- a single model across robot form factors and tasks |
| **Toyota Research** | "Large Behavior Models" -- learning dexterous manipulation through generative modeling |

### Scope

```
Physical AI
├── Embodied Agents (robots, AVs, drones)
├── World Models (internal physics simulators)
├── Robot Foundation Models (general-purpose pretrained control)
├── Simulation & Digital Twins
├── 3D Perception & Spatial Intelligence
└── Human-Robot Interaction (language-guided control)
```

---

## 2. Key Research Areas

### 2.1 Vision-Language-Action (VLA) Models
The **dominant paradigm** as of 2024-2025. Architecture pattern:
```
[Camera Images] + [Language Instruction] --> [VLM Backbone] --> [Robot Actions]
```

### 2.2 Diffusion Policies for Robot Control
Using denoising diffusion (same family as Stable Diffusion) to generate action trajectories. Handles multi-modal distributions naturally (multiple valid ways to do a task).

### 2.3 World Models
AI systems that predict how the physical world evolves. Applications: planning, data augmentation, reward learning, safety validation.

### 2.4 Sim-to-Real Transfer
Training in simulation (MuJoCo, Isaac Sim) and deploying on real hardware via domain randomization, system identification, and reality gap mitigation.

### 2.5 Cross-Embodiment Learning
A single model controlling different robot hardware -- enabled by the Open X-Embodiment dataset (1M+ episodes across 22 robots).

### 2.6 Dexterous Manipulation
The new frontier: folding laundry, cooking, assembly -- tasks requiring fine motor control and contact-rich interaction.

---

## 3. SOTA Architectures

### 3.1 Robot Foundation Models (Evolution)

```
RT-1 (2022) --> RT-2 (2023) --> Octo (2024) --> OpenVLA (2024) --> pi0 (2024) --> pi0.5 (2025)
  Custom arch     VLM repurposed   Open, modular   Open 7B VLA    Flow matching   + Hierarchical planning
  130K demos      Web knowledge    800K episodes    970K episodes  Dexterous tasks Long-horizon tasks
```

#### RT-2 (Google DeepMind, 2023)
- Fine-tuned VLM (PaLI-X 55B) that outputs actions as text tokens
- **Key insight**: Web-scale vision-language pretraining transfers to robot control
- `arxiv: 2307.15818`

#### Octo (Berkeley, 2024)
- 93M param transformer, modular "readout head" for different action spaces
- Designed as a fine-tunable base model (like Llama for robotics)
- Trained on Open X-Embodiment (800K+ episodes)
- `arxiv: 2405.12213`

#### OpenVLA (Stanford/Berkeley, 2024)
- 7B params (SigLIP + DinoV2 + Llama 2 backbone)
- **Matches RT-2-X (55B) while being 7B.** Fully open-source.
- Actions as discretized tokens (256 bins per dimension)
- `arxiv: 2406.09246`

#### pi0 (Physical Intelligence, 2024) -- Current SOTA
- PaliGemma 3B VLM + **flow matching** action head
- Flow matching > tokenized actions for dexterous tasks
- Demonstrated folding laundry, busing tables, assembling boxes
- `arxiv: 2410.24164`

#### pi0.5 (Physical Intelligence, 2025)
- Added hierarchical planning: high-level VLM planner + pi0 low-level policy
- Follows vague instructions ("clean up the kitchen") by decomposing into subtasks

### Architecture Comparison

| Model | Size | Action Repr. | Data | Open Source? |
|-------|------|-------------|------|-------------|
| RT-2 | 55B | Discretized tokens | Google internal | No |
| Octo | 93M | Diffusion head | Open X-Embodiment | Yes (MIT) |
| OpenVLA | 7B | Discretized tokens | Open X-Embodiment | Yes (Apache 2.0) |
| pi0 | ~3B | Flow matching | Multi-platform | No |
| HPT | Varies | Embodiment-specific heads | Heterogeneous | Yes |
| RDT-1B | 1.2B | Diffusion transformer | Multi-robot | Yes |

### 3.2 World Models

| Model | From | Architecture | Application |
|-------|------|-------------|-------------|
| **Cosmos** (2025) | NVIDIA | Diffusion + Autoregressive (4B-14B) | General Physical AI |
| **GAIA-1** (2023) | Wayve | Autoregressive transformer | Autonomous driving |
| **UniSim** (2023) | Google DeepMind | Video generation conditioned on actions | Universal simulation |
| **Genie 2** (2024) | Google DeepMind | Spatiotemporal transformer | 3D world generation |
| **World Labs** (2024) | Fei-Fei Li | Large World Models | Spatial intelligence |

### 3.3 Diffusion Policy Deep Dive

```python
# Conceptual Diffusion Policy Pipeline
# Input:  observation (image + proprioception)
# Output: chunk of future actions (e.g., next 16 timesteps)

# 1. Encode observation
obs_embedding = vision_encoder(image) + proprio_encoder(joint_states)

# 2. Start with random noise actions
action_trajectory = torch.randn(horizon, action_dim)  # e.g., (16, 7)

# 3. Iteratively denoise (DDPM/DDIM)
for t in reversed(range(num_diffusion_steps)):
    noise_pred = denoiser(action_trajectory, t, obs_embedding)
    action_trajectory = denoise_step(action_trajectory, noise_pred, t)

# 4. Execute first few actions, then replan
robot.execute(action_trajectory[:execute_horizon])
```

**Key design decisions in Diffusion Policy:**
- **CNN-based** (1D temporal U-Net) vs **Transformer-based** (DiT-style) denoiser
- **Action chunking**: predict 16 steps, execute 8, replan (temporal consistency)
- **DDIM** for faster inference (~10 steps vs 100 for DDPM)

---

## 4. Open Source Ecosystem

### 4.1 Foundation Models & Policies

| Repository | Stars | What It Does | License |
|-----------|-------|-------------|---------|
| [huggingface/lerobot](https://github.com/huggingface/lerobot) | 8K-10K+ | End-to-end robot learning platform (models + data + hardware) | Apache 2.0 |
| [openvla/openvla](https://github.com/openvla/openvla) | 2K+ | 7B Vision-Language-Action model | Apache 2.0 |
| [moojink/openvla-oft](https://github.com/moojink/openvla-oft) | 500+ | OpenVLA v2 with 26x faster inference | Apache 2.0 |
| [octo-models/octo](https://github.com/octo-models/octo) | 1K+ | 93M generalist robot policy (JAX) | MIT |
| [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) | 1.5K+ | Diffusion models for visuomotor policy | MIT |
| [tonyzhaozh/act](https://github.com/tonyzhaozh/act) | 2K+ | Action Chunking Transformer (ALOHA) | MIT |
| [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab) | 2.5K+ | GPU-accelerated robot learning framework | BSD-3 |
| [YanjieZe/3D-Diffusion-Policy](https://github.com/YanjieZe/3D-Diffusion-Policy) | 500+ | 3D point cloud + diffusion policy | -- |

### 4.2 Simulation Environments

| Simulator | Best For | GPU Parallel? | Activity |
|-----------|---------|--------------|----------|
| **[MuJoCo](https://github.com/google-deepmind/mujoco)** (8K+ stars) | General robotics RL, research standard | Yes (MJX/JAX) | Very High |
| **[Isaac Lab](https://github.com/isaac-sim/IsaacLab)** | Production-scale RL, locomotion | Native | High |
| **[ManiSkill 3](https://github.com/haosulab/ManiSkill)** (1K+) | Manipulation research | Yes | High |
| **[SAPIEN](https://github.com/haosulab/SAPIEN)** | Articulated objects | Yes | High |
| **[Habitat](https://github.com/facebookresearch/habitat-sim)** (2.5K+) | Indoor navigation | Moderate | Moderate |
| **[Genesis](https://github.com/Genesis-Embodied-AI/Genesis)** | Multi-physics (new!) | Yes | Growing fast |

### 4.3 Datasets

| Dataset | Scale | Focus | Access |
|---------|-------|-------|--------|
| **Open X-Embodiment** | 1M+ episodes, 22 robots | Multi-robot, multi-task | [GitHub](https://github.com/google-deepmind/open_x_embodiment) |
| **DROID** | 76K trajectories, 350+ scenes | Franka manipulation | [GitHub](https://github.com/droid-dataset/droid) |
| **Bridge Data V2** | 60K trajectories | WidowX manipulation | Berkeley |
| **RH20T** | 110K+ episodes | Multi-robot | Tsinghua |
| **HuggingFace LeRobot Hub** | Growing rapidly | Standardized format | [HF Hub](https://huggingface.co/lerobot) |

### 4.4 Hardware (Low-Cost to Pro)

| Platform | Cost | DOF | Notes |
|----------|------|-----|-------|
| **SO-100** | ~$110/arm | 6 | 3D-printed, LeRobot-native, lowest entry point |
| **Koch v1.1** | ~$200-300 | 6 | 3D-printed, LeRobot-compatible |
| **UMI Gripper** | ~$300 | - | Portable data collection (hand-held, transfers to any arm) |
| **ALOHA** | ~$20K | 2x7 DOF | Bimanual, professional quality |
| **WidowX 250** | ~$3,500 | 6 | Used in Bridge Data |
| **Franka Emika** | ~$30K+ | 7 | Research standard |

---

## 5. Your Skills Transfer Map

### What You Already Have

```
YOUR SKILLS                          PHYSICAL AI APPLICATION
─────────────────────────────────────────────────────────────
Python + PyTorch           ──────>   Same stack (PyTorch, JAX)
2D Object Detection        ──────>   Object localization for grasping
Image Segmentation         ──────>   Scene understanding for manipulation
Depth Estimation           ──────>   3D perception (+ point clouds)
Image Generation (Diffusion) ────>   Diffusion Policy for actions
NeRF / 3DGS               ──────>   Scene representation for planning
LLM Tool Calling           ──────>   Robot skill invocation (SayCan)
LLM Code Generation        ──────>   Code-as-Policies
Chain-of-thought           ──────>   Multi-step task planning
Agent Memory               ──────>   Experience replay, context
Multi-modal VLMs           ──────>   VLA models (the core of Physical AI)
```

### What You Need to Add

**Tier 1 -- Essential (weeks 1-8)**
1. **Robot Kinematics**: Forward/inverse kinematics, Jacobians (3-4 weeks)
2. **Control Theory**: PID, trajectory tracking, impedance control (3-4 weeks)
3. **3D Vision**: SE(3) transforms, quaternions, point clouds, hand-eye calibration (2-3 weeks)

**Tier 2 -- Core Robot Learning (weeks 8-16)**
4. **RL for Robotics**: PPO, SAC, reward shaping, sim-to-real (4-6 weeks)
5. **Imitation Learning**: Behavior cloning, Diffusion Policy, ACT (3-4 weeks)

**Tier 3 -- Advanced (weeks 16-24)**
6. **Foundation Models for Robotics**: VLA fine-tuning, Octo, OpenVLA (2-3 weeks)
7. **Sim-to-Real**: Domain randomization, system identification (3-4 weeks)

### CV Engineer Transfer Checklist

| CV Skill | Robot Vision Application | Gap to Fill |
|----------|------------------------|-------------|
| Object detection/segmentation | Object localization for grasping | SE(3) pose, depth integration |
| Optical flow | Motion estimation for tracking | Predictive control integration |
| Monocular depth | 3D perception | Metric accuracy, point cloud processing |
| Diffusion models (images) | Diffusion Policy (actions) | Action space vs. pixel space |
| NeRF / 3DGS | Scene representation for planning | Grasp planning, physics reasoning |

### LLM Agent Transfer Checklist

| LLM Agent Skill | Robot Application | Gap to Fill |
|-----------------|-------------------|-------------|
| Tool calling / function calling | Robot skill invocation | Physical affordance grounding |
| Code generation | Code-as-Policies | Robot API knowledge, spatial math |
| Chain-of-thought | Multi-step task planning | Physical feasibility checking |
| RAG / retrieval | Skill library lookup | Robot capability representation |
| Multi-modal (vision + language) | VLA models | Action output heads, embodiment |

---

## 6. Learning Path (6-Month Plan)

### Month 1: Foundations
- [ ] **MIT 6.4210 Robotic Manipulation** (Russ Tedrake) -- https://manipulation.csail.mit.edu
  - THE single best resource. Free, modern, ML-oriented. Chapters 1-5.
- [ ] **Modern Robotics Coursera** (Northwestern/Kevin Lynch) -- Modules 1-3 (kinematics)
- [ ] Set up MuJoCo and run basic simulations
- [ ] Read: SayCan paper, Code-as-Policies paper

### Month 2: 3D Vision + Control
- [ ] Continue Robotic Manipulation course -- Chapters 6-10
- [ ] Implement basic pick-and-place in MuJoCo
- [ ] Open3D tutorials with depth data
- [ ] Study SE(3) transforms, quaternions, hand-eye calibration
- [ ] **Start Project**: LLM Task Planner in simulation

### Month 3: Reinforcement Learning
- [ ] **UC Berkeley CS 285** (Sergey Levine) -- YouTube lectures (first half)
- [ ] Train PPO/SAC for manipulation in Gymnasium-Robotics
- [ ] Domain randomization experiments
- [ ] Read: Diffusion Policy paper, ACT paper

### Month 4: Imitation Learning + Foundation Models
- [ ] **Reproduce Diffusion Policy** results in simulation
- [ ] Study ACT architecture deeply
- [ ] Read: RT-2, Octo, OpenVLA, pi0 papers
- [ ] **Start Project**: OpenVLA fine-tuning on custom task

### Month 5: Hardware + Real-World
- [ ] Order and assemble **SO-100** or **Koch v1.1** arm (~$110-300)
- [ ] Set up **LeRobot** framework
- [ ] Collect teleoperation demonstrations
- [ ] Train ACT/Diffusion Policy on real robot data
- [ ] Begin contributing to LeRobot open source

### Month 6: Integration + Portfolio
- [ ] Build VLM + robot arm demo (language-guided manipulation)
- [ ] Polish all portfolio projects with documentation and videos
- [ ] Write blog posts about learnings
- [ ] Make open-source contributions
- [ ] Apply to Physical AI roles

### Key Resources

| Resource | Type | URL |
|----------|------|-----|
| MIT Robotic Manipulation | Course (free) | https://manipulation.csail.mit.edu |
| MIT Underactuated Robotics | Course (free) | https://underactuated.csail.mit.edu |
| UC Berkeley CS 285 | Lectures (YouTube) | Search "CS 285 Sergey Levine" |
| Modern Robotics | Coursera + free textbook | Coursera specialization |
| Lilian Weng's Blog | Technical posts | https://lilianweng.github.io |
| LeRobot Docs | Getting started | https://github.com/huggingface/lerobot |

---

## 7. Portfolio Projects

### Tier 1: Simulation Only ($0)

**Project 1: Diffusion Policy for Simulated Manipulation**
```
Tools: MuJoCo + diffusion_policy repo + robomimic
Skills: Policy learning, diffusion models, simulation
Steps: Clone repo -> reproduce results -> extend to new task
```

**Project 2: LLM Task Planner with Simulated Robot**
```
Tools: Claude/GPT API + PyBullet/MuJoCo
Skills: LLM grounding, code-as-policies, closed-loop planning
Steps: Setup tabletop env -> define primitives -> build LLM agent
```

**Project 3: NeRF/3DGS Scene Understanding for Grasp Planning**
```
Tools: Nerfstudio or gsplat + GraspNet + Open3D
Skills: 3D reconstruction, grasp planning, perception pipeline
```

**Project 4: VLA Fine-tuning on Simulation Data**
```
Tools: OpenVLA + LIBERO or RLBench benchmark
Skills: Foundation model fine-tuning, evaluation, embodied AI
```

### Tier 2: Low-Cost Hardware ($100-500)

**Project 5: LeRobot + SO-100 Arms** (RECOMMENDED FIRST HARDWARE PROJECT)
```
Hardware: SO-100 arm kit (~$110) or pair (~$220) + USB webcam
Tools: LeRobot framework
Skills: Real robot data collection, imitation learning, system integration
```

**Project 6: Visual Language Control with Low-Cost Arm**
```
Hardware: SO-100 + webcam
Tools: LeRobot + VLM (Claude/GPT-4V) + custom integration
Skills: Full-stack Physical AI: perception -> planning -> control
```

### Tier 3: Open Source Contributions

| Project | Why Contribute | Opportunities |
|---------|---------------|--------------|
| **LeRobot** | Most active community, fastest growing | New hardware support, new policies, documentation |
| **OpenVLA** | Core VLA model | Fine-tuning recipes, benchmarks, efficiency |
| **ManiSkill** | Major manipulation benchmark | New environments, integration with new methods |
| **Gymnasium-Robotics** | Standard RL benchmarks | New environments, bug fixes |

---

## 8. Papers Reading List

### Recommended Reading Order

| # | Paper | Year | ArXiv | Why Read It |
|---|-------|------|-------|-------------|
| 1 | Diffusion Policy | 2023 | 2303.04137 | The action generation paradigm |
| 2 | RT-2: Vision-Language-Action Models | 2023 | 2307.15818 | How VLMs became VLAs |
| 3 | SayCan | 2022 | 2204.01691 | LLM grounding with affordances |
| 4 | Code as Policies | 2023 | 2209.07753 | LLMs writing robot control code |
| 5 | Open X-Embodiment / RT-X | 2023 | 2310.08864 | The data landscape |
| 6 | ACT / ALOHA | 2023 | 2304.13705 | Action Chunking Transformer |
| 7 | Octo | 2024 | 2405.12213 | Open-source fine-tunable foundation model |
| 8 | OpenVLA | 2024 | 2406.09246 | Open-source VLA you can use today |
| 9 | pi0 | 2024 | 2410.24164 | Current SOTA, flow matching |
| 10 | NVIDIA Cosmos | 2025 | 2501.03575 | World foundation models |
| 11 | Genie | 2024 | 2402.15391 | Generative world models |
| 12 | HPT | 2024 | 2409.20537 | Heterogeneous robot data handling |

### Supplementary Papers

| Paper | ArXiv | Topic |
|-------|-------|-------|
| RT-1 | 2212.06817 | First Robotics Transformer |
| GAIA-1 | 2309.17080 | World model for autonomous driving |
| UniSim | 2310.06114 | Universal simulator from video |
| Isaac Gym | 2108.10470 | GPU-accelerated physics sim |
| MimicGen | 2310.17596 | Automated demo generation |
| RoboCasa | 2406.02523 | Large-scale household sim |
| 3D Diffusion Policy | 2403.03954 | 3D point cloud + diffusion |
| RDT-1B | 2410.07864 | Scaling diffusion transformers |
| LeCun "Path Towards AMI" | -- | World models are essential (position paper) |

---

## 9. Companies & Labs

### Tier 1: Defining the Field

| Company/Lab | Focus | Key Models/Products |
|------------|-------|-------------------|
| **NVIDIA** | Full-stack Physical AI platform | Cosmos, Isaac Sim/Lab, GR00T, Omniverse |
| **Google DeepMind** | Robot foundation models | RT-1/2/X, ALOHA, UniSim, Genie |
| **Physical Intelligence** | General-purpose robot brain | pi0, pi0.5 |
| **Toyota Research (TRI)** | Dexterous manipulation | Diffusion Policy, Large Behavior Models |

### Tier 2: Major Contributors

| Company/Lab | Focus | Notable |
|------------|-------|---------|
| **Stanford** (IRIS, SVL) | OpenVLA, ALOHA, RoboCasa | Chelsea Finn, Fei-Fei Li |
| **UC Berkeley** (BAIR) | Octo, OpenVLA, Bridge Data | Sergey Levine, Pieter Abbeel |
| **MIT** (CSAIL) | HPT, manipulation | Pulkit Agrawal |
| **Hugging Face** | LeRobot platform | Democratizing robot learning |
| **World Labs** | Large World Models | Fei-Fei Li, spatial intelligence |

### Tier 3: Humanoid & Industry

| Company | Product | Valuation/Funding |
|---------|---------|------------------|
| **Tesla** (Optimus) | Humanoid robot | Tesla market cap |
| **Figure AI** | Figure 01/02 humanoid | $2.6B+ |
| **1X Technologies** | NEO humanoid | OpenAI-backed |
| **Skild AI** | Robot foundation model | $1.5B |
| **Unitree** | Low-cost humanoids (G1, H1) | Making hardware affordable |
| **Boston Dynamics** | Atlas (Electric) | Hyundai-owned |
| **Agility Robotics** | Digit (logistics humanoid) | Amazon warehouse pilots |
| **Wayve** | GAIA-1 (AV world model) | $1B+ from SoftBank |
| **Amazon/Covariant** | Warehouse manipulation | Acquired 2025 |

### Key Conferences

- **CoRL** -- Conference on Robot Learning (top venue)
- **RSS** -- Robotics: Science and Systems
- **ICRA** -- IEEE International Conference on Robotics and Automation (largest)
- **Embodied AI Workshop @ CVPR** -- Bridges CV and robotics

### Communities

- **LeRobot Discord** (Hugging Face) -- Most active open community
- **r/reinforcementlearning** -- RL discussions
- **Physical Intelligence blog** -- Updates on pi0
- **NVIDIA Developer blog** -- Isaac, Cosmos updates

---

## Quick Start Checklist

If you want to get hands-on TODAY:

1. **Star and clone** [huggingface/lerobot](https://github.com/huggingface/lerobot)
2. **Run a simulation** example from LeRobot to see a trained policy
3. **Read** the Diffusion Policy paper (arxiv: 2303.04137)
4. **Start** MIT 6.4210 (https://manipulation.csail.mit.edu)
5. **Order** SO-100 arm parts (~$110) for Month 5

---

*Guide compiled March 2026. The Physical AI field moves extremely fast -- verify latest developments on arxiv (cs.RO), LeRobot GitHub, and conference proceedings (CoRL, RSS, ICRA).*
