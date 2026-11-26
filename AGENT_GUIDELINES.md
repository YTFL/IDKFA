**Domain:** Deep Reinforcement Learning, Spatial Intelligence & Autonomous Control  
**Environment:** DOOM via ViZDoom Engine (`vizdoom==1.3.3`, Freedoom 2 IWAD, Skill 3)

---

## 1. Mission Brief

Design and train an autonomous RL agent capable of playing unseen DOOM levels **zero-shot**—surviving demon ambushes, navigating complex multi-room layouts, handling doors and keycards, and reaching the level exit.

- **No Static Dataset:** The agent must learn by interacting directly with the game engine.
- **Sensory Guidance:** Avoid relying purely on raw RGB pixels. The official `doom_env` wrapper provides fast, ego-centric spatial telemetry at every step (running at thousands of ticks/sec on CPU):
  - **64 ego-centric raycasts** with hit types (`wall`, `door`, `locked door`, `enemy`, `item`)
  - **Depth buffer**
  - **Vitals vector** (health, armor, ammo, keycards, velocity)
  - **Visible entities list** in relative coordinates
  - **Low-resolution RGB frame**
  - *Note:* No global map coordinates are provided. Any subset or hybrid combination of these signals is allowed.

---

## 2. Core Challenge Objectives

1. **Partial Observability:** Navigate 3D spaces without a god-view radar, relying strictly on current FOV and local sensory range.
2. **Long-Horizon Progression:** Solve multi-step room dependencies (explore branches $\rightarrow$ find Keycard $\rightarrow$ remember Locked Door location $\rightarrow$ backtrack $\rightarrow$ unlock and reach Exit).
3. **Tactical Combat & Survival:** Manage health, ammo, and momentum while dodging incoming projectiles and prioritizing multiple threats.
4. **Zero-Shot Generalization:** Perform on unseen map layouts without overfitting to a single scenario.

---

## 3. System Architecture & Deliverables

### Required Engineering Components
- **Sensory & State Representation:** Custom encoders and preprocessing on top of the fixed official observation dictionary (identical in training and evaluation).
- **Model Architecture & Memory:** Custom neural network topology addressing spatial memory across rooms and variable numbers of enemies.
- **Training & Exploration Pipeline:** RL implementation, custom reward shaping, and exploration mechanisms to prevent corner-trapping and drive novel room discovery.

> **Innovation Note:** Off-the-shelf, plug-and-play Stable-Baselines3 PPO scripts with default hyperparameters will score poorly. Custom-engineered architectures (bespoke neural modules, custom spatial memory, intrinsic curiosity, or hierarchical control policies) are heavily rewarded.

### Training Maps
- Only **one small test map** is provided in the starter template (not used in evaluation).
- Sourcing training maps is part of the challenge: use classic DOOM/Freedoom levels, community WADs, procedural level generators, or custom maps. The wrapper loads any Doom-format or UDMF WAD.

---

## 4. Submission Requirements

| Requirement | Specification |
| :--- | :--- |
| **Repository** | Public GitHub repo created from `devforge-doom-template` during the hackathon. |
| **Size Limit** | $\le 1\text{ GB}$ total (including Git LFS). Store model weights via **Git LFS**. |
| **Entry Point** | `submission/agent.py` containing stateful class `DoomAgent`. |
| **Dependencies** | **No Dockerfile.** Runs in the official base image (Python 3.13, `vizdoom==1.3.3`, `gymnasium`, `torch`, `jax`/`flax`, `numpy`, `scipy`, `opencv`, common ML utils). Optional small `submission/requirements.txt` with pinned versions for extra packages. |
| **Validation** | Must pass `python validate.py` (checks repo size, interface, full dry-run episode, and latency). |
| **Documentation** | **2-Page Project Brief (PDF)** summarizing USP, neural architecture, spatial memory design, and training/exploration pipeline. |
| **Training Code** | Must be kept in the repository for reviewer/judge inspection. |

### `DoomAgent` Interface (`submission/agent.py`)
```python
class DoomAgent:
    def __init__(self, observation_space, action_space, device):
        """Load model weights once."""
        pass

    def reset(self):
        """Clear episodic/spatial memory before every map."""
        pass

    def act(self, observation):
        """Return an action. Called once every 4 game ticks."""
        pass