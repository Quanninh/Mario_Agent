# 🍄 Reinforcement Learning Mario Agent: Architecture & Workflow

> **Course**: Machine Learning — Project 5: Reinforcement Learning-Based Game Agent  
> **Core Approach**: Tabular Q-Learning on a Custom 2D Pygame Platformer  
> **Design Pattern**: Clean, modular SOLID architecture  
> **Team Size**: 4 Members  

---

## 1. Project Overview & Problem Formulation

### 1.1 Project Objective
The goal of this project is to develop an autonomous AI agent that learns to play a 2D Mario-style platformer game through **interaction and experience**, rather than manually programmed rules or heuristics.

Initially, the agent knows nothing about the game mechanics and acts randomly. Through hundreds of gameplay episodes, the agent receives numerical feedback (rewards for moving right and reaching the goal, penalties for falling into pits or hitting hazards) and gradually learns an optimal navigation policy.

### 1.2 Core Problem Formulation (Markov Decision Process)
The game is modeled as a standard Reinforcement Learning framework:
- **Agent**: Mario, who observes the environment and selects actions.
- **Environment**: A 2D platformer containing ground, obstacles (pipes/blocks), pits, and a finish line (flagpole).
- **State ($S$)**: Discretized representations of Mario's position, altitude, velocities, and distance to nearby hazards.
- **Action ($A$)**: Discrete decisions (`Stay`, `Right`, `Left`, `Jump`, `Right + Jump`).
- **Reward ($R$)**: Feedback signal reinforcing forward progress and penalizing death or stalling.
- **Algorithm**: **Tabular Q-Learning** with a Q-Table storing $Q(s, a)$ values updated via Temporal Difference (TD) learning, guided by an $\epsilon$-greedy exploration policy.

### 1.3 Key Deliverables
1. **Playable Game Environment**: A functional 2D Pygame platformer with keyboard controls for humans and an API for the AI.
2. **Trained AI Agent**: A Q-learning agent capable of consistently reaching the destination.
3. **Training Telemetry & Graphs**: Learning curves plotting rewards, success rates, and steps per episode.
4. **Technical Report**: Analysis comparing different reward formulations, exploration rates, and generalization across level layouts.

---

## 2. System Activity Diagram

The end-to-end lifecycle follows the standard Reinforcement Learning training and evaluation workflow:

![Activity Diagram: Reinforcement Learning Training Lifecycle](docs/activity_diagram.svg)

---

## 3. Core Workflow

The system executes across 4 high-level phases during each step and episode:

1. **Environment Step**:
   - Executes the selected action (`Stay`, `Right`, `Left`, `Jump`, `Right + Jump`).
   - Updates player physics, checks bounding-box collisions, and emits game transition events.

2. **State & Reward Processing**:
   - **Discretizer**: Converts continuous coordinates and obstacle distances into a discrete state tuple $s$.
   - **Reward Strategy**: Computes scalar reward $r$ based on forward progress, death penalties, and goal bonuses.

3. **Q-Learning Update**:
   - Updates the Q-Table using the Bellman Temporal Difference equation:
     $$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$
   - Decays exploration rate $\epsilon$ after each episode.

4. **Evaluation & Visualization**:
   - Runs the trained agent greedily ($\epsilon = 0$) across different level configurations.
   - Records success rates, average rewards, and plots training curves.

---

## 4. High-Level Project Structure

```text
Mario_Agent/
├── core/            # Base interfaces (Environment, Agent, Discretizer, Reward)
├── env/             # Pygame 2D Mario environment & level layouts
├── discretizer/     # Continuous-to-discrete state mapping
├── rewards/         # Pluggable reward strategies (Sparse, Dense, Balanced)
├── agent/           # Q-Learning agent (Q-Table & Epsilon-Greedy policy)
├── training/        # Training pipeline & hyperparameter configuration
├── evaluation/      # Metrics logger, Matplotlib plotter, and visual playback
├── main.py          # Training entry point
└── play_human.py    # Playable game entry point (manual keyboard control)
```

---

## 5. Team Responsibilities (4 Members)

| Member | Focus Area | Primary Responsibilities |
|:---|:---|:---|
| **Member 1** | **Environment & Physics** | Pygame 2D platformer engine, jump/gravity physics, collision detection, and level layouts. |
| **Member 2** | **Agent & State Discretizer** | Q-Table implementation, Bellman TD update, continuous-to-discrete state quantizer, and $\epsilon$-greedy policy. |
| **Member 3** | **Reward & Training** | Modular reward functions, training orchestration loop, and hyperparameter tuning ($\alpha, \gamma, \epsilon$). |
| **Member 4** | **Evaluation & Report** | Metrics logging, Matplotlib learning curves, visual playback demo, and technical report. |
