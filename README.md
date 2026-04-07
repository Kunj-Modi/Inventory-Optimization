# Cooperative Multi-Agent Reinforcement Learning for Two-Echelon Inventory Management

This repository implements a cooperative multi-agent reinforcement learning (MARL) framework for optimizing a two-echelon, multi-product inventory system under stochastic demand and lead times.

The approach combines behavioral cloning for policy initialization, Proximal Policy Optimization (PPO) for learning, and a structure-informed monotonicity regularization term to enforce domain-specific constraints.

---

## Overview

The system models a supply chain with two interacting agents:

- Finished Goods (FG) Agent: Responsible for production planning of multiple products
- Raw Material (RM) Agent: Responsible for procurement of raw materials

Both agents operate under a shared objective of minimizing total system cost, which includes holding costs and shortage penalties.

---

## Key Features

- Cooperative multi-agent formulation using a shared reward signal
- Two-stage training pipeline:
  - Behavioral Cloning (BC) from heuristic policy
  - PPO-based reinforcement learning
- Monotonicity regularization to enforce economically consistent policies
- Continuous action spaces for both agents
- Simulation-based environment with stochastic demand and lead times

---

## Repository Structure
```
.
│
├─── inventory_optimization.ipynb
├─── main_training.py
├─── README.md
├─── requirements.txt
├─── .gitignore
│
├─── agents
│     └── ppo_agent.py
│
├─── configs
│     └── config.py
│
├─── data
├─── environment
│     └── inventory_env.py
│
├─── models
│     └── networks.py
│
├─── results
│     ├── Average finished goods inventory comparison.png
│     ├── Cumulative system cost - Heuristic policy.png
│     ├── Cumulative system cost - Trained policy.png
│     ├── Daily system cost comparison.png
│     ├── Finished goods inventory - Heuristic policy.png
│     ├── Finished goods inventory - Trained policy.png
│     ├── Lost sales - Heuristic policy.png
│     ├── Lost sales - Trained policy.png
│     ├── Lost sales comparison.png
│     ├── Multi-agent training progress.png
│     ├── Raw material inventory - Heuristic policy.png
│     ├── Raw material inventory - Trained policy.png
│     ├── Raw material inventory level comparison.png
│     ├── Total system cost - Heuristic policy.png
│     └── Total system cost - Trained policy.png
│
├─── scripts
└─── utils
      ├── evaluation.py
      ├── helpers.py
      ├── heuristic_policy.py
      ├── training_utils.py
      └── visualization.py

```

---

## Methodology

### Problem Setup

- Two-echelon system:
  - 5 finished goods
  - 1 shared raw material
- Stochastic demand modeled using a normal distribution
- Lead times:
  - Finished goods: deterministic
  - Raw material: stochastic

---

### Agent Design

| Agent | Responsibility | Action Space |
|------|---------------|-------------|
| FG Agent | Production planning | Continuous vector (size 5) |
| RM Agent | Procurement | Continuous scalar |

- Actions are bounded and continuous
- Execution order: FG agent acts first, RM agent follows

---

### Training Pipeline

#### Stage 1: Behavioral Cloning

- Train actor networks using data generated from a heuristic base-stock policy
- Loss function: Mean Squared Error (MSE)
- Purpose: stabilize training and reduce exploration overhead

#### Stage 2: PPO Fine-Tuning

- Standard PPO with clipped objective
- Advantage estimation using GAE
- Shared reward based on total system cost

#### Monotonicity Regularization

A regularization term is added to enforce:

d(action) / d(demand) >= 0

This ensures that learned policies follow basic economic intuition.

---

### Model Architecture

- Multi-layer perceptron (MLP):
  - Hidden layers: 256 → 128 → 64
  - Activation: ReLU
- Separate actor and critic heads
- Gaussian policy for continuous actions
- Optimizer: Adam

---

## Results Summary

| Metric | Heuristic Policy | Trained Policy | Change |
|--------|----------------|---------------|--------|
| Total Cost | 53,264 | 50,418 | -5.3% |
| Avg Daily Cost | 88.77 | 84.03 | -5.3% |
| Service Level | 100% | 98.2% | -1.8% |
| Lost Sales | 0 | Small | Trade-off |

The trained policy reduces costs by maintaining lower inventory levels while allowing minimal controlled lost sales.

---

## Outputs

The `results/` directory contains:

- Inventory trajectories (FG and RM)
- Daily and cumulative cost comparisons
- Lost sales analysis
- Training curves showing convergence behavior

---

## Installation

Install required dependencies:

```bash
pip install numpy pandas matplotlib torch
```

## Usage

The entire pipeline (environment simulation, training, and evaluation) is implemented in a single notebook.

To run the project:

```bash
jupyter notebook inventory_optimization.ipynb
```

### Execution order:

1. Initialize environment and parameters
2. Generate heuristic dataset (behavioral cloning)
3. Pre-train policies using behavioral cloning
4. Train agents using PPO
5. Run evaluation rollouts
6. Generate plots in the results/ directory

Note:
- Training is compute-intensive and may take significant time depending on hardware
- GPU acceleration is recommended but not strictly required

## Reproducibility
- The environment is fully simulation-based and does not rely on external datasets
- Demand and lead times are stochastic, so results may vary across runs
- For consistent results:
    - Fix random seeds in NumPy and PyTorch
    - Keep hyperparameters unchanged
    - Use identical rollout and evaluation settings

Evaluation protocol:
- Multiple independent rollouts
- Long-horizon simulations to assess stability
- Aggregate metrics (cost, service level, inventory trends)

## Limitations
- The environment assumes stationary demand distributions and does not capture regime shifts
- Results are based entirely on simulation; real-world performance is not validated
- The two-echelon setup is simplified and does not include complex supply chain constraints (capacity limits, multi-supplier dynamics, etc.)
- Coordination is enforced via shared rewards; no explicit communication mechanism between agents
- Training with PPO in a multi-agent setting can be sample-inefficient and computationally expensive

## Future Work
- Extend the framework to non-stationary and regime-switching demand environments
- Scale to multi-echelon, multi-supplier supply chain networks
- Incorporate real-world constraints such as production capacity and transportation delays
- Explore alternative MARL methods (e.g., CTDE-based approaches, value decomposition)
- Improve sample efficiency using offline RL or model-based methods
- Integrate with real-world data sources and ERP systems
- Investigate sim-to-real transfer techniques for deployment

## Authors
- Kunj Modi
- Hitansh Mehta
- Vidit Thakkar
- Darshana Sankhe
- Paresh Nasikkar
- Ankita Gupta
- Deepali Patil
- Pratik Kanani

## License
This project is licensed under the MIT License. See the LICENSE file for details.