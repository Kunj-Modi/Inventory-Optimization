"""
Utility functions for multi-agent training coordination
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
from agents.ppo_agent import PPOAgent

class MultiAgentTrainer:
    """
    Coordinates training of multiple PPO agents
    """

    def __init__(self, agents: Dict[str, PPOAgent], agent_names: List[str]):
        self.agents = agents
        self.agent_names = agent_names

    def act(self, observations: Dict[str, np.ndarray], deterministic: bool = False) -> Dict[str, np.ndarray]:
        """Get actions from all agents"""
        actions = {}
        for name in self.agent_names:
            action, _, _ = self.agents[name].act(observations[name], deterministic)
            actions[name] = action
        return actions

    def store_transitions(self, observations: Dict[str, np.ndarray],
                         actions: Dict[str, np.ndarray],
                         log_probs: Dict[str, np.ndarray],
                         reward: float, values: Dict[str, np.ndarray], done: bool):
        """Store transitions for all agents"""
        for name in self.agent_names:
            self.agents[name].store_transition(
                observations[name], actions[name], log_probs[name], reward, values[name], done
            )

    def update_all(self) -> Dict[str, Dict[str, float]]:
        """Update all agents and return combined statistics"""
        stats = {}
        for name in self.agent_names:
            stats[name] = self.agents[name].update()
        return stats

    def get_values(self, observations: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Get value estimates from all agents"""
        values = {}
        for name in self.agent_names:
            _, _, value = self.agents[name].act(observations[name], deterministic=True)
            values[name] = value
        return values

    def save_checkpoints(self, base_path: str):
        """Save checkpoints for all agents"""
        for name in self.agent_names:
            filepath = f"{base_path}_{name}.pth"
            self.agents[name].save_checkpoint(filepath)

    def load_checkpoints(self, base_path: str):
        """Load checkpoints for all agents"""
        for name in self.agent_names:
            filepath = f"{base_path}_{name}.pth"
            self.agents[name].load_checkpoint(filepath)

def create_agents(env) -> Tuple[Dict[str, PPOAgent], List[str]]:
    """Create PPO agents for the inventory management environment"""
    from models.networks import FGActorCritic, RMActorCritic

    # Create networks
    fg_network = FGActorCritic(
        state_dim=env.observation_space['fg_agent'].shape[0],
        action_dim=env.action_space_fg.shape[0]
    )

    rm_network = RMActorCritic(
        state_dim=env.observation_space['rm_agent'].shape[0],
        action_dim=env.action_space_rm.shape[0]
    )

    # Create agents
    agents = {
        'fg_agent': PPOAgent(fg_network),
        'rm_agent': PPOAgent(rm_network)
    }

    agent_names = ['fg_agent', 'rm_agent']

    return agents, agent_names

print("Multi-agent training utilities created successfully!")
