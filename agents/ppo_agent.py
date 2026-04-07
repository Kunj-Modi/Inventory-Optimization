"""
PPO agent with IMPROVED monotonicity regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
import configs.config as config

class PPOMemory:
    """Experience replay buffer"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.ptr = 0

    def store(self, state: np.ndarray, action: np.ndarray, log_prob: np.ndarray,
              reward: dict, value: float, done: bool):
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32).flatten())
        self.log_probs.append(np.asarray(log_prob, dtype=np.float32).flatten())
        self.rewards.append(reward)
        self.values.append(float(value))
        self.dones.append(bool(done))
        self.ptr += 1

    def get_batches(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        if len(self.states) == 0:
            return [], [], [], [], [], []

        states = np.array(self.states)
        max_action_dim = max(action.shape[0] for action in self.actions) if self.actions else 1
        max_log_prob_dim = max(log_prob.shape[0] for log_prob in self.log_probs) if self.log_probs else 1

        actions_array = []
        for action in self.actions:
            padded_action = np.zeros(max_action_dim, dtype=np.float32)
            padded_action[:action.shape[0]] = action
            actions_array.append(padded_action)

        log_probs_array = []
        for log_prob in self.log_probs:
            padded_log_prob = np.zeros(max_log_prob_dim, dtype=np.float32)
            padded_log_prob[:log_prob.shape[0]] = log_prob
            log_probs_array.append(padded_log_prob)

        actions = np.array(actions_array)
        log_probs_old = np.array(log_probs_array)
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)

        advantages, returns = self._compute_advantages(rewards, values, dones)

        if len(advantages) > 1:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        batch_start = np.arange(0, len(states), batch_size)
        indices = np.arange(len(states), dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+batch_size] for i in batch_start]

        return states, actions, log_probs_old, advantages, returns, batches

    def _compute_advantages(self, rewards: np.ndarray, values: np.ndarray,
                          dones: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cfg = config.config.ppo
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        last_advantage = 0
        next_value = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t + 1]

            delta = rewards[t] + cfg.GAMMA * next_values * next_non_terminal - values[t]
            advantages[t] = delta + cfg.GAMMA * cfg.LAMBDA_GAE * next_non_terminal * last_advantage
            last_advantage = advantages[t]
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.ptr = 0

    def __len__(self):
        return len(self.states)

class PPOAgent:
    """
    IMPROVED VERSION: PPO agent with robust monotonicity regularization
    """

    def __init__(self, actor_critic_network: nn.Module, agent_type: str = "fg_agent", config_obj=None):
        self.cfg = config_obj if config_obj else config.config
        self.ppo_cfg = self.cfg.ppo
        self.model_cfg = self.cfg.model
        self.agent_type = agent_type

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cfg.colab.USE_GPU else "cpu")

        # Network
        self.actor_critic = actor_critic_network.to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(
            list(self.actor_critic.actor_mean.parameters()) +
            list(self.actor_critic.actor_log_std.parameters()),
            lr=self.model_cfg.LEARNING_RATE_ACTOR,
            eps=self.model_cfg.OPTIMIZER_EPS
        )
        self.critic_optimizer = optim.Adam(
            self.actor_critic.critic.parameters(),
            lr=self.model_cfg.LEARNING_RATE_CRITIC,
            eps=self.model_cfg.OPTIMIZER_EPS
        )

        # Memory buffer
        self.memory = PPOMemory()

        # Training stats
        self.training_steps = 0

        if self.cfg.colab.DEBUG_MODE:
            print(f"PPOAgent ({agent_type}) initialized on {self.device}")

    def _compute_monotonicity_penalty(self, states: torch.Tensor, actions_mu: torch.Tensor) -> torch.Tensor:
        """
        IMPROVED: Robust monotonicity regularization
        Only applied to FG agent, penalizes d_action/d_inventory > 0
        """
        if self.agent_type != "fg_agent":
            # Only apply monotonicity penalty to FG agent
            return torch.tensor(0.0, device=self.device)

        try:
            num_products = config.config.env.NUM_PRODUCTS

            # For FG agent, inventory positions are the first num_products elements
            inventory_indices = list(range(num_products))

            monotonicity_penalties = []

            for product_idx, inv_idx in enumerate(inventory_indices):
                if inv_idx >= states.shape[1]:
                    continue  # Skip if inventory index is out of bounds

                # Compute gradient: d(action[product_idx]) / d(inventory[inv_idx])
                if states.requires_grad:
                    # Reset gradients
                    if states.grad is not None:
                        states.grad.zero_()

                # Ensure we're tracking gradients
                states_requires_grad = states.requires_grad
                if not states_requires_grad:
                    states.requires_grad_(True)

                # Forward pass to recompute actions with gradient tracking
                mu_recompute, _, _ = self.actor_critic(states)

                # Compute gradient for this specific product
                grad_outputs = torch.ones_like(mu_recompute[:, product_idx])

                gradients = torch.autograd.grad(
                    outputs=mu_recompute[:, product_idx],
                    inputs=states,
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )

                if gradients[0] is not None:
                    # Extract gradient for this inventory position
                    inventory_grad = gradients[0][:, inv_idx]

                    # Penalize positive gradients (increasing order when inventory increases)
                    # Using smooth L2 penalty for positive gradients only
                    positive_grads = F.relu(inventory_grad)
                    product_penalty = (positive_grads ** 2).mean()
                    monotonicity_penalties.append(product_penalty)

                # Restore original requires_grad state
                if not states_requires_grad:
                    states.requires_grad_(False)

            if monotonicity_penalties:
                total_penalty = torch.stack(monotonicity_penalties).mean()
            else:
                total_penalty = torch.tensor(0.0, device=self.device)

            return total_penalty

        except Exception as e:
            print(f"▲ Monotonicity penalty computation failed: {e}")
            return torch.tensor(0.0, device=self.device)

    def store_transition(self, state: np.ndarray, action: np.ndarray,
                        log_prob: np.ndarray, reward: dict,
                        value: float, done: bool):
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        log_prob = np.asarray(log_prob, dtype=np.float32)
        self.memory.store(state, action, log_prob, reward, value, done)

    def act(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, value = self.actor_critic.get_action(
                state_tensor, deterministic=deterministic
            )
            action_np = action.cpu().numpy()[0]
            log_prob_np = log_prob.cpu().numpy()[0]
            value_np = value.cpu().numpy()[0]

        return action_np, log_prob_np, value_np

    def update(self) -> Dict[str, float]:
        """Perform PPO update with robust monotonicity regularization"""
        if len(self.memory) < self.ppo_cfg.MINIBATCH_SIZE:
            return {"skipped": 1.0}

        try:
            # Get training data
            states, actions, log_probs_old, advantages, returns, batches = self.memory.get_batches(
                self.ppo_cfg.MINIBATCH_SIZE
            )
            if len(states) == 0:
                return {"skipped": 1.0}

            # Convert to tensors
            states_tensor = torch.FloatTensor(states).to(self.device)
            actions_tensor = torch.FloatTensor(actions).to(self.device)
            log_probs_old_tensor = torch.FloatTensor(log_probs_old).to(self.device)
            advantages_tensor = torch.FloatTensor(advantages).to(self.device)
            returns_tensor = torch.FloatTensor(returns).to(self.device)

            # Training statistics
            stats = {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy_loss': 0.0,
                'total_loss': 0.0,
                'clip_fraction': 0.0,
                'monotonicity_penalty': 0.0
            }

            # PPO epochs
            update_count = 0
            for _ in range(self.ppo_cfg.PPO_EPOCHS):
                for batch_idx in batches:
                    if len(batch_idx) == 0:
                        continue

                    # Get batch data
                    batch_states = states_tensor[batch_idx].clone().detach().requires_grad_(True)
                    batch_actions = actions_tensor[batch_idx]
                    batch_log_probs_old = log_probs_old_tensor[batch_idx]
                    batch_advantages = advantages_tensor[batch_idx]
                    batch_returns = returns_tensor[batch_idx]

                    # Get current policy outputs
                    mu, sigma, values = self.actor_critic(batch_states)

                    # Handle different action dimensions
                    action_dim = min(mu.shape[1], batch_actions.shape[1])
                    mu = mu[:, :action_dim]
                    sigma = sigma[:, :action_dim]
                    batch_actions = batch_actions[:, :action_dim]

                    if batch_log_probs_old.dim() > 1:
                        batch_log_probs_old = batch_log_probs_old[:, :action_dim]

                    # Create distribution and calculate new log probs
                    dist = torch.distributions.Normal(mu, sigma)
                    log_probs_new = dist.log_prob(batch_actions).sum(dim=1, keepdim=True)

                    # Calculate probability ratio
                    ratio = torch.exp(log_probs_new - batch_log_probs_old)

                    # PPO Clipped Objective
                    surr1 = ratio * batch_advantages.unsqueeze(1)
                    surr2 = torch.clamp(ratio, 1 - self.ppo_cfg.EPSILON_CLIP,
                                      1 + self.ppo_cfg.EPSILON_CLIP) * batch_advantages.unsqueeze(1)
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value function loss
                    value_loss = F.mse_loss(values, batch_returns.unsqueeze(1))

                    # Entropy bonus
                    entropy_loss = -dist.entropy().mean()

                    # Monotonicity regularization (only for FG agent)
                    monotonicity_penalty = self._compute_monotonicity_penalty(batch_states, mu)

                    # Total loss with monotonicity regularization
                    total_loss = (policy_loss +
                                self.ppo_cfg.VALUE_COEFF * value_loss +
                                self.ppo_cfg.ENTROPY_COEFF * entropy_loss +
                                self.cfg.training.LAMBDA_MONO * monotonicity_penalty)

                    # Gradient updates
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()

                    total_loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        list(self.actor_critic.actor_mean.parameters()) +
                        list(self.actor_critic.actor_log_std.parameters()),
                        self.ppo_cfg.MAX_GRAD_NORM
                    )

                    torch.nn.utils.clip_grad_norm_(
                        self.actor_critic.critic.parameters(),
                        self.ppo_cfg.MAX_GRAD_NORM
                    )

                    self.actor_optimizer.step()
                    self.critic_optimizer.step()

                    # Update statistics
                    stats['policy_loss'] += policy_loss.item()
                    stats['value_loss'] += value_loss.item()
                    stats['entropy_loss'] += entropy_loss.item()
                    stats['total_loss'] += total_loss.item()
                    stats['monotonicity_penalty'] += monotonicity_penalty.item()

                    # Calculate clip fraction
                    clipped = torch.logical_or(
                        ratio < (1 - self.ppo_cfg.EPSILON_CLIP),
                        ratio > (1 + self.ppo_cfg.EPSILON_CLIP)
                    )
                    stats['clip_fraction'] += clipped.float().mean().item()

                    self.training_steps += 1
                    update_count += 1

            # Average statistics
            if update_count > 0:
                for key in stats:
                    if key != 'skipped':
                        stats[key] /= update_count

            # Clear memory
            self.memory.clear()

            return stats

        except Exception as e:
            print(f"▲ PPO update failed: {e}")
            self.memory.clear()
            return {'error': 1.0}

    def save_checkpoint(self, filepath: str):
        checkpoint = {
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_steps': self.training_steps
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.training_steps = checkpoint['training_steps']
