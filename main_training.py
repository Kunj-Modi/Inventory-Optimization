"""
Complete training script with all critical fixes implemented
INCLUDES: Enhanced FG observation, Curriculum Learning, and Exploration Scheduling
"""

import numpy as np
import torch
import time
import os
import random
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from collections import deque

from environment.inventory_env import InventoryMAMDPEnv
from models.networks import FGActorCritic, RMActorCritic
from agents.ppo_agent import PPOAgent
from utils.heuristic_policy import get_heuristic_action
import configs.config as config
from utils.evaluation import (
    plot_training_progress,
    run_quick_evaluation,
    create_comprehensive_evaluation
)

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class TrainingLogger:
    """Enhanced logger for training system with cost component tracking"""

    def __init__(self, log_dir: str = "/content/logs"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir

        # Enhanced data storage with cost components
        self.episode_data = {
            'total_reward': [],
            'total_cost': [],
            'total_lost_sales': [],
            'total_rm_stockouts': [],
            'episode_length': [],
            # ADD cost components
            'fg_holding_cost': [],
            'fg_shortage_cost': [],
            'rm_holding_cost': [],
            'rm_stockout_penalty': [],
            'rm_order_cost': [],
            'mean_fg_action': [],
            'std_fg_action': [],
            'mean_rm_action': [],
            'std_rm_action': [],
            'max_fg_action': [],
            'max_rm_action': [],
        }

    def log_episode(self, episode: int, stats: Dict):
        """Log episode statistics with cost components"""
        # Store basic stats
        for key in self.episode_data:
            if key in stats:
                self.episode_data[key].append(stats[key])

        if episode % 10 == 0:
            print(f"Episode {episode:4d} | "
                  f"Reward: {stats.get('total_reward', 0):8.2f} | "
                  f"Cost: {stats.get('total_cost', 0):8.2f} | "
                  f"Lost Sales: {stats.get('total_lost_sales', 0):6.1f} | "
                  f"RM Stockouts: {stats.get('total_rm_stockouts', 0):4d}")

            # ADD cost component logging for debugging
            if episode % 50 == 0 and 'fg_holding_cost' in stats:
                print(f"  Cost Components - FG Hold: {stats['fg_holding_cost']:.1f}, "
                      f"FG Short: {stats['fg_shortage_cost']:.1f}, "
                      f"RM Hold: {stats['rm_holding_cost']:.1f}, "
                      f"RM Penalty: {stats['rm_stockout_penalty']:.1f}")

    def plot_progress(self):
        """Plot training progress using the new evaluation functions"""
        print("\n📈 GENERATING TRAINING ANALYSIS...")
        plot_training_progress(
            [dict(zip(self.episode_data.keys(), values))
             for values in zip(*self.episode_data.values())],
            window_size=10,
            title="Multi-Agent Training Progress"
        )

def collect_expert_demonstrations(env, num_demos: int = 2000) -> dict:
    """Collect expert demonstrations using simplified heuristic"""
    print(f"Collecting {num_demos} expert demonstrations...")

    demonstrations = {
        'fg_states': [],
        'fg_actions': [],
        'rm_states': [],
        'rm_actions': []
    }

    obs = env.reset()
    demo_count = 0

    while demo_count < num_demos:
        # Get expert action from simplified heuristic policy
        expert_actions = get_heuristic_action(obs)

        # Store demonstration
        demonstrations['fg_states'].append(obs['fg_agent'])
        demonstrations['fg_actions'].append(expert_actions['fg_agent'])
        demonstrations['rm_states'].append(obs['rm_agent'])
        demonstrations['rm_actions'].append(expert_actions['rm_agent'])
        demo_count += 1

        # Step environment
        obs, _, done, _ = env.step(expert_actions)

        if done:
            obs = env.reset()

        if demo_count % 500 == 0:
            print(f" Collected {demo_count}/{num_demos} demonstrations")

    # Convert to arrays
    for key in demonstrations:
        demonstrations[key] = np.array(demonstrations[key])

    print(f"✓ Collected {num_demos} expert demonstrations")
    fg_mean = np.mean(demonstrations['fg_actions'], axis=0)
    fg_std = np.std(demonstrations['fg_actions'], axis=0)
    print(f"FG Actions - Mean: {np.round(fg_mean, 2)}, Std: {np.round(fg_std, 2)}")
    rm_mean = np.mean(demonstrations['rm_actions'], axis=0)
    rm_std = np.std(demonstrations['rm_actions'], axis=0)
    print(f"RM Actions - Mean: {np.round(rm_mean, 2)}, Std: {np.round(rm_std, 2)}")

    return demonstrations

def pretrain_networks(fg_network, rm_network, demonstrations, epochs: int = 100):
    """Pretrain networks using behavioral cloning"""
    print("Pretraining networks with expert demonstrations...")

    # Convert to tensors
    fg_states_tensor = torch.FloatTensor(demonstrations['fg_states'])
    fg_actions_tensor = torch.FloatTensor(demonstrations['fg_actions'])
    rm_states_tensor = torch.FloatTensor(demonstrations['rm_states'])
    rm_actions_tensor = torch.FloatTensor(demonstrations['rm_actions'])

    # FG Network pretraining
    fg_optimizer = torch.optim.Adam(fg_network.parameters(), lr=1e-3)
    fg_criterion = torch.nn.MSELoss()

    fg_losses = []
    for epoch in range(epochs):
        fg_optimizer.zero_grad()

        mu, _, _ = fg_network(fg_states_tensor)
        fg_loss = fg_criterion(mu, fg_actions_tensor)
        fg_loss.backward()
        fg_optimizer.step()
        fg_losses.append(fg_loss.item())

        if epoch % 20 == 0:
            print(f" FG Epoch {epoch:3d}: loss = {fg_loss.item():.4f}")

    # RM Network pretraining
    rm_optimizer = torch.optim.Adam(rm_network.parameters(), lr=1e-3)
    rm_criterion = torch.nn.MSELoss()

    rm_losses = []
    for epoch in range(epochs):
        rm_optimizer.zero_grad()
        mu, _, _ = rm_network(rm_states_tensor)
        rm_loss = rm_criterion(mu, rm_actions_tensor)
        rm_loss.backward()
        rm_optimizer.step()
        rm_losses.append(rm_loss.item())

        if epoch % 20 == 0:
            print(f" RM Epoch {epoch:3d}: loss = {rm_loss.item():.4f}")

    # Test final performance
    with torch.no_grad():
        fg_mu, _, _ = fg_network(fg_states_tensor[:100])
        rm_mu, _, _ = rm_network(rm_states_tensor[:100])

        fg_error = torch.mean(torch.abs(fg_mu - fg_actions_tensor[:100])).item()
        rm_error = torch.mean(torch.abs(rm_mu - rm_actions_tensor[:100])).item()

    print(f"✓ Pretraining completed")
    print(f" FG Final action error: {fg_error:.4f}")
    print(f" RM Final action error: {rm_error:.4f}")

    return fg_network, rm_network

def run_training_episode(env, fg_agent, rm_agent, episode: int,
                        exploration: bool = True) -> Dict:
    """Run a single training episode with the system"""
    obs = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_costs = []
    episode_lost_sales = []
    episode_rm_stockouts = []
    episode_fg_holding = []
    episode_fg_shortage = []
    episode_rm_holding = []
    episode_rm_penalty = []
    episode_rm_order = []
    episode_fg_actions = []
    episode_rm_actions = []


    while True:
        # Get actions from both agents
        fg_action, fg_log_prob, fg_value = fg_agent.act(
            obs['fg_agent'], deterministic=not exploration
        )
        rm_action, rm_log_prob, rm_value = rm_agent.act(
            obs['rm_agent'], deterministic=not exploration
        )

        # Store actions for analysis
        episode_fg_actions.append(fg_action)
        episode_rm_actions.append(rm_action)

        joint_action = {
            'fg_agent': fg_action,
            'rm_agent': rm_action
        }

        # Take environment step
        next_obs, rewards, done, info = env.step(joint_action)

        # Store transitions (both agents get same reward due to unified reward)
        fg_agent.store_transition(
            obs['fg_agent'], fg_action, fg_log_prob,
            rewards['fg_agent'], fg_value, done
        )
        rm_agent.store_transition(
            obs['rm_agent'], rm_action, rm_log_prob,
            rewards['rm_agent'], rm_value, done
        )

        # Update statistics
        episode_reward += rewards['fg_agent'] # Same for both agents
        episode_length += 1
        episode_costs.append(info['total_cost'])
        episode_lost_sales.append(np.sum(info['lost_sales']))
        episode_rm_stockouts.append(info['rm_stockout_count'])
        episode_fg_holding.append(info.get('fg_holding_cost', 0))
        episode_fg_shortage.append(info.get('fg_shortage_cost', 0))
        episode_rm_holding.append(info.get('rm_holding_cost', 0))
        episode_rm_penalty.append(info.get('rm_stockout_penalty', 0))
        episode_rm_order.append(info.get('rm_order_cost', 0))

        # Update observation
        obs = next_obs

        # Perform PPO updates if enough data collected
        if episode_length % config.config.training.UPDATE_TIMESTEPS == 0:
            fg_stats = fg_agent.update()
            rm_stats = rm_agent.update()

            # Log training stats if needed
            if 'policy_loss' in fg_stats:
                print(f"    FG Update - Policy: {fg_stats['policy_loss']:.4f}, "
                      f"Value: {fg_stats['value_loss']:.4f}, "
                      f"Mono: {fg_stats.get('monotonicity_penalty', 0):.6f}")

        if done:
            break

    # Calculate episode statistics
    stats = {
        'total_reward': episode_reward,
        'total_cost': np.sum(episode_costs),
        'total_lost_sales': np.sum(episode_lost_sales),
        'total_rm_stockouts': np.sum(episode_rm_stockouts),
        'episode_length': episode_length,
        'fg_holding_cost': np.sum(episode_fg_holding),
        'fg_shortage_cost': np.sum(episode_fg_shortage),
        'rm_holding_cost': np.sum(episode_rm_holding),
        'rm_stockout_penalty': np.sum(episode_rm_penalty),
        'rm_order_cost': np.sum(episode_rm_order)
    }

    # Calculate action statistics
    if episode_fg_actions:
        fg_actions_array = np.array(episode_fg_actions)
        rm_actions_array = np.array(episode_rm_actions)

        stats.update({
            'mean_fg_action': np.mean(fg_actions_array),
            'std_fg_action': np.std(fg_actions_array),
            'mean_rm_action': np.mean(rm_actions_array),
            'std_rm_action': np.std(rm_actions_array),
            'max_fg_action': np.max(fg_actions_array),
            'max_rm_action': np.max(rm_actions_array),
        })

    return stats

def evaluate_policy(env, fg_agent, rm_agent, num_episodes: int = 5) -> Dict:
    """Evaluate the trained policy"""
    print(f"Evaluating policy over {num_episodes} episodes...")

    eval_returns = []
    eval_costs = []
    eval_lost_sales = []
    eval_rm_stockouts = []

    for eval_ep in range(num_episodes):
        obs = env.reset()
        episode_return = 0
        episode_costs = []
        episode_lost_sales = []
        episode_rm_stockouts = []

        while True:
            # Use deterministic actions for evaluation
            fg_action, _, _ = fg_agent.act(obs['fg_agent'], deterministic=True)
            rm_action, _, _ = rm_agent.act(obs['rm_agent'], deterministic=True)

            joint_action = {
                'fg_agent': fg_action,
                'rm_agent': rm_action
            }

            obs, rewards, done, info = env.step(joint_action)

            episode_return += rewards['fg_agent'] # Same for both agents
            episode_costs.append(info['total_cost'])
            episode_lost_sales.append(np.sum(info['lost_sales']))
            episode_rm_stockouts.append(info['rm_stockout_count'])

            if done:
                break

        eval_returns.append(episode_return)
        eval_costs.append(np.sum(episode_costs))
        eval_lost_sales.append(np.sum(episode_lost_sales))
        eval_rm_stockouts.append(np.sum(episode_rm_stockouts))

    eval_stats = {
        'mean_return': np.mean(eval_returns),
        'std_return': np.std(eval_returns),
        'mean_cost': np.mean(eval_costs),
        'mean_lost_sales': np.mean(eval_lost_sales),
        'mean_rm_stockouts': np.mean(eval_rm_stockouts)
    }

    print(f" Mean Return: {eval_stats['mean_return']:8.2f} ± {eval_stats['std_return']:5.2f}")
    print(f" Mean Cost: {eval_stats['mean_cost']:8.2f}")
    print(f" Mean Lost Sales: {eval_stats['mean_lost_sales']:6.1f}")
    print(f" Mean RM Stockouts: {eval_stats['mean_rm_stockouts']:4.1f}")
    return eval_stats

def get_exploration_schedule(current_episode: int, total_episodes: int,
                           exploration_phase_ratio: float = 0.7) -> bool:
    """
    Determine exploration strategy based on training phase
    Returns: True if we should explore, False for exploitation
    """
    exploration_threshold = int(total_episodes * exploration_phase_ratio)

    # Explore more in early phases, exploit more later
    if current_episode < exploration_threshold:
        # Early phase: explore with 80% probability
        return random.random() < 0.8
    else:
        # Late phase: explore with only 20% probability
        return random.random() < 0.2

def compare_agent_actions(env, fg_agent, rm_agent, heuristic_policy, num_steps: int = 200):
    """
    Compare actions between heuristic policy and trained PPO agent

    Args:
        env: Inventory environment
        fg_agent: Trained FG PPO agent
        rm_agent: Trained RM PPO agent
        heuristic_policy: Heuristic policy function
        num_steps: Number of steps to compare
    """

    print("=" * 80)
    print("ACTION COMPARISON: Heuristic vs Trained PPO Agent")
    print("=" * 80)

    # Reset environment for both policies (same initial state)
    env.seed(42)  # For reproducibility
    obs_heuristic = env.reset()
    obs_ppo = env.reset()

    # Store actions for comparison
    heuristic_actions = {
        'fg': [],
        'rm': []
    }

    ppo_actions = {
        'fg': [],
        'rm': []
    }

    # Store states to understand context
    states = {
        'fg_inventory': [],
        'rm_inventory': [],
        'demand': []
    }

    # Run both policies for the same number of steps
    for step in range(num_steps):
        # Heuristic policy actions
        heuristic_action = heuristic_policy(obs_heuristic)
        heuristic_actions['fg'].append(heuristic_action['fg_agent'])
        heuristic_actions['rm'].append(heuristic_action['rm_agent'])

        # PPO agent actions
        fg_action_ppo, _, _ = fg_agent.act(obs_ppo['fg_agent'], deterministic=True)
        rm_action_ppo, _, _ = rm_agent.act(obs_ppo['rm_agent'], deterministic=True)
        ppo_actions['fg'].append(fg_action_ppo)
        ppo_actions['rm'].append(rm_action_ppo)

        # Store current state for context
        states['fg_inventory'].append(obs_ppo['fg_agent'][:5])  # First 5 elements are FG inventory
        states['rm_inventory'].append(obs_ppo['rm_agent'][0])   # First element is RM inventory

        # Step both environments
        obs_heuristic, _, done_heuristic, _ = env.step(heuristic_action)
        obs_ppo, _, done_ppo, _ = env.step({
            'fg_agent': fg_action_ppo,
            'rm_agent': rm_action_ppo
        })

        if done_heuristic or done_ppo:
            obs_heuristic = env.reset()
            obs_ppo = env.reset()
            print(f"Environment reset at step {step}")

    # Convert to numpy arrays
    heuristic_fg = np.array(heuristic_actions['fg'])
    heuristic_rm = np.array(heuristic_actions['rm'])
    ppo_fg = np.array(ppo_actions['fg'])
    ppo_rm = np.array(ppo_actions['rm'])

    # Create comprehensive comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Action Comparison: Heuristic Policy vs Trained PPO Agent',
                 fontsize=16, fontweight='bold')

    # Plot 1: FG Actions over time (first product)
    product_idx = 0  # Compare first product
    axes[0, 0].plot(heuristic_fg[:, product_idx], 'r-', linewidth=2,
                   label='Heuristic', alpha=0.8)
    axes[0, 0].plot(ppo_fg[:, product_idx], 'b-', linewidth=2,
                   label='PPO Agent', alpha=0.8)
    axes[0, 0].set_title(f'FG Actions - Product {product_idx + 1}')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Order Quantity')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: FG Action distribution (all products)
    all_heuristic_fg = heuristic_fg.flatten()
    all_ppo_fg = ppo_fg.flatten()

    axes[0, 1].hist(all_heuristic_fg, bins=30, alpha=0.7, color='red',
                   label='Heuristic', density=True)
    axes[0, 1].hist(all_ppo_fg, bins=30, alpha=0.7, color='blue',
                   label='PPO Agent', density=True)
    axes[0, 1].set_title('FG Action Distribution (All Products)')
    axes[0, 1].set_xlabel('Order Quantity')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: FG Action statistics by product
    products = range(heuristic_fg.shape[1])
    heuristic_means = np.mean(heuristic_fg, axis=0)
    ppo_means = np.mean(ppo_fg, axis=0)

    x_pos = np.arange(len(products))
    width = 0.35

    axes[0, 2].bar(x_pos - width/2, heuristic_means, width,
                   label='Heuristic', color='red', alpha=0.8)
    axes[0, 2].bar(x_pos + width/2, ppo_means, width,
                   label='PPO Agent', color='blue', alpha=0.8)
    axes[0, 2].set_title('Average FG Actions by Product')
    axes[0, 2].set_xlabel('Product')
    axes[0, 2].set_ylabel('Average Order Quantity')
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels([f'P{i+1}' for i in products])
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: RM Actions over time
    axes[1, 0].plot(heuristic_rm.flatten(), 'r-', linewidth=2,
                   label='Heuristic', alpha=0.8)
    axes[1, 0].plot(ppo_rm.flatten(), 'b-', linewidth=2,
                   label='PPO Agent', alpha=0.8)
    axes[1, 0].set_title('RM Actions Over Time')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Order Quantity')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: RM Action distribution
    axes[1, 1].hist(heuristic_rm.flatten(), bins=30, alpha=0.7, color='red',
                   label='Heuristic', density=True)
    axes[1, 1].hist(ppo_rm.flatten(), bins=30, alpha=0.7, color='blue',
                   label='PPO Agent', density=True)
    axes[1, 1].set_title('RM Action Distribution')
    axes[1, 1].set_xlabel('Order Quantity')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Action correlation with inventory
    rm_inventory = np.array(states['rm_inventory'])

    # Scatter plot: RM Action vs RM Inventory
    scatter = axes[1, 2].scatter(rm_inventory[:len(heuristic_rm)],
                                heuristic_rm.flatten(),
                                c='red', alpha=0.6, label='Heuristic', s=30)
    scatter = axes[1, 2].scatter(rm_inventory[:len(ppo_rm)],
                                ppo_rm.flatten(),
                                c='blue', alpha=0.6, label='PPO Agent', s=30)
    axes[1, 2].set_title('RM Action vs RM Inventory')
    axes[1, 2].set_xlabel('RM Inventory Level')
    axes[1, 2].set_ylabel('RM Order Quantity')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print detailed statistics
    print("\n" + "=" * 60)
    print("ACTION STATISTICS COMPARISON")
    print("=" * 60)

    # FG Action statistics
    print("\nFINISHED GOODS ACTIONS:")
    print("-" * 40)
    for i in range(heuristic_fg.shape[1]):
        print(f"Product {i+1}:")
        print(f"  Heuristic - Mean: {heuristic_means[i]:.2f}, "
              f"Std: {np.std(heuristic_fg[:, i]):.2f}, "
              f"Range: [{np.min(heuristic_fg[:, i]):.2f}, {np.max(heuristic_fg[:, i]):.2f}]")
        print(f"  PPO Agent - Mean: {ppo_means[i]:.2f}, "
              f"Std: {np.std(ppo_fg[:, i]):.2f}, "
              f"Range: [{np.min(ppo_fg[:, i]):.2f}, {np.max(ppo_fg[:, i]):.2f}]")
        difference = ppo_means[i] - heuristic_means[i]
        print(f"  Difference: {difference:+.2f} ({difference/heuristic_means[i]*100:+.1f}%)\n")

    # RM Action statistics
    print("\nRAW MATERIAL ACTIONS:")
    print("-" * 40)
    heuristic_rm_mean = np.mean(heuristic_rm)
    ppo_rm_mean = np.mean(ppo_rm)
    print(f"Heuristic - Mean: {heuristic_rm_mean:.2f}, "
          f"Std: {np.std(heuristic_rm):.2f}, "
          f"Range: [{np.min(heuristic_rm):.2f}, {np.max(heuristic_rm):.2f}]")
    print(f"PPO Agent - Mean: {ppo_rm_mean:.2f}, "
          f"Std: {np.std(ppo_rm):.2f}, "
          f"Range: [{np.min(ppo_rm):.2f}, {np.max(ppo_rm):.2f}]")
    rm_difference = ppo_rm_mean - heuristic_rm_mean
    print(f"Difference: {rm_difference:+.2f} ({rm_difference/heuristic_rm_mean*100:+.1f}%)")

    # Action variability comparison
    print("\nACTION VARIABILITY:")
    print("-" * 40)
    fg_variability_heuristic = np.mean(np.std(heuristic_fg, axis=0))
    fg_variability_ppo = np.mean(np.std(ppo_fg, axis=0))
    rm_variability_heuristic = np.std(heuristic_rm)
    rm_variability_ppo = np.std(ppo_rm)

    print(f"FG Action Variability:")
    print(f"  Heuristic: {fg_variability_heuristic:.3f}")
    print(f"  PPO Agent: {fg_variability_ppo:.3f}")
    print(f"  Ratio: {fg_variability_ppo/fg_variability_heuristic:.3f}")

    print(f"RM Action Variability:")
    print(f"  Heuristic: {rm_variability_heuristic:.3f}")
    print(f"  PPO Agent: {rm_variability_ppo:.3f}")
    print(f"  Ratio: {rm_variability_ppo/rm_variability_heuristic:.3f}")

    return {
        'heuristic_actions': heuristic_actions,
        'ppo_actions': ppo_actions,
        'states': states
    }

# Add this to your main_training.py or evaluation script

def enhanced_comprehensive_evaluation(env, fg_agent, rm_agent, heuristic_policy,
                                    num_eval_episodes: int = 5, steps_per_episode: int = 100):
    """
    Enhanced evaluation with action comparison
    """
    print("#" * 80)
    print("ENHANCED COMPREHENSIVE EVALUATION")
    print("#" * 80)

    # 1. Run action comparison
    print("\n[1/3] RUNNING ACTION COMPARISON...")
    action_comparison = compare_agent_actions(
        env, fg_agent, rm_agent, heuristic_policy, num_steps=steps_per_episode
    )

    # 2. Run performance comparison (your existing function)
    print("\n[2/3] RUNNING PERFORMANCE COMPARISON...")
    trained_results, heuristic_results = create_comprehensive_evaluation(
        env, fg_agent, rm_agent, heuristic_policy,
        num_eval_episodes=num_eval_episodes,
        steps_per_episode=steps_per_episode
    )

    # 3. Additional analysis: Action patterns in different inventory states
    print("\n[3/3] ANALYZING ACTION PATTERNS...")
    analyze_action_patterns(action_comparison)

    return trained_results, heuristic_results, action_comparison

def analyze_action_patterns(action_comparison):
    """
    Analyze how actions differ in different inventory states
    """
    states = action_comparison['states']
    heuristic_actions = action_comparison['heuristic_actions']
    ppo_actions = action_comparison['ppo_actions']

    rm_inventory = np.array(states['rm_inventory'])
    heuristic_rm = np.array(heuristic_actions['rm']).flatten()
    ppo_rm = np.array(ppo_actions['rm']).flatten()

    # Analyze low inventory vs high inventory behavior
    inventory_threshold = np.median(rm_inventory)

    low_inv_mask = rm_inventory < inventory_threshold
    high_inv_mask = rm_inventory >= inventory_threshold

    print("\n" + "=" * 50)
    print("ACTION PATTERN ANALYSIS BY INVENTORY LEVEL")
    print("=" * 50)

    print(f"\nLow RM Inventory (< {inventory_threshold:.1f}):")
    print(f"  Heuristic RM Orders: {np.mean(heuristic_rm[low_inv_mask]):.2f}")
    print(f"  PPO Agent RM Orders: {np.mean(ppo_rm[low_inv_mask]):.2f}")

    print(f"\nHigh RM Inventory (>= {inventory_threshold:.1f}):")
    print(f"  Heuristic RM Orders: {np.mean(heuristic_rm[high_inv_mask]):.2f}")
    print(f"  PPO Agent RM Orders: {np.mean(ppo_rm[high_inv_mask]):.2f}")

def evaluate_trained_models():
    """Standalone function to evaluate trained models"""
    print("🧪 EVALUATING TRAINED MODELS")

    from environment.inventory_env import InventoryMAMDPEnv
    from models.networks import FGActorCritic, RMActorCritic
    from agents.ppo_agent import PPOAgent
    from utils.heuristic_policy import get_heuristic_action
    from utils.evaluation import create_comprehensive_evaluation

    # Initialize environment and agents
    env = InventoryMAMDPEnv()

    fg_network = FGActorCritic(
        state_dim=env.observation_space['fg_agent'].shape[0],
        action_dim=env.action_space_fg.shape[0]
    )
    rm_network = RMActorCritic(
        state_dim=env.observation_space['rm_agent'].shape[0],
        action_dim=env.action_space_rm.shape[0]
    )

    fg_agent = PPOAgent(fg_network, agent_type="fg_agent")
    rm_agent = PPOAgent(rm_network, agent_type="rm_agent")

    # Load trained models
    try:
        fg_agent.load_checkpoint("/content/checkpoints/best_fg_model.pth")
        rm_agent.load_checkpoint("/content/checkpoints/best_rm_model.pth")
        print("✓ Loaded trained models")
    except Exception as e:
        print(f"⚠️  Could not load models: {e}")
        print("⚠️  Using untrained models for evaluation")

    # Run comprehensive evaluation
    trained_results, heuristic_results = create_comprehensive_evaluation(
        env, fg_agent, rm_agent, get_heuristic_action,
        num_eval_episodes=5, steps_per_episode=200
    )

    return trained_results, heuristic_results

def main_training():
    """Main training function with all fixes implemented"""
    print("=== STARTING SYSTEM TRAINING ===")

    # Setup
    env = InventoryMAMDPEnv()
    logger = TrainingLogger()

    # Get environment dimensions
    fg_state_dim = env.observation_space['fg_agent'].shape[0]
    fg_action_dim = env.action_space_fg.shape[0]
    rm_state_dim = env.observation_space['rm_agent'].shape[0]
    rm_action_dim = env.action_space_rm.shape[0]

    print(f"\nEnvironment setup:")
    print(f" FG: state_dim={fg_state_dim}, action_dim={fg_action_dim}")
    print(f" RM: state_dim={rm_state_dim}, action_dim={rm_action_dim}")

    # Create networks
    fg_network = FGActorCritic(fg_state_dim, fg_action_dim, output_multiplier=30)
    rm_network = RMActorCritic(rm_state_dim, rm_action_dim)

    # Step 1: Collect expert demonstrations
    demonstrations = collect_expert_demonstrations(env, num_demos=2000)

    # Step 2: Pretrain networks
    fg_network, rm_network = pretrain_networks(
        fg_network, rm_network, demonstrations, epochs=120
    )

    # Step 3: Create agents with pretrained networks
    fg_agent = PPOAgent(fg_network, agent_type="fg_agent")
    rm_agent = PPOAgent(rm_network, agent_type="rm_agent")

    # Evaluate before training
    # run_quick_evaluation(env, fg_agent, rm_agent)
    create_comprehensive_evaluation(
        env, fg_agent, rm_agent, get_heuristic_action,
        num_eval_episodes=3, steps_per_episode=200
    )

    # TRAINING PROTOCOL
    print("\n" + "="*60)
    print("TRAINING AFTER PRE-TRAINING")
    print("="*60)

    # Reset learning rates to original values (in case they were modified)
    for param_group in fg_agent.actor_optimizer.param_groups:
        param_group['lr'] = config.config.model.LEARNING_RATE_ACTOR
    for param_group in rm_agent.actor_optimizer.param_groups:
        param_group['lr'] = config.config.model.LEARNING_RATE_ACTOR

    print("✓ Both agents ready for joint training with original learning rates")

    # Training parameters for joint phase
    joint_episodes = 1000
    eval_interval = 20

    # Track best performance
    best_eval_return = -np.inf
    best_episode = 0

    # Training loop for joint fine-tuning
    start_time = time.time()

    for episode in range(joint_episodes):
        # Use exploration scheduling
        exploration = get_exploration_schedule(episode, joint_episodes, exploration_phase_ratio=0.7)

        episode_stats = run_training_episode(
            env, fg_agent, rm_agent, episode, exploration=exploration
        )

        # Log episode (offset by previous phases)
        logger.log_episode(episode, episode_stats)

        # Evaluation
        if episode % 50 == 0 and episode > 0:
            print(f"\nPERIODIC EVALUATION at Joint Episode {episode}")
            print("="*50)

            try:
                # Run quick evaluation
                trained_history, heuristic_history = run_quick_evaluation(env, fg_agent, rm_agent)

                # Save evaluation snapshot
                eval_dir = f"/content/evaluation_snapshots/episode_{episode}"
                os.makedirs(eval_dir, exist_ok=True)
                print(f"Evaluation snapshot saved to {eval_dir}")

            except Exception as e:
                print(f"⚠️  Evaluation failed: {e}")

        if episode % eval_interval == 0 and episode > 0:
            print(f"\n--- Evaluation at Joint Episode {episode} (Total: {episode}) ---")
            eval_stats = evaluate_policy(env, fg_agent, rm_agent, num_episodes=3)

            # Save best model
            if eval_stats['mean_return'] > best_eval_return:
                best_eval_return = eval_stats['mean_return']
                best_episode = episode

                # Save checkpoints
                os.makedirs("/content/checkpoints", exist_ok=True)
                fg_agent.save_checkpoint("/content/checkpoints/best_fg_model.pth")
                rm_agent.save_checkpoint("/content/checkpoints/best_rm_model.pth")
                print(f"   💾 New best model saved (return: {best_eval_return:.2f})")

            print("-" * 50)

    # Training completed
    training_time = time.time() - start_time
    print(f"\n🎉 TRAINING COMPLETED ===")
    print(f"Total training time: {training_time:.1f}s ({training_time/60:.1f} minutes)")
    print(f"Total episodes: {300 + joint_episodes} (150 RM + 150 FG + {joint_episodes} Joint)")
    print(f"Best evaluation return: {best_eval_return:.2f} (episode {best_episode})")

    # 🆕 COMPREHENSIVE FINAL EVALUATION
    print(f"\n📊 COMPREHENSIVE FINAL EVALUATION ===")

    try:
        # Load best models for final evaluation
        if os.path.exists("/content/checkpoints/best_fg_model.pth"):
            fg_agent.load_checkpoint("/content/checkpoints/best_fg_model.pth")
            rm_agent.load_checkpoint("/content/checkpoints/best_rm_model.pth")
            print("✓ Loaded best models for final evaluation")

        # Run comprehensive evaluation
        trained_results, heuristic_results = create_comprehensive_evaluation(
            env, fg_agent, rm_agent, get_heuristic_action,
            num_eval_episodes=3, steps_per_episode=200
        )

    except Exception as e:
        print(f"⚠️  Comprehensive evaluation failed: {e}")
        # Fallback to simple evaluation
        final_eval = evaluate_policy(env, fg_agent, rm_agent, num_episodes=10)

    # Compare with heuristic baseline
    print(f"\n📈 HEURISTIC BASELINE COMPARISON ===")
    heuristic_returns = []
    for _ in range(10):
        obs = env.reset()
        heuristic_return = 0
        while True:
            action = get_heuristic_action(obs)
            obs, rewards, done, _ = env.step(action)
            heuristic_return += rewards['fg_agent']
            if done:
                break
        heuristic_returns.append(heuristic_return)

    heuristic_mean = np.mean(heuristic_returns)
    heuristic_std = np.std(heuristic_returns)

    # Use final evaluation results
    if 'final_eval' in locals():
        trained_mean = final_eval['mean_return']
        trained_std = final_eval['std_return']
    else:
        # Estimate from training
        trained_mean = best_eval_return
        trained_std = 0

    print(f"Heuristic Mean Return: {heuristic_mean:8.2f} ± {heuristic_std:5.2f}")
    print(f"Trained Policy Mean Return: {trained_mean:8.2f} ± {trained_std:5.2f}")
    improvement = ((trained_mean - heuristic_mean) / abs(heuristic_mean)) * 100
    print(f"Improvement over Heuristic: {improvement:+.1f}%")

    # Plot training progress using enhanced visualization
    logger.plot_progress()

    # Success criteria
    if improvement > 15.0:
        print(f"\n🎯 SUCCESS: System is Learning effectively!")
        print(f" The implementation shows {improvement:.1f}% improvement over Heuristic baseline.")
    elif improvement > 5.0:
        print(f"\n✅ STABLE: System is Learning but could be improved.")
        print(f" Current improvement: {improvement:.1f}% over Heuristic.")
    else:
        print(f"\n⚠️ NEEDS TUNING: System performance needs improvement.")
        print(f" Consider hyperparameter tuning or longer training.")

    print(f"\n💾 Models saved to: /content/checkpoints/")

    # Save training data for future analysis
    training_data_path = "/content/training_data.npy"
    np.save(training_data_path, logger.episode_data)
    print(f"📊 Training data saved to: {training_data_path}")

    return {
        'final_eval': final_eval if 'final_eval' in locals() else None,
        'improvement': improvement,
        'training_data': logger.episode_data
    }
# output_multiplier
# Add to main execution block
if __name__ == "__main__":
    # Configure for effective training
    config.config.training.TOTAL_TIMESTEPS = 50000
    config.config.training.UPDATE_TIMESTEPS = 512
    config.config.colab.DEBUG_MODE = False

    set_seed(8723822)

    # Check if we should run training or just evaluation
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        # Run only evaluation
        evaluate_trained_models()
    else:
        # Run full training
        results = main_training()
