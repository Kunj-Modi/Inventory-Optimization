"""
Comprehensive evaluation and visualization utilities for multi-agent inventory management
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import seaborn as sns
from collections import defaultdict
import torch

from utils.visualization import plot_environment_run

def plot_training_progress(training_history: List[Dict],
                         window_size: int = 10,
                         title: str = "Training Progress Analysis"):
    """
    Comprehensive training progress visualization

    Args:
        training_history: List of episode statistics from training
        window_size: Window for moving averages
        title: Plot title
    """
    if not training_history:
        print("No training data available")
        return

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(training_history)

    print(df.head())

    # Create subplots - increased to 3 rows to accommodate new plots
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Plot 1: Rewards over time
    axes[0, 0].plot(df.index, df.get('total_reward', []), 'b-', alpha=0.3, label='Raw')
    if len(df) > window_size:
        rewards_ma = df['total_reward'].rolling(window=window_size).mean()
        axes[0, 0].plot(df.index[window_size-1:], rewards_ma[window_size-1:],
                       'b-', linewidth=2, label=f'MA({window_size})')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Costs over time
    axes[0, 1].plot(df.index, df.get('total_cost', []), 'r-', alpha=0.3, label='Raw')
    if len(df) > window_size:
        costs_ma = df['total_cost'].rolling(window=window_size).mean()
        axes[0, 1].plot(df.index[window_size-1:], costs_ma[window_size-1:],
                       'r-', linewidth=2, label=f'MA({window_size})')
    axes[0, 1].set_title('Episode Costs')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Total Cost')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Lost Sales over time
    axes[0, 2].plot(df.index, df.get('total_lost_sales', []), 'orange', alpha=0.3, label='Raw')
    if len(df) > window_size:
        lost_sales_ma = df['total_lost_sales'].rolling(window=window_size).mean()
        axes[0, 2].plot(df.index[window_size-1:], lost_sales_ma[window_size-1:],
                       'orange', linewidth=2, label=f'MA({window_size})')
    axes[0, 2].set_title('Lost Sales per Episode')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Lost Sales Quantity')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: RM Stockouts over time
    axes[1, 0].plot(df.index, df.get('total_rm_stockouts', []), 'purple', alpha=0.3, label='Raw')
    if len(df) > window_size:
        stockouts_ma = df['total_rm_stockouts'].rolling(window=window_size).mean()
        axes[1, 0].plot(df.index[window_size-1:], stockouts_ma[window_size-1:],
                       'purple', linewidth=2, label=f'MA({window_size})')
    axes[1, 0].set_title('RM Stockouts per Episode')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('RM Stockouts')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Episode lengths
    axes[1, 1].plot(df.index, df.get('episode_length', []), 'green', alpha=0.3, label='Raw')
    if len(df) > window_size:
        length_ma = df['episode_length'].rolling(window=window_size).mean()
        axes[1, 1].plot(df.index[window_size-1:], length_ma[window_size-1:],
                       'green', linewidth=2, label=f'MA({window_size})')
    axes[1, 1].set_title('Episode Lengths')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Steps per Episode')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: FG Actions over time
    if 'mean_fg_action' in df.columns:
        axes[1, 2].plot(df.index, df['mean_fg_action'], 'teal', alpha=0.3, label='Raw')
        if len(df) > window_size:
            fg_actions_ma = df['mean_fg_action'].rolling(window=window_size).mean()
            axes[1, 2].plot(df.index[window_size-1:], fg_actions_ma[window_size-1:],
                           'teal', linewidth=2, label=f'MA({window_size})')
        axes[1, 2].set_title('Average FG Action per Episode')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Average FG Action')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].text(0.5, 0.5, 'FG Action Data\nNot Available',
                       ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Average FG Action per Episode')

    # Plot 7: RM Actions over time
    if 'mean_rm_action' in df.columns:
        axes[2, 0].plot(df.index, df['mean_rm_action'], 'brown', alpha=0.3, label='Raw')
        if len(df) > window_size:
            rm_actions_ma = df['mean_rm_action'].rolling(window=window_size).mean()
            axes[2, 0].plot(df.index[window_size-1:], rm_actions_ma[window_size-1:],
                           'brown', linewidth=2, label=f'MA({window_size})')
        axes[2, 0].set_title('Average RM Action per Episode')
        axes[2, 0].set_xlabel('Episode')
        axes[2, 0].set_ylabel('Average RM Action')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
    else:
        axes[2, 0].text(0.5, 0.5, 'RM Action Data\nNot Available',
                       ha='center', va='center', transform=axes[2, 0].transAxes)
        axes[2, 0].set_title('Average RM Action per Episode')

    # Plot 8: FG Action Variability
    if 'std_fg_action' in df.columns:
        axes[2, 1].plot(df.index, df['std_fg_action'], 'darkblue', alpha=0.3, label='Raw')
        if len(df) > window_size:
            fg_std_ma = df['std_fg_action'].rolling(window=window_size).mean()
            axes[2, 1].plot(df.index[window_size-1:], fg_std_ma[window_size-1:],
                           'darkblue', linewidth=2, label=f'MA({window_size})')
        axes[2, 1].set_title('FG Action Variability (Std Dev)')
        axes[2, 1].set_xlabel('Episode')
        axes[2, 1].set_ylabel('FG Action Std Dev')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
    else:
        axes[2, 1].text(0.5, 0.5, 'FG Action Std Dev\nNot Available',
                       ha='center', va='center', transform=axes[2, 1].transAxes)
        axes[2, 1].set_title('FG Action Variability')

    # Plot 9: Performance metrics correlation
    if len(df) > 10:
        # Include action metrics in correlation if available
        correlation_columns = ['total_reward', 'total_cost', 'total_lost_sales', 'total_rm_stockouts']
        if 'mean_fg_action' in df.columns:
            correlation_columns.append('mean_fg_action')
        if 'mean_rm_action' in df.columns:
            correlation_columns.append('mean_rm_action')

        correlation_data = df[correlation_columns].corr()
        im = axes[2, 2].imshow(correlation_data, cmap='coolwarm', vmin=-1, vmax=1)
        axes[2, 2].set_title('Metrics Correlation')
        axes[2, 2].set_xticks(range(len(correlation_data.columns)))
        axes[2, 2].set_yticks(range(len(correlation_data.columns)))
        axes[2, 2].set_xticklabels(correlation_data.columns, rotation=45)
        axes[2, 2].set_yticklabels(correlation_data.columns)

        # Add correlation values as text
        for i in range(len(correlation_data.columns)):
            for j in range(len(correlation_data.columns)):
                axes[2, 2].text(j, i, f'{correlation_data.iloc[i, j]:.2f}',
                               ha='center', va='center', fontweight='bold')

        plt.colorbar(im, ax=axes[2, 2])
    else:
        axes[2, 2].text(0.5, 0.5, 'Correlation Matrix\nNot Available\n(Need >10 episodes)',
                       ha='center', va='center', transform=axes[2, 2].transAxes)
        axes[2, 2].set_title('Metrics Correlation')

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print_summary_statistics(df)

def print_summary_statistics(df: pd.DataFrame):
    """Print comprehensive training statistics"""
    print("\n" + "="*60)
    print("TRAINING SUMMARY STATISTICS")
    print("="*60)

    if 'total_reward' in df.columns:
        print(f"Rewards:    Mean = {df['total_reward'].mean():8.2f} ± {df['total_reward'].std():6.2f}")
        print(f"            Best  = {df['total_reward'].max():8.2f} (episode {df['total_reward'].idxmax()})")
        print(f"            Worst = {df['total_reward'].min():8.2f} (episode {df['total_reward'].idxmin()})")

    if 'total_cost' in df.columns:
        print(f"Costs:      Mean = {df['total_cost'].mean():8.2f} ± {df['total_cost'].std():6.2f}")

    if 'total_lost_sales' in df.columns:
        print(f"Lost Sales: Mean = {df['total_lost_sales'].mean():8.2f} ± {df['total_lost_sales'].std():6.2f}")

    if 'total_rm_stockouts' in df.columns:
        print(f"RM Stockouts: Mean = {df['total_rm_stockouts'].mean():8.2f} ± {df['total_rm_stockouts'].std():6.2f}")

    if 'episode_length' in df.columns:
        print(f"Episode Length: Mean = {df['episode_length'].mean():6.1f} steps")

def plot_agent_behavior_comparison(heuristic_history: List[Dict],
                                 trained_history: List[Dict],
                                 titles: Tuple[str, str] = ("Heuristic Policy", "Trained Policy")):
    """
    Compare heuristic vs trained policy performance

    Args:
        heuristic_history: Simulation history with heuristic policy
        trained_history: Simulation history with trained policy
        titles: Titles for the two policies
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Policy Comparison: Heuristic vs Trained', fontsize=16, fontweight='bold')

    # Extract data
    heuristic_costs = [step['total_cost'] for step in heuristic_history]
    trained_costs = [step['total_cost'] for step in trained_history]

    heuristic_lost_sales = [np.sum(step['lost_sales']) for step in heuristic_history]
    trained_lost_sales = [np.sum(step['lost_sales']) for step in trained_history]

    heuristic_rm_inv = [step['rm_inventory'] for step in heuristic_history]
    trained_rm_inv = [step['rm_inventory'] for step in trained_history]

    heuristic_fg_inv = [np.mean(step['fg_inventory']) for step in heuristic_history]
    trained_fg_inv = [np.mean(step['fg_inventory']) for step in trained_history]

    # Plot 1: Cost comparison
    time_steps = range(len(heuristic_costs))
    axes[0, 0].plot(time_steps, heuristic_costs, 'r-', label=titles[0], alpha=0.7)
    axes[0, 0].plot(time_steps, trained_costs, 'b-', label=titles[1], alpha=0.7)
    axes[0, 0].set_title('Daily System Cost Comparison')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Cost')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Lost sales comparison
    axes[0, 1].plot(time_steps, heuristic_lost_sales, 'r-', label=titles[0], alpha=0.7)
    axes[0, 1].plot(time_steps, trained_lost_sales, 'b-', label=titles[1], alpha=0.7)
    axes[0, 1].set_title('Lost Sales Comparison')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Lost Sales Quantity')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: RM inventory comparison
    axes[1, 0].plot(time_steps, heuristic_rm_inv, 'r-', label=titles[0], alpha=0.7)
    axes[1, 0].plot(time_steps, trained_rm_inv, 'b-', label=titles[1], alpha=0.7)
    axes[1, 0].set_title('RM Inventory Level Comparison')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('RM Inventory')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: FG inventory comparison
    axes[1, 1].plot(time_steps, heuristic_fg_inv, 'r-', label=titles[0], alpha=0.7)
    axes[1, 1].plot(time_steps, trained_fg_inv, 'b-', label=titles[1], alpha=0.7)
    axes[1, 1].set_title('Average FG Inventory Comparison')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Average FG Inventory')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print comparison statistics
    print_comparison_statistics(heuristic_history, trained_history, titles)

def print_comparison_statistics(heuristic_history: List[Dict],
                              trained_history: List[Dict],
                              titles: Tuple[str, str]):
    """Print detailed comparison statistics"""
    print("\n" + "="*60)
    print("POLICY COMPARISON STATISTICS")
    print("="*60)

    heuristic_costs = [step['total_cost'] for step in heuristic_history]
    trained_costs = [step['total_cost'] for step in trained_history]

    heuristic_lost_sales = [np.sum(step['lost_sales']) for step in heuristic_history]
    trained_lost_sales = [np.sum(step['lost_sales']) for step in trained_history]

    heuristic_rm_stockouts = [step.get('rm_stockout_count', 0) for step in heuristic_history]
    trained_rm_stockouts = [step.get('rm_stockout_count', 0) for step in trained_history]

    print(f"{'Metric':<25} {titles[0]:<15} {titles[1]:<15} {'Improvement':<15}")
    print("-" * 70)

    # Total cost comparison
    heuristic_total_cost = np.sum(heuristic_costs)
    trained_total_cost = np.sum(trained_costs)
    cost_improvement = ((heuristic_total_cost - trained_total_cost) / heuristic_total_cost) * 100
    print(f"{'Total Cost:':<25} {heuristic_total_cost:<15.2f} {trained_total_cost:<15.2f} {cost_improvement:>+6.1f}%")

    # Average daily cost
    heuristic_avg_cost = np.mean(heuristic_costs)
    trained_avg_cost = np.mean(trained_costs)
    avg_cost_improvement = ((heuristic_avg_cost - trained_avg_cost) / heuristic_avg_cost) * 100
    print(f"{'Avg Daily Cost:':<25} {heuristic_avg_cost:<15.2f} {trained_avg_cost:<15.2f} {avg_cost_improvement:>+6.1f}%")

    # Total lost sales
    heuristic_total_lost = np.sum(heuristic_lost_sales)
    trained_total_lost = np.sum(trained_lost_sales)
    lost_improvement = ((heuristic_total_lost - trained_total_lost) / heuristic_total_lost) * 100
    print(f"{'Total Lost Sales:':<25} {heuristic_total_lost:<15.2f} {trained_total_lost:<15.2f} {lost_improvement:>+6.1f}%")

    # RM stockouts
    heuristic_total_rm_stockouts = np.sum(heuristic_rm_stockouts)
    trained_total_rm_stockouts = np.sum(trained_rm_stockouts)
    rm_stockout_improvement = ((heuristic_total_rm_stockouts - trained_total_rm_stockouts) /
                              (heuristic_total_rm_stockouts + 1e-6)) * 100
    print(f"{'RM Stockouts:':<25} {heuristic_total_rm_stockouts:<15.0f} {trained_total_rm_stockouts:<15.0f} {rm_stockout_improvement:>+6.1f}%")

def plot_action_distribution(heuristic_actions: Dict,
                           trained_actions: Dict,
                           agent_type: str = "fg_agent"):
    """
    Plot action distribution comparison between heuristic and trained policies

    Args:
        heuristic_actions: Dictionary of actions from heuristic policy
        trained_actions: Dictionary of actions from trained policy
        agent_type: Type of agent ('fg_agent' or 'rm_agent')
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'{agent_type.upper()} Action Distribution Comparison', fontsize=14)

    if agent_type == "fg_agent":
        # FG agent actions (multiple products)
        heuristic_fg = np.array(heuristic_actions['fg_agent'])
        trained_fg = np.array(trained_actions['fg_agent'])

        num_products = heuristic_fg.shape[1] if len(heuristic_fg.shape) > 1 else 1

        if num_products > 1:
            # Plot distribution for each product
            for product in range(num_products):
                axes[0].hist(heuristic_fg[:, product], alpha=0.7,
                           label=f'Product {product+1}', bins=20)
            axes[0].set_title('Heuristic Policy - FG Actions')
            axes[0].set_xlabel('Order Quantity')
            axes[0].set_ylabel('Frequency')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            for product in range(num_products):
                axes[1].hist(trained_fg[:, product], alpha=0.7,
                           label=f'Product {product+1}', bins=20)
            axes[1].set_title('Trained Policy - FG Actions')
            axes[1].set_xlabel('Order Quantity')
            axes[1].set_ylabel('Frequency')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

    else:
        # RM agent actions (single dimension)
        heuristic_rm = np.array(heuristic_actions['rm_agent']).flatten()
        trained_rm = np.array(trained_actions['rm_agent']).flatten()

        axes[0].hist(heuristic_rm, bins=30, alpha=0.7, color='red', label='Heuristic')
        axes[0].set_title('Heuristic Policy - RM Actions')
        axes[0].set_xlabel('Order Quantity')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(trained_rm, bins=30, alpha=0.7, color='blue', label='Trained')
        axes[1].set_title('Trained Policy - RM Actions')
        axes[1].set_xlabel('Order Quantity')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def create_comprehensive_evaluation(env,
                                  fg_agent,
                                  rm_agent,
                                  heuristic_policy,
                                  num_eval_episodes: int = 5,
                                  steps_per_episode: int = 100):
    """
    Run comprehensive evaluation comparing trained policy vs heuristic

    Args:
        env: Inventory environment
        fg_agent: Trained FG agent
        rm_agent: Trained RM agent
        heuristic_policy: Heuristic policy function
        num_eval_episodes: Number of evaluation episodes
        steps_per_episode: Steps per evaluation episode
    """
    print("🚀 RUNNING COMPREHENSIVE EVALUATION")
    print("="*60)

    # Store results
    trained_results = []
    heuristic_results = []

    trained_actions = {'fg_agent': [], 'rm_agent': []}
    heuristic_actions = {'fg_agent': [], 'rm_agent': []}

    # Evaluate trained policy
    print("Evaluating Trained Policy...")
    for episode in range(num_eval_episodes):
        obs = env.reset()
        episode_history = []

        for step in range(steps_per_episode):
            # Trained policy
            fg_action, _, _ = fg_agent.act(obs['fg_agent'], deterministic=True)
            rm_action, _, _ = rm_agent.act(obs['rm_agent'], deterministic=True)

            trained_actions['fg_agent'].append(fg_action)
            trained_actions['rm_agent'].append(rm_action)

            joint_action = {'fg_agent': fg_action, 'rm_agent': rm_action}
            obs, _, done, info = env.step(joint_action)
            episode_history.append(info)

            if done:
                break

        trained_results.extend(episode_history)

    # Evaluate heuristic policy
    print("Evaluating Heuristic Policy...")
    for episode in range(num_eval_episodes):
        obs = env.reset()
        episode_history = []

        for step in range(steps_per_episode):
            # Heuristic policy
            heuristic_action = heuristic_policy(obs)

            heuristic_actions['fg_agent'].append(heuristic_action['fg_agent'])
            heuristic_actions['rm_agent'].append(heuristic_action['rm_agent'])

            obs, _, done, info = env.step(heuristic_action)
            episode_history.append(info)

            if done:
                break

        heuristic_results.extend(episode_history)

    # Generate comprehensive visualizations
    print("\n📊 GENERATING VISUALIZATIONS...")

    # 1. Policy comparison
    plot_agent_behavior_comparison(heuristic_results, trained_results)

    # 2. Action distributions
    plot_action_distribution(heuristic_actions, trained_actions, "fg_agent")
    plot_action_distribution(heuristic_actions, trained_actions, "rm_agent")

    # 3. Detailed trajectory analysis (first episode)
    plot_environment_run(heuristic_results[:steps_per_episode], "Heuristic Policy - Detailed Trajectory")
    plot_environment_run(trained_results[:steps_per_episode], "Trained Policy - Detailed Trajectory")

    return trained_results, heuristic_results

# Example usage function
def run_quick_evaluation(env, fg_agent, rm_agent):
    """
    Quick evaluation function for trained agents
    """
    from utils.heuristic_policy import get_heuristic_action

    print("🔍 RUNNING QUICK EVALUATION")
    print("="*50)

    # Run a single episode with trained policy
    obs = env.reset()
    trained_history = []

    for step in range(200):  # Shorter evaluation
        fg_action, _, _ = fg_agent.act(obs['fg_agent'], deterministic=True)
        rm_action, _, _ = rm_agent.act(obs['rm_agent'], deterministic=True)

        joint_action = {'fg_agent': fg_action, 'rm_agent': rm_action}
        obs, reward, done, info = env.step(joint_action)
        trained_history.append(info)

        if done:
            break

    # Run same episode with heuristic
    obs = env.reset()
    heuristic_history = []

    for step in range(200):
        heuristic_action = get_heuristic_action(obs)
        obs, reward, done, info = env.step(heuristic_action)
        heuristic_history.append(info)

        if done:
            break

    # Compare policies
    plot_agent_behavior_comparison(heuristic_history, trained_history)

    return trained_history, heuristic_history

if __name__ == "__main__":
    # Test the evaluation functions
    from environment.inventory_env import InventoryMAMDPEnv
    from utils.heuristic_policy import get_heuristic_action

    env = InventoryMAMDPEnv()

    # Create dummy agents for testing (you would replace with your trained agents)
    from models.networks import FGActorCritic, RMActorCritic
    from agents.ppo_agent import PPOAgent

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

    # Run quick evaluation
    trained_history, heuristic_history = run_quick_evaluation(env, fg_agent, rm_agent)
