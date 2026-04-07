"""
Visualization utilities for the inventory management environment
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict

def plot_environment_run(history: List[Dict], title: str = "Inventory Management Simulation"):
    """
    Plot the results of an environment run

    Args:
        history: List of info dictionaries from each step
        title: Plot title
    """
    steps = len(history)
    if steps == 0:
        return

    # Extract data
    fg_inventory_history = np.array([step['fg_inventory'] for step in history])
    rm_inventory_history = np.array([step['rm_inventory'] for step in history])
    lost_sales_history = np.array([step['lost_sales'] for step in history])
    total_costs = np.array([step['total_cost'] for step in history])

    num_products = fg_inventory_history.shape[1]

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)

    # Plot 1: FG Inventory levels
    for product in range(num_products):
        axes[0, 0].plot(fg_inventory_history[:, product], label=f'Product {product+1}')
    axes[0, 0].set_title('Finished Goods Inventory')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Inventory Level')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot 2: RM Inventory
    axes[0, 1].plot(rm_inventory_history)
    axes[0, 1].set_title('Raw Material Inventory')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Inventory Level')
    axes[0, 1].grid(True)

    # Plot 3: Lost Sales
    for product in range(num_products):
        axes[0, 2].plot(lost_sales_history[:, product], label=f'Product {product+1}')
    axes[0, 2].set_title('Lost Sales')
    axes[0, 2].set_xlabel('Step')
    axes[0, 2].set_ylabel('Lost Sales Quantity')
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # Plot 4: Total Cost
    axes[1, 0].plot(total_costs)
    axes[1, 0].set_title('Total System Cost')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Cost')
    axes[1, 0].grid(True)

    # Plot 5: Cumulative Cost
    cumulative_cost = np.cumsum(total_costs)
    axes[1, 1].plot(cumulative_cost)
    axes[1, 1].set_title('Cumulative System Cost')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Cumulative Cost')
    axes[1, 1].grid(True)

    # Print summary statistics
    print(f"\n=== Simulation Summary ({steps} steps) ===")
    print(f"Total Cost: {np.sum(total_costs):.2f}")
    print(f"Average Cost per Step: {np.mean(total_costs):.2f}")
    print(f"Total Lost Sales: {np.sum(lost_sales_history):.2f}")
    print(f"Average FG Inventory: {np.mean(fg_inventory_history):.2f}")
    print(f"Average RM Inventory: {np.mean(rm_inventory_history):.2f}")

# Test the visualization with a short simulation
if __name__ == "__main__":
    from environment.inventory_env import InventoryMAMDPEnv

    env = InventoryMAMDPEnv()
    obs = env.reset()

    history = []
    for step in range(100):  # Short test run
        # Simple heuristic: order based on current inventory
        fg_action = np.maximum(0, 10 - env.fg_inventory)
        rm_action = np.array([max(0, 20 - env.rm_inventory)])

        joint_action = {'fg_agent': fg_action, 'rm_agent': rm_action}
        obs, reward, done, info = env.step(joint_action)
        history.append(info)

        if done:
            break

    plot_environment_run(history, "Basic Heuristic Policy Test")
