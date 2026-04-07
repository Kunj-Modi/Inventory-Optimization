"""
Utility functions for the inventory management project
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List
from torch.utils.tensorboard import SummaryWriter  # Correct import location

def normalize_array(arr: np.ndarray) -> np.ndarray:
    """Normalize numpy array to [0, 1] range"""
    if np.max(arr) == np.min(arr):
        return np.zeros_like(arr)
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def denormalize_array(arr: np.ndarray, original_min: float, original_max: float) -> np.ndarray:
    """Denormalize array back to original range"""
    return arr * (original_max - original_min) + original_min

def calculate_moving_average(data: List[float], window_size: int) -> List[float]:
    """Calculate moving average of a time series"""
    if len(data) < window_size:
        return data
    return [np.mean(data[i:i+window_size]) for i in range(len(data) - window_size + 1)]

def plot_training_curves(rewards: List[float], losses: List[float], title: str = "Training Progress"):
    """Plot training rewards and losses"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot rewards
    ax1.plot(rewards)
    ax1.set_title(f'{title} - Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)

    # Plot losses
    ax2.plot(losses)
    ax2.set_title(f'{title} - Losses')
    ax2.set_xlabel('Update Step')
    ax2.set_ylabel('Loss')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def dict_to_tensor(state_dict: Dict, device: torch.device) -> torch.Tensor:
    """Convert state dictionary to flattened tensor"""
    tensors = []
    for key in sorted(state_dict.keys()):
        value = state_dict[key]
        if isinstance(value, (int, float)):
            value = [value]
        tensors.append(torch.tensor(value, dtype=torch.float32))

    return torch.cat(tensors).to(device)

class Logger:
    """Simple logger for training progress"""
    def __init__(self, log_dir: str = "/content/logs"):
        self.writer = SummaryWriter(log_dir)
        self.episode_rewards = []

    def log_episode(self, episode: int, reward: float, length: int):
        """Log episode statistics"""
        self.episode_rewards.append(reward)
        self.writer.add_scalar('Training/Episode_Reward', reward, episode)
        self.writer.add_scalar('Training/Episode_Length', length, episode)

    def log_losses(self, step: int, policy_loss: float, value_loss: float, entropy_loss: float):
        """Log training losses"""
        self.writer.add_scalar('Losses/Policy_Loss', policy_loss, step)
        self.writer.add_scalar('Losses/Value_Loss', value_loss, step)
        self.writer.add_scalar('Losses/Entropy_Loss', entropy_loss, step)

    def close(self):
        """Close the logger"""
        self.writer.close()

# Test the helpers
if __name__ == "__main__":
    # Test normalization
    test_data = np.array([1, 2, 3, 4, 5])
    normalized = normalize_array(test_data)
    print(f"Original: {test_data}")
    print(f"Normalized: {normalized}")

    # Test moving average
    moving_avg = calculate_moving_average(list(test_data), 3)
    print(f"Moving average (window=3): {moving_avg}")
