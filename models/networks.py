"""
neural network architectures with proper error handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import configs.config as config

class TimeSeriesEncoder(nn.Module):
    """WaveNet-style dilated causal convolutions for time-series feature extraction"""

    def __init__(self, input_dim: int, config_obj=None):
        super(TimeSeriesEncoder, self).__init__()

        self.cfg = config_obj if config_obj else config.config.model

        self.input_dim = input_dim
        self.hidden_dim = self.cfg.HIDDEN_DIM
        self.out_channels = self.cfg.CNN_OUT_CHANNELS
        self.kernel_size = self.cfg.CNN_KERNEL_SIZE
        self.num_layers = self.cfg.CNN_NUM_LAYERS

        # Input projection - FIXED: handle variable input dimensions
        self.input_conv = nn.Conv1d(
            in_channels=input_dim,  # This should match the feature dimension
            out_channels=self.out_channels,
            kernel_size=1
        )

        # Dilated causal convolution layers
        self.dilated_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        for i in range(self.num_layers):
            dilation = 2 ** i

            # Dilated causal convolution
            dilated_conv = nn.Conv1d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                dilation=dilation,
                padding=0
            )
            self.dilated_convs.append(dilated_conv)

            # Residual connection
            residual_conv = nn.Conv1d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=1
            )
            self.residual_convs.append(residual_conv)

            # Skip connection
            skip_conv = nn.Conv1d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=1
            )
            self.skip_convs.append(skip_conv)

        # Output projection
        self.output_conv = nn.Conv1d(
            in_channels=self.out_channels,
            out_channels=self.hidden_dim,
            kernel_size=1
        )

        # Activation functions
        self.activation = nn.ReLU()
        self.tanh = nn.Tanh()

    def _causal_pad(self, x: torch.Tensor, dilation: int) -> torch.Tensor:
        """Apply causal padding to the left of the input sequence"""
        padding = (self.kernel_size - 1) * dilation
        return F.pad(x, (padding, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the time-series encoder"""

        if x.numel() == 0:
            # Return zeros if no forecast data
            batch_size = x.shape[0] if len(x.shape) > 0 else 1
            return torch.zeros(batch_size, self.hidden_dim, device=x.device)

        # Input shape: (batch_size, seq_len, input_dim)
        # Transpose to: (batch_size, input_dim, seq_len) for Conv1d
        x = x.transpose(1, 2)

        # Input projection
        x = self.input_conv(x)
        residual = x

        skip_connections = []

        # Dilated causal convolution layers
        for i, (dilated_conv, residual_conv, skip_conv) in enumerate(zip(
            self.dilated_convs, self.residual_convs, self.skip_convs)):

            dilation = 2 ** i

            # Apply causal padding
            x_padded = self._causal_pad(x, dilation)

            # Dilated convolution
            x_out = dilated_conv(x_padded)
            x_out = self.tanh(x_out)

            # Residual connection
            residual_out = residual_conv(x_out)

            # Skip connection
            skip_out = skip_conv(x_out)
            skip_connections.append(skip_out)

            # Update for next layer
            x = x + residual_out
            residual = x_out

        # Combine skip connections
        if skip_connections:
            x = torch.stack(skip_connections).sum(dim=0)

        # Output projection
        x = self.output_conv(x)
        x = self.activation(x)

        # Global average pooling over time dimension
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)

        return x

class FGActorCritic(nn.Module):
    """FG Actor-Critic network with simplified processing for new state structure"""

    def __init__(self, state_dim: int, action_dim: int, config_obj=None, output_multiplier=0.3):
        super(FGActorCritic, self).__init__()

        self.cfg = config_obj if config_obj else config.config.model

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Simple MLP architecture - no time-series encoder for now
        # The enhanced state already has aggregated temporal features
        self.shared_mlp = nn.Sequential(
            nn.Linear(state_dim, self.cfg.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(self.cfg.HIDDEN_DIM, self.cfg.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(self.cfg.HIDDEN_DIM, self.cfg.HIDDEN_DIM),
            nn.ReLU()
        )

        # Actor head - outputs mean and log_std for each action dimension
        self.actor_mean = nn.Linear(self.cfg.HIDDEN_DIM, action_dim)
        self.actor_log_std = nn.Linear(self.cfg.HIDDEN_DIM, action_dim)

        # Critic head - outputs state value
        self.critic = nn.Linear(self.cfg.HIDDEN_DIM, 1)

        self.output_multiplier = output_multiplier

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.1)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the FG agent network"""

        # Direct MLP processing - the state already contains enhanced features
        shared_features = self.shared_mlp(state)

        # Actor outputs
        mu = self.actor_mean(shared_features) * self.output_multiplier
        log_std = self.actor_log_std(shared_features)

        # Ensure positive standard deviation using softplus
        sigma = F.softplus(log_std) + 1e-4

        # Critic output
        value = self.critic(shared_features)

        return mu, sigma, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action from the policy distribution"""
        mu, sigma, value = self.forward(state)

        if deterministic:
            action = mu
            log_prob = torch.zeros_like(mu)
        else:
            # Create normal distribution
            dist = torch.distributions.Normal(mu, sigma)
            # Sample action
            action = dist.rsample()
            # Calculate log probability
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            # Ensure actions are non-negative
            action = F.softplus(action)

        return action, log_prob, value

class RMActorCritic(nn.Module):
    """RM Actor-Critic network - simple MLP"""

    def __init__(self, state_dim: int, action_dim: int, config_obj=None):
        super(RMActorCritic, self).__init__()

        self.cfg = config_obj if config_obj else config.config.model

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Simple MLP architecture
        self.shared_mlp = nn.Sequential(
            nn.Linear(state_dim, self.cfg.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(self.cfg.HIDDEN_DIM, self.cfg.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(self.cfg.HIDDEN_DIM, self.cfg.HIDDEN_DIM),
            nn.ReLU()
        )

        # Actor head
        self.actor_mean = nn.Linear(self.cfg.HIDDEN_DIM, action_dim)
        self.actor_log_std = nn.Linear(self.cfg.HIDDEN_DIM, action_dim)

        # Critic head
        self.critic = nn.Linear(self.cfg.HIDDEN_DIM, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the RM agent network"""
        # Shared MLP processing
        shared_features = self.shared_mlp(state)

        # Actor outputs
        mu = self.actor_mean(shared_features)
        log_std = self.actor_log_std(shared_features)

        # Ensure positive standard deviation
        sigma = F.softplus(log_std) + 1e-4

        # Critic output
        value = self.critic(shared_features)

        return mu, sigma, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action from the policy distribution"""
        mu, sigma, value = self.forward(state)

        if deterministic:
            action = mu
            log_prob = torch.zeros_like(mu)
        else:
            dist = torch.distributions.Normal(mu, sigma)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            # Ensure actions are non-negative
            action = F.softplus(action)

        return action, log_prob, value
