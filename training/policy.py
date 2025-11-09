"""
Neural network policy for PPO training.

This module implements a CPU-only PyTorch Actor-Critic network with:
- Shared encoder for processing observations
- Actor head (policy) for action selection
- Critic head (value function) for state value estimation

All operations are CPU-only to comply with competition constraints.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

from training.config import config


class ActorCriticPolicy(nn.Module):
    """
    Actor-Critic neural network for Tron agent.
    
    Architecture:
    1. Shared encoder: Processes observation into latent vector
    2. Actor head: Outputs action logits (categorical distribution)
    3. Critic head: Outputs value estimate (scalar)
    
    Design constraints:
    - CPU-only (no CUDA)
    - Small model size (<5MB)
    - Fast inference (<5ms per step)
    - Uses only MLPs (no convolutions for speed)
    """
    
    def __init__(self, 
                 board_h: int = None,
                 board_w: int = None,
                 num_actions: int = None,
                 latent_size: int = None,
                 device: str = None):
        """
        Initialize Actor-Critic policy.
        
        Args:
            board_h: Board height (uses config if None)
            board_w: Board width (uses config if None)
            num_actions: Number of discrete actions (uses config if None)
            latent_size: Size of latent feature vector (uses config if None)
            device: Device to use (uses config if None, should be 'cpu')
        """
        super().__init__()
        
        self.board_h = board_h if board_h is not None else config.BOARD_H
        self.board_w = board_w if board_w is not None else config.BOARD_W
        self.num_actions = num_actions if num_actions is not None else config.NUM_ACTIONS
        self.latent_size = latent_size if latent_size is not None else config.LATENT_SIZE
        self.device = torch.device(device if device is not None else config.DEVICE)
        
        # Calculate input sizes
        self.board_size = self.board_h * self.board_w
        # Inputs: board (H*W) + my_head (2) + opp_head (2) + boosts (2) + turns (1) + directions (2) + alive (2)
        self.feature_size = self.board_size + 2 + 2 + 2 + 1 + 2 + 2
        
        # Shared encoder
        # Use 2-layer MLP to keep model small and fast
        self.encoder = nn.Sequential(
            nn.Linear(self.feature_size, self.latent_size),
            nn.ReLU(),
            nn.Linear(self.latent_size, self.latent_size),
            nn.ReLU(),
        )
        
        # Actor head (policy)
        # Small MLP for action logits
        self.actor = nn.Sequential(
            nn.Linear(self.latent_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions),
        )
        
        # Critic head (value function)
        # Small MLP for value estimate
        self.critic = nn.Sequential(
            nn.Linear(self.latent_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        
        # Move to device (CPU only)
        self.to(self.device)
        
        # Print model size
        self._print_model_info()
    
    def _print_model_info(self):
        """Print model architecture and size information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Estimate model size (4 bytes per float32 parameter)
        model_size_mb = (total_params * 4) / (1024 * 1024)
        
        print(f"[Policy] Initialized Actor-Critic network")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Estimated size: {model_size_mb:.2f} MB")
        print(f"  Device: {self.device}")
        
        if model_size_mb > 5.0:
            print(f"  WARNING: Model size exceeds 5MB limit!")
    
    def _obs_to_tensor(self, obs: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Convert observation dictionary to input tensor.
        
        Args:
            obs: Observation dictionary with keys:
                - board: (H, W) float32
                - my_head: (2,) float32
                - opp_head: (2,) float32
                - boosts: (2,) float32
                - turns: (1,) float32
                - directions: (2,) float32
                - alive: (2,) float32
        
        Returns:
            Tensor of shape (feature_size,) on CPU
        """
        # Flatten board
        board_flat = obs['board'].flatten()
        
        # Concatenate all features
        features = np.concatenate([
            board_flat,
            obs['my_head'],
            obs['opp_head'],
            obs['boosts'],
            obs['turns'],
            obs['directions'],
            obs['alive'],
        ]).astype(np.float32)
        
        # Convert to tensor (CPU only)
        return torch.from_numpy(features).to(self.device)
    
    def _obs_batch_to_tensor(self, obs_list: List[Dict[str, np.ndarray]]) -> torch.Tensor:
        """
        Convert list of observations to batched tensor.
        
        Args:
            obs_list: List of observation dictionaries
        
        Returns:
            Tensor of shape (batch_size, feature_size) on CPU
        """
        batch = [self._obs_to_tensor(obs) for obs in obs_list]
        return torch.stack(batch, dim=0)
    
    def forward(self, obs_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            obs_tensor: Input tensor of shape (batch_size, feature_size) or (feature_size,)
        
        Returns:
            Tuple of (action_logits, values)
            - action_logits: (batch_size, num_actions) or (num_actions,)
            - values: (batch_size, 1) or (1,)
        """
        # Shared encoder
        latent = self.encoder(obs_tensor)
        
        # Actor and critic heads
        action_logits = self.actor(latent)
        values = self.critic(latent)
        
        return action_logits, values
    
    def act(self, obs: Dict[str, np.ndarray]) -> Tuple[int, float, float]:
        """
        Select an action given observation.
        
        Args:
            obs: Observation dictionary
        
        Returns:
            Tuple of (action, logprob, value)
            - action: int (0-3)
            - logprob: float (log probability of selected action)
            - value: float (value estimate)
        """
        with torch.no_grad():
            # Convert observation to tensor
            obs_tensor = self._obs_to_tensor(obs).unsqueeze(0)  # Add batch dimension
            
            # Forward pass
            action_logits, values = self.forward(obs_tensor)
            
            # Sample action from categorical distribution
            action_probs = F.softmax(action_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            
            # Get log probability
            logprob = action_dist.log_prob(action)
            
            # Extract scalar values
            action = action.item()
            logprob = logprob.item()
            value = values.squeeze().item()
        
        return action, logprob, value
    
    def evaluate_actions(self, 
                        observations: List[Dict[str, np.ndarray]], 
                        actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions given observations (for PPO update).
        
        Args:
            observations: List of observation dictionaries
            actions: Tensor of actions taken, shape (batch_size,)
        
        Returns:
            Tuple of (logprobs, values, entropy)
            - logprobs: Tensor of log probabilities, shape (batch_size,)
            - values: Tensor of value estimates, shape (batch_size,)
            - entropy: Tensor of policy entropy, shape (batch_size,)
        """
        # Convert observations to tensor
        obs_tensor = self._obs_batch_to_tensor(observations)
        
        # Forward pass
        action_logits, values = self.forward(obs_tensor)
        
        # Compute action probabilities and distribution
        action_probs = F.softmax(action_logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        
        # Get log probabilities for given actions
        logprobs = action_dist.log_prob(actions)
        
        # Get entropy
        entropy = action_dist.entropy()
        
        # Squeeze values to match expected shape
        values = values.squeeze(-1)
        
        return logprobs, values, entropy
    
    def get_value(self, obs: Dict[str, np.ndarray]) -> float:
        """
        Get value estimate for a single observation.
        
        Args:
            obs: Observation dictionary
        
        Returns:
            Value estimate (float)
        """
        with torch.no_grad():
            obs_tensor = self._obs_to_tensor(obs).unsqueeze(0)
            _, values = self.forward(obs_tensor)
            return values.squeeze().item()
