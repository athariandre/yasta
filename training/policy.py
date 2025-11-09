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
                 device: str = None,
                 verbose: bool = False):
        """
        Initialize Actor-Critic policy.
        
        Args:
            board_h: Board height (uses config if None)
            board_w: Board width (uses config if None)
            num_actions: Number of discrete actions (uses config if None)
            latent_size: Size of latent feature vector (uses config if None)
            device: Device to use (uses config if None, should be 'cpu')
            verbose: Whether to print model info (default: False)
        """
        super().__init__()
        
        self.board_h = board_h if board_h is not None else config.BOARD_H
        self.board_w = board_w if board_w is not None else config.BOARD_W
        self.num_actions = num_actions if num_actions is not None else config.NUM_ACTIONS
        self.latent_size = latent_size if latent_size is not None else config.LATENT_SIZE
        self.device = torch.device(device if device is not None else config.DEVICE)
        self.verbose = verbose
        
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
        if verbose:
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
    
    def _obs_to_tensor(self, obs: Dict[str, np.ndarray], normalize: bool = True) -> torch.Tensor:
        """
        Convert observation dictionary to input tensor with normalization.
        
        Args:
            obs: Observation dictionary with keys:
                - board: (H, W) float32 (values in {0, 1, 2})
                - my_head: (2,) float32 (x, y coordinates)
                - opp_head: (2,) float32 (x, y coordinates)
                - boosts: (2,) float32 (remaining boosts)
                - turns: (1,) float32 (current turn)
                - directions: (2,) float32 (encoded 0-3)
                - alive: (2,) float32 (binary 0/1)
            normalize: Whether to normalize features (default: True)
        
        Returns:
            Tensor of shape (feature_size,) on CPU
        """
        # Flatten board and normalize to [0, 1]
        board_flat = obs['board'].flatten()
        if normalize:
            # Board values are {0, 1, 2}, normalize to [0, 1]
            board_flat = board_flat / 2.0
            
            # Normalize head positions by board dimensions
            my_head_norm = obs['my_head'] / np.array([self.board_w, self.board_h], dtype=np.float32)
            opp_head_norm = obs['opp_head'] / np.array([self.board_w, self.board_h], dtype=np.float32)
            
            # Normalize boosts by max boosts
            boosts_norm = obs['boosts'] / config.MAX_BOOST
            
            # Normalize turns by max turns
            turns_norm = obs['turns'] / config.MAX_TURNS
            
            # Normalize directions to [0, 1]
            directions_norm = obs['directions'] / 3.0
            
            # Alive is already binary [0, 1]
            alive_norm = obs['alive']
            
            # Concatenate all features
            features = np.concatenate([
                board_flat,
                my_head_norm,
                opp_head_norm,
                boosts_norm,
                turns_norm,
                directions_norm,
                alive_norm,
            ]).astype(np.float32)
        else:
            # No normalization (for backward compatibility)
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
    
    def encode_obs(self, obs):
        """
        Helper for rollout to request encoded observations.
        
        Args:
            obs: Observation dictionary
        
        Returns:
            Encoded observation tensor
        """
        return self._obs_to_tensor(obs)
    
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
    
    def _compute_legal_actions_mask(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute legal action mask from observation based on board boundaries and collisions.
        
        Args:
            obs: Observation dictionary with board state
        
        Returns:
            Boolean mask (1=legal, 0=illegal) of shape (num_actions,)
        """
        # Initialize all actions as legal
        mask = np.ones(self.num_actions, dtype=np.float32)
        
        # Get current head position
        my_head = obs['my_head']  # (x, y) format
        x, y = int(my_head[0]), int(my_head[1])
        
        # Get board dimensions
        board = obs['board']
        board_h, board_w = board.shape
        
        # Direction mappings: 0=up, 1=right, 2=down, 3=left
        # Board coordinates: y increases downward, x increases rightward
        directions = [
            (0, -1),  # up: y decreases
            (1, 0),   # right: x increases
            (0, 1),   # down: y increases
            (-1, 0),  # left: x decreases
        ]
        
        # Check each action for legality
        for action_idx, (dx, dy) in enumerate(directions):
            nx, ny = x + dx, y + dy
            
            # Check bounds
            if nx < 0 or nx >= board_w or ny < 0 or ny >= board_h:
                mask[action_idx] = 0.0
                continue
            
            # Check collision with non-empty cells (walls, trails)
            # board[ny][nx] == 0 means empty, != 0 means occupied
            if board[ny][nx] != 0:
                mask[action_idx] = 0.0
        
        # Ensure at least one action is legal (safety fallback)
        if mask.sum() == 0:
            # If all moves are illegal, allow all (agent will have to pick best bad option)
            mask = np.ones(self.num_actions, dtype=np.float32)
        
        return mask
    
    def act(self, obs: Dict[str, np.ndarray], action_mask: np.ndarray = None) -> Tuple[int, float, float]:
        """
        Select an action given observation with optional action masking.
        
        Args:
            obs: Observation dictionary
            action_mask: Optional boolean mask (1=valid, 0=invalid) of shape (num_actions,)
                        If None, will attempt to compute legal actions (simplified heuristic)
        
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
            action_logits = action_logits.squeeze(0)  # Remove batch dimension
            
            # Apply action mask
            if action_mask is None:
                # Compute legal actions heuristically if not provided
                action_mask = self._compute_legal_actions_mask(obs)
            
            mask_tensor = torch.from_numpy(action_mask.astype(np.float32)).to(self.device)
            # Set logits of invalid actions to very negative value
            action_logits = torch.where(mask_tensor > 0, action_logits, torch.tensor(-1e8, dtype=torch.float32, device=self.device))
            
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
                        obs_tensors: torch.Tensor,
                        actions: torch.Tensor,
                        action_masks: torch.Tensor = None,
                        old_values: torch.Tensor = None,
                        old_logprobs: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions given observation tensors (for PPO update).
        
        Args:
            obs_tensors: Tensor of observations, shape (batch_size, feature_size)
            actions: Tensor of actions taken, shape (batch_size,)
            action_masks: Optional tensor of action masks, shape (batch_size, num_actions)
                         If None, no masking is applied (assumes all actions were legal during rollout)
            old_values: Optional tensor of old value estimates for value clipping
            old_logprobs: Optional tensor of old log probabilities for KL computation
        
        Returns:
            Tuple of (logprobs, values, entropy, kl_div)
            - logprobs: Tensor of log probabilities, shape (batch_size,)
            - values: Tensor of value estimates, shape (batch_size,)
            - entropy: Tensor of policy entropy, shape (batch_size,)
            - kl_div: Scalar tensor of approximate KL divergence (for early stopping)
        """
        # Ensure obs_tensors has correct shape
        if obs_tensors.dim() == 1:
            obs_tensors = obs_tensors.unsqueeze(0)
        
        # Forward pass
        action_logits, values = self.forward(obs_tensors)
        
        # Apply action masks
        # If no masks provided, assume all actions are legal (fallback)
        if action_masks is None:
            # Fallback: treat all actions as legal for now
            # Later PRs can improve this by recomputing masks from observations
            action_masks = torch.ones_like(action_logits, dtype=torch.float32)
        
        # Apply masking to logits
        action_logits = torch.where(action_masks > 0, action_logits, 
                                   torch.tensor(-1e8, dtype=torch.float32, device=self.device))
        
        # Compute action probabilities and distribution
        action_probs = F.softmax(action_logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        
        # Get log probabilities for given actions
        logprobs = action_dist.log_prob(actions)
        
        # Get entropy
        entropy = action_dist.entropy()
        
        # Squeeze values to match expected shape
        values = values.squeeze(-1)
        
        # Compute approximate KL divergence
        if old_logprobs is not None:
            # KL divergence: KL = old_log_prob - new_log_prob
            # Clamp to avoid NaN from unstable samples
            kl = torch.clamp(old_logprobs - logprobs, min=-1e6, max=1e6)
            kl_div = kl.mean()
        else:
            kl_div = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        
        return logprobs, values, entropy, kl_div
    
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
