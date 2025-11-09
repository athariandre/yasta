"""
Rollout collector for PPO training.

This module handles collecting trajectories from the environment for PPO updates.
All data is stored as CPU tensors for compatibility with the training loop.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple

from training.config import config


class RolloutCollector:
    """
    Collect rollouts (trajectories) for PPO training.
    
    Stores:
    - Observations
    - Actions
    - Log probabilities
    - Rewards
    - Dones
    - Value estimates
    
    All data is stored as CPU tensors and handles episode boundaries correctly.
    """
    
    def __init__(self, rollout_length: int = None, num_envs: int = 1):
        """
        Initialize rollout collector.
        
        Args:
            rollout_length: Number of steps to collect per rollout
            num_envs: Number of parallel environments (currently only supports 1)
        """
        self.rollout_length = rollout_length if rollout_length else config.ROLLOUT_LENGTH
        self.num_envs = num_envs
        
        # Preallocate storage for efficiency
        self.reset_storage()
    
    def reset_storage(self):
        """Reset storage buffers for new rollout collection."""
        self.observations = []
        self.obs_tensors = []  # Store pre-encoded tensors
        self.actions = []
        self.action_masks = []  # Store action masks
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.episode_lengths = []
        self.episode_rewards = []
        
        self.step_count = 0
    
    def collect(self, env, policy, num_steps: int = None):
        """
        Collect a rollout from the environment using the policy.
        
        Args:
            env: Environment instance
            policy: Policy with act(obs, action_mask) -> (action, logprob, value)
            num_steps: Number of steps to collect (default: self.rollout_length)
        
        Returns:
            Dictionary containing:
            - obs: List of raw observation dicts
            - actions: np.ndarray of actions
            - action_masks: np.ndarray of action masks
            - logprobs: np.ndarray of log probabilities
            - values: np.ndarray of value estimates
            - rewards: np.ndarray of rewards
            - dones: np.ndarray of done flags
            - final_obs: Single observation dict for bootstrapping
            - episode_info: Dict with episode statistics including completed_episodes
        """
        if num_steps is None:
            num_steps = self.rollout_length
        
        self.reset_storage()
        
        # Start episode
        obs = env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        completed_episodes = []
        final_obs = obs
        
        for step in range(num_steps):
            # Compute action mask using policy's mask function
            action_mask = policy._compute_legal_actions_mask(obs)
            
            # Get action from policy with action mask
            action, logprob, value = policy.act(obs, action_mask)
            
            # Store observation (raw dict)
            self.observations.append(obs)
            
            # Execute action in environment
            next_obs, reward, done, info = env.step(action)
            
            # Store transition (as lists, will convert to numpy later)
            self.actions.append(action)
            self.action_masks.append(action_mask)
            self.logprobs.append(logprob)
            self.values.append(value)
            self.rewards.append(reward)
            self.dones.append(done)
            
            # Track episode stats
            episode_reward += reward
            episode_length += 1
            
            # Handle episode termination
            if done:
                # Store episode stats with POV-corrected outcome
                completed_episodes.append({
                    'reward': episode_reward,
                    'length': episode_length,
                    'outcome': info.get('result_pov', info.get('result', None)),
                })
                
                # Reset environment and continue collecting within same rollout
                obs = env.reset()
                episode_reward = 0.0
                episode_length = 0
            else:
                obs = next_obs
            
            # Track final observation for bootstrapping
            final_obs = obs
            
            self.step_count += 1
        
        # Convert to numpy arrays (CPU only, no tensors during rollout)
        rollout_data = {
            'obs': self.observations,  # List of raw observation dicts
            'actions': np.array(self.actions, dtype=np.int64),
            'action_masks': np.array(self.action_masks, dtype=np.float32),
            'logprobs': np.array(self.logprobs, dtype=np.float32),
            'values': np.array(self.values, dtype=np.float32),
            'rewards': np.array(self.rewards, dtype=np.float32),
            'dones': np.array(self.dones, dtype=np.float32),
            'final_obs': final_obs,  # Single observation dict for bootstrapping
            'episode_info': {
                'completed_episodes': completed_episodes,
                'num_completed': len(completed_episodes),
                'mean_reward': np.mean([ep['reward'] for ep in completed_episodes]) if completed_episodes else 0.0,
                'mean_length': np.mean([ep['length'] for ep in completed_episodes]) if completed_episodes else 0.0,
            }
        }
        
        return rollout_data
    
    def compute_advantages(self, rollout_data: Dict, gamma: float = None, gae_lambda: float = None, policy=None):
        """
        Compute GAE (Generalized Advantage Estimation) advantages and returns.
        
        Implements:
            delta_t = r_t + gamma * V_{t+1} * (1 - done_{t+1}) - V_t
            A_t = delta_t + gamma * lambda * (1 - done_{t+1}) * A_{t+1}
        
        Bootstrapping:
            V_bootstrap = policy.get_value(final_obs) if final step is not terminal
            V_bootstrap = 0 if final step is terminal (done=True)
        
        Args:
            rollout_data: Dictionary from collect()
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            policy: Policy for bootstrapping value on final step
        
        Returns:
            Updated rollout_data with 'advantages' and 'returns' keys (numpy arrays)
        """
        if gamma is None:
            gamma = config.GAMMA
        if gae_lambda is None:
            gae_lambda = config.GAE_LAMBDA
        
        rewards = rollout_data['rewards']
        values = rollout_data['values']
        dones = rollout_data['dones']
        
        num_steps = len(rewards)
        advantages = np.zeros_like(rewards)
        
        # Compute bootstrap value for final step
        # If rollout ends in terminal state (done=True), bootstrap_value = 0
        # If rollout ends in non-terminal state (done=False), bootstrap from critic
        final_done = dones[-1] if num_steps > 0 else True
        if final_done:
            bootstrap_value = 0.0
        else:
            if policy is not None:
                bootstrap_value = policy.get_value(rollout_data['final_obs'])
            else:
                bootstrap_value = 0.0
        
        # Compute advantages using GAE in reverse order
        gae = 0.0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                # Last step uses bootstrap value
                next_value = bootstrap_value
                next_non_terminal = 1.0 - final_done
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t + 1]  # Fixed: use dones[t+1], not dones[t]
            
            # TD error: delta_t = r_t + gamma * V_{t+1} * (1 - done_{t+1}) - V_t
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            
            # GAE: A_t = delta_t + gamma * lambda * (1 - done_{t+1}) * A_{t+1}
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[t] = gae
        
        # Compute returns: returns = advantages + values
        returns = advantages + values
        
        rollout_data['advantages'] = advantages
        rollout_data['returns'] = returns
        
        return rollout_data
    
    def prepare_batch_data(self, rollout_data: Dict):
        """
        Prepare rollout data for PPO training.
        
        Converts everything to torch tensors and normalizes advantages:
            advantages = (advantages - mean) / (std + 1e-8)
        
        Produces:
            - obs_tensors: shape (T, feature_size)
            - actions: shape (T,)
            - action_masks: shape (T, num_actions)
            - logprobs: shape (T,)
            - values: shape (T,)
            - returns: shape (T,)
            - advantages: normalized
        
        Args:
            rollout_data: Dictionary from collect() with advantages computed
        
        Returns:
            Processed rollout_data ready for training (all numpy arrays converted to tensors)
        """
        # Convert observations to tensors - need to encode obs list
        # Use policy's encode_obs if available, otherwise use _obs_to_tensor
        # For now, we'll need policy access - this will be handled by trainer
        # For prepare_batch_data, we'll store as-is and convert in get_mini_batches
        
        # Convert numpy arrays to torch tensors
        advantages = torch.from_numpy(rollout_data['advantages']).float()
        returns = torch.from_numpy(rollout_data['returns']).float()
        logprobs = torch.from_numpy(rollout_data['logprobs']).float()
        values = torch.from_numpy(rollout_data['values']).float()
        actions = torch.from_numpy(rollout_data['actions']).long()
        action_masks = torch.from_numpy(rollout_data['action_masks']).float()
        rewards = torch.from_numpy(rollout_data['rewards']).float()
        dones = torch.from_numpy(rollout_data['dones']).float()
        
        # Normalize advantages: (advantages - mean) / (std + 1e-8)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update rollout_data with tensors
        rollout_data['advantages'] = advantages
        rollout_data['returns'] = returns
        rollout_data['logprobs'] = logprobs
        rollout_data['values'] = values
        rollout_data['actions'] = actions
        rollout_data['action_masks'] = action_masks
        rollout_data['rewards'] = rewards
        rollout_data['dones'] = dones
        
        return rollout_data
    
    def get_mini_batches(self, rollout_data: Dict, batch_size: int = None, policy=None):
        """
        Split rollout data into mini-batches for PPO training.
        
        Shuffles indices and yields mini-batches with matching ordering for every tensor.
        Encodes observations to tensors on-the-fly using policy.encode_obs().
        
        Args:
            rollout_data: Dictionary from prepare_batch_data()
            batch_size: Size of mini-batches
            policy: Policy object for encoding observations (required)
        
        Yields:
            Mini-batch dictionaries with:
            - obs_tensors: (batch_size, feature_size)
            - actions: (batch_size,)
            - action_masks: (batch_size, num_actions)
            - logprobs: (batch_size,)
            - values: (batch_size,)
            - returns: (batch_size,)
            - advantages: (batch_size,)
        """
        if batch_size is None:
            batch_size = config.BATCH_SIZE
        
        num_steps = len(rollout_data['actions'])
        indices = np.random.permutation(num_steps)
        
        for start in range(0, num_steps, batch_size):
            end = min(start + batch_size, num_steps)
            batch_indices = indices[start:end]
            
            # Encode observations using policy
            obs_list = [rollout_data['obs'][i] for i in batch_indices]
            if policy is not None:
                # Use policy.encode_obs() to encode observations
                obs_tensors = torch.stack([policy.encode_obs(obs) for obs in obs_list], dim=0)
            else:
                # Fallback: just use empty tensors (should not happen)
                obs_tensors = torch.zeros((len(batch_indices), 1))
            
            batch = {
                'obs_tensors': obs_tensors,
                'actions': rollout_data['actions'][batch_indices],
                'action_masks': rollout_data['action_masks'][batch_indices],
                'logprobs': rollout_data['logprobs'][batch_indices],
                'advantages': rollout_data['advantages'][batch_indices],
                'returns': rollout_data['returns'][batch_indices],
                'values': rollout_data['values'][batch_indices],
            }
            
            yield batch


class DummyPolicy:
    """
    Placeholder policy for PR-2.
    
    This will be replaced with a real neural network in PR-3.
    The interface is designed to be compatible with PPO training.
    """
    
    def __init__(self, action_space: int = 4):
        """
        Initialize dummy policy.
        
        Args:
            action_space: Number of discrete actions
        """
        self.action_space = action_space
    
    def act(self, obs: Dict):
        """
        Select an action given observation.
        
        Args:
            obs: Observation dictionary
        
        Returns:
            Tuple of (action, logprob, value)
            - action: int (0-3)
            - logprob: float (fake log probability)
            - value: float (fake value estimate)
        """
        # Random action
        action = np.random.randint(0, self.action_space)
        
        # Fake log probability (uniform distribution)
        logprob = -np.log(self.action_space)
        
        # Fake value estimate (random)
        value = np.random.randn() * 0.1
        
        return action, logprob, value
    
    def evaluate_actions(self, observations: List[Dict], actions: torch.Tensor):
        """
        Evaluate actions given observations (for PPO update).
        
        Args:
            observations: List of observation dictionaries
            actions: Tensor of actions taken
        
        Returns:
            Tuple of (logprobs, values, entropy)
            - logprobs: Tensor of log probabilities
            - values: Tensor of value estimates
            - entropy: Tensor of policy entropy
        """
        batch_size = len(observations)
        
        # Fake outputs
        logprobs = torch.full((batch_size,), -np.log(self.action_space), dtype=torch.float32)
        values = torch.randn(batch_size, dtype=torch.float32) * 0.1
        entropy = torch.full((batch_size,), np.log(self.action_space), dtype=torch.float32)
        
        return logprobs, values, entropy
    
    def save(self, path: str):
        """Save policy (dummy implementation)."""
        print(f"[DummyPolicy] Fake save to {path}")
    
    def load(self, path: str):
        """Load policy (dummy implementation)."""
        print(f"[DummyPolicy] Fake load from {path}")
