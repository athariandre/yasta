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
            policy: Policy with act(obs) -> (action, logprob, value)
            num_steps: Number of steps to collect (default: self.rollout_length)
        
        Returns:
            Dictionary containing:
            - observations: List of observations
            - actions: torch.Tensor of actions
            - logprobs: torch.Tensor of log probabilities
            - rewards: torch.Tensor of rewards
            - dones: torch.Tensor of done flags
            - values: torch.Tensor of value estimates
            - episode_info: Dict with episode statistics
        """
        if num_steps is None:
            num_steps = self.rollout_length
        
        self.reset_storage()
        
        # Start episode
        obs = env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        completed_episodes = []
        
        for step in range(num_steps):
            # Get action from policy
            action, logprob, value = policy.act(obs)
            
            # Pre-encode observation to tensor and store both
            obs_tensor = policy._obs_to_tensor(obs)
            self.observations.append(obs)
            self.obs_tensors.append(obs_tensor)
            
            # Execute action in environment
            next_obs, reward, done, info = env.step(action)
            
            # Store transition
            self.actions.append(action)
            self.logprobs.append(logprob)
            self.rewards.append(reward)
            self.dones.append(done)
            self.values.append(value)
            
            # Track episode stats
            episode_reward += reward
            episode_length += 1
            
            # Handle episode termination
            if done:
                # Store episode stats with POV-corrected outcome
                completed_episodes.append({
                    'reward': episode_reward,
                    'length': episode_length,
                    'outcome': info.get('result_pov', info.get('result', None)),  # Use POV result
                })
                
                # Reset environment
                obs = env.reset()
                episode_reward = 0.0
                episode_length = 0
            else:
                obs = next_obs
            
            self.step_count += 1
        
        # Convert to tensors (CPU only) - store pre-encoded tensors
        rollout_data = {
            'observations': self.observations,  # Keep for backward compatibility
            'obs_tensors': torch.stack(self.obs_tensors, dim=0),  # Pre-encoded tensors (batch_size, feature_size)
            'actions': torch.tensor(self.actions, dtype=torch.long),
            'logprobs': torch.tensor(self.logprobs, dtype=torch.float32),
            'rewards': torch.tensor(self.rewards, dtype=torch.float32),
            'dones': torch.tensor(self.dones, dtype=torch.float32),
            'values': torch.tensor(self.values, dtype=torch.float32),
            'episode_info': {
                'completed_episodes': completed_episodes,
                'num_completed': len(completed_episodes),
                'mean_reward': np.mean([ep['reward'] for ep in completed_episodes]) if completed_episodes else 0.0,
                'mean_length': np.mean([ep['length'] for ep in completed_episodes]) if completed_episodes else 0.0,
            }
        }
        
        return rollout_data
    
    def compute_advantages(self, rollout_data: Dict, gamma: float = None, gae_lambda: float = None):
        """
        Compute GAE (Generalized Advantage Estimation) advantages and returns.
        
        Args:
            rollout_data: Dictionary from collect()
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        
        Returns:
            Updated rollout_data with 'advantages' and 'returns' keys
        """
        if gamma is None:
            gamma = config.GAMMA
        if gae_lambda is None:
            gae_lambda = config.GAE_LAMBDA
        
        rewards = rollout_data['rewards']
        values = rollout_data['values']
        dones = rollout_data['dones']
        
        num_steps = len(rewards)
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Compute advantages using GAE
        gae = 0.0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                # TODO (PR-3): Bootstrap from critic on final next-obs when not done
                # For now, we use 0.0 as next_value for the final step
                # When implementing the real policy network, query the critic here:
                #   next_value = policy.evaluate_value(observations[t+1]) if not done[t] else 0.0
                next_value = 0.0  # Bootstrap value (will use critic in PR-3)
                next_non_terminal = 0.0
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]
            
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[t] = gae
        
        # Compute returns (advantages + values)
        returns = advantages + values
        
        rollout_data['advantages'] = advantages
        rollout_data['returns'] = returns
        
        return rollout_data
    
    def prepare_batch_data(self, rollout_data: Dict):
        """
        Prepare rollout data for PPO training (normalize advantages, detach tensors).
        
        Args:
            rollout_data: Dictionary from collect() with advantages computed
        
        Returns:
            Processed rollout_data ready for training
        """
        # Detach advantages, returns, and logprobs to prevent backprop through rollout
        advantages = rollout_data['advantages'].detach()
        returns = rollout_data['returns'].detach()
        logprobs = rollout_data['logprobs'].detach()
        values = rollout_data['values'].detach()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        rollout_data['advantages'] = advantages
        rollout_data['returns'] = returns
        rollout_data['logprobs'] = logprobs
        rollout_data['values'] = values
        
        return rollout_data
    
    def get_mini_batches(self, rollout_data: Dict, batch_size: int = None):
        """
        Split rollout data into mini-batches for PPO training.
        
        Args:
            rollout_data: Dictionary from prepare_batch_data()
            batch_size: Size of mini-batches
        
        Yields:
            Mini-batch dictionaries
        """
        if batch_size is None:
            batch_size = config.BATCH_SIZE
        
        num_steps = len(rollout_data['actions'])
        indices = np.random.permutation(num_steps)
        
        for start in range(0, num_steps, batch_size):
            end = min(start + batch_size, num_steps)
            batch_indices = indices[start:end]
            
            batch = {
                'observations': [rollout_data['observations'][i] for i in batch_indices],  # Backward compatibility
                'obs_tensors': rollout_data['obs_tensors'][batch_indices],  # Pre-encoded tensors
                'actions': rollout_data['actions'][batch_indices],
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
