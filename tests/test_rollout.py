"""
Unit tests for the rollout collector and GAE computation.

Tests cover:
- GAE computation for terminal episodes
- GAE computation with bootstrapping
- Advantage normalization
- Mini-batch generation
"""

import pytest
import numpy as np
import torch
from training.rollout import RolloutCollector
from training.policy import ActorCriticPolicy
from training.config import config


class MockPolicy:
    """Mock policy for testing rollout collection."""
    
    def __init__(self):
        self.device = torch.device('cpu')
        self.feature_size = 100
    
    def _compute_legal_actions_mask(self, obs):
        """Return all actions legal."""
        return np.ones(4, dtype=np.float32)
    
    def act(self, obs, action_mask=None):
        """Return random action."""
        action = np.random.randint(0, 4)
        logprob = -np.log(4.0)
        value = 0.0
        return action, logprob, value
    
    def get_value(self, obs):
        """Return fixed bootstrap value."""
        return 1.0
    
    def encode_obs(self, obs):
        """Return dummy tensor."""
        return torch.zeros(self.feature_size)


@pytest.fixture
def collector():
    """Create a rollout collector instance."""
    return RolloutCollector(rollout_length=10)


@pytest.fixture
def mock_policy():
    """Create a mock policy."""
    return MockPolicy()


@pytest.mark.unit
def test_gae_terminal(collector, mock_policy):
    """Test GAE computation for a terminal episode (no bootstrap)."""
    # Create rollout data with terminal episode
    rollout_data = {
        'rewards': np.array([1.0], dtype=np.float32),
        'values': np.array([0.5], dtype=np.float32),
        'dones': np.array([1.0], dtype=np.float32),  # Terminal
        'final_obs': None,
    }
    
    # Compute advantages with gamma=0.99, lambda=0.95
    rollout_data = collector.compute_advantages(
        rollout_data, 
        gamma=0.99, 
        gae_lambda=0.95, 
        policy=mock_policy
    )
    
    # For terminal step: delta = reward + 0 * next_value - value = 1.0 - 0.5 = 0.5
    # advantage = delta (no GAE continuation for terminal)
    # return = advantage + value = 0.5 + 0.5 = 1.0
    expected_advantage = 0.5
    expected_return = 1.0
    
    assert np.isclose(rollout_data['advantages'][0], expected_advantage, atol=1e-5), \
        f"Advantage: {rollout_data['advantages'][0]} vs {expected_advantage}"
    assert np.isclose(rollout_data['returns'][0], expected_return, atol=1e-5), \
        f"Return: {rollout_data['returns'][0]} vs {expected_return}"


@pytest.mark.unit
def test_gae_bootstrap(collector, mock_policy):
    """Test GAE computation with bootstrapping (non-terminal)."""
    # Create rollout data with non-terminal episode
    rollout_data = {
        'rewards': np.array([1.0], dtype=np.float32),
        'values': np.array([0.5], dtype=np.float32),
        'dones': np.array([0.0], dtype=np.float32),  # Non-terminal
        'final_obs': {},  # Dummy obs
    }
    
    # Compute advantages - mock_policy.get_value() returns 1.0
    rollout_data = collector.compute_advantages(
        rollout_data, 
        gamma=0.99, 
        gae_lambda=0.95, 
        policy=mock_policy
    )
    
    # For non-terminal: delta = reward + gamma * bootstrap - value
    # delta = 1.0 + 0.99 * 1.0 - 0.5 = 1.49
    # advantage = delta
    # return = advantage + value = 1.49 + 0.5 = 1.99
    expected_delta = 1.0 + 0.99 * 1.0 - 0.5
    expected_advantage = expected_delta
    expected_return = expected_advantage + 0.5
    
    assert np.isclose(rollout_data['advantages'][0], expected_advantage, atol=1e-5), \
        f"Advantage: {rollout_data['advantages'][0]} vs {expected_advantage}"
    assert np.isclose(rollout_data['returns'][0], expected_return, atol=1e-5), \
        f"Return: {rollout_data['returns'][0]} vs {expected_return}"


@pytest.mark.unit
def test_advantage_normalization(collector):
    """Test that advantage normalization produces mean≈0, std≈1."""
    # Create rollout data with diverse advantages
    rollout_data = {
        'advantages': np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        'returns': np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        'logprobs': np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        'values': np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        'actions': np.array([0, 1, 2, 3, 0], dtype=np.int64),
        'action_masks': np.ones((5, 4), dtype=np.float32),
        'rewards': np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        'dones': np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        'obs': [{}] * 5,
    }
    
    # Prepare batch data (normalizes advantages)
    rollout_data = collector.prepare_batch_data(rollout_data)
    
    advantages = rollout_data['advantages']
    mean = advantages.mean().item()
    std = advantages.std().item()
    
    assert np.isclose(mean, 0.0, atol=1e-5), f"Mean: {mean}"
    assert np.isclose(std, 1.0, atol=1e-5), f"Std: {std}"


@pytest.mark.unit
def test_mini_batching(collector, mock_policy):
    """Test that mini-batching produces consistent slices."""
    # Create rollout data
    num_steps = 20
    rollout_data = {
        'advantages': torch.randn(num_steps),
        'returns': torch.randn(num_steps),
        'logprobs': torch.randn(num_steps),
        'values': torch.randn(num_steps),
        'actions': torch.randint(0, 4, (num_steps,)),
        'action_masks': torch.ones(num_steps, 4),
        'obs': [{}] * num_steps,
    }
    
    batch_size = 8
    batches = list(collector.get_mini_batches(rollout_data, batch_size=batch_size, policy=mock_policy))
    
    # Check that all batches have consistent keys
    for batch in batches:
        assert 'obs_tensors' in batch
        assert 'actions' in batch
        assert 'action_masks' in batch
        assert 'logprobs' in batch
        assert 'advantages' in batch
        assert 'returns' in batch
        assert 'values' in batch
        
        # Check shapes
        batch_len = len(batch['actions'])
        assert batch['obs_tensors'].shape[0] == batch_len
        assert batch['action_masks'].shape[0] == batch_len
        assert batch['obs_tensors'].device.type == 'cpu'


@pytest.mark.unit
def test_shapes_after_prepare(collector):
    """Test that all arrays have matching lengths after prepare_batch_data."""
    num_steps = 10
    rollout_data = {
        'advantages': np.random.randn(num_steps).astype(np.float32),
        'returns': np.random.randn(num_steps).astype(np.float32),
        'logprobs': np.random.randn(num_steps).astype(np.float32),
        'values': np.random.randn(num_steps).astype(np.float32),
        'actions': np.random.randint(0, 4, num_steps).astype(np.int64),
        'action_masks': np.ones((num_steps, 4), dtype=np.float32),
        'rewards': np.random.randn(num_steps).astype(np.float32),
        'dones': np.zeros(num_steps, dtype=np.float32),
        'obs': [{}] * num_steps,
    }
    
    # Should not raise assertion errors
    rollout_data = collector.prepare_batch_data(rollout_data)
    
    # Check all tensors are on CPU
    assert rollout_data['advantages'].device.type == 'cpu'
    assert rollout_data['actions'].device.type == 'cpu'
