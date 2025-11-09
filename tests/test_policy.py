"""
Unit tests for the PPO policy (ActorCriticPolicy).

Tests cover:
- Policy shapes and output types
- Action masking functionality
- act() method returns scalars
- evaluate_actions() method shapes and entropy
- KL divergence guards for NaN/inf
"""

import pytest
import numpy as np
import torch
from training.policy import ActorCriticPolicy
from training.config import config


@pytest.fixture
def policy():
    """Create a policy instance for testing."""
    return ActorCriticPolicy(
        board_h=15,
        board_w=15,
        num_actions=4,
        latent_size=64,  # Smaller for faster tests
        device='cpu',
        verbose=False
    )


@pytest.fixture
def sample_obs():
    """Create a sample observation."""
    return {
        'board': np.zeros((15, 15), dtype=np.float32),
        'my_head': np.array([7.0, 7.0], dtype=np.float32),
        'opp_head': np.array([10.0, 10.0], dtype=np.float32),
        'boosts': np.array([3.0, 3.0], dtype=np.float32),
        'turns': np.array([0.0], dtype=np.float32),
        'directions': np.array([0.0, 2.0], dtype=np.float32),
        'alive': np.array([1.0, 1.0], dtype=np.float32),
    }


@pytest.mark.unit
def test_policy_shapes(policy):
    """Test that policy forward pass produces correct shapes."""
    batch_size = 8
    obs_tensor = torch.randn(batch_size, policy.feature_size)
    
    logits, values = policy.forward(obs_tensor)
    
    assert logits.shape == (batch_size, 4), f"Logits shape: {logits.shape}"
    assert values.shape == (batch_size, 1), f"Values shape: {values.shape}"


@pytest.mark.unit
def test_encode_obs_shape(policy, sample_obs):
    """Test that encode_obs produces correct shape."""
    obs_tensor = policy.encode_obs(sample_obs)
    
    assert obs_tensor.shape == (policy.feature_size,), f"Encoded obs shape: {obs_tensor.shape}"
    assert obs_tensor.device.type == 'cpu', "Obs tensor should be on CPU"
    assert not obs_tensor.requires_grad, "Encoded obs should be detached"


@pytest.mark.unit
def test_masking_blocks_illegal(policy):
    """Test that action masking blocks illegal moves."""
    # Create a board with walls on three sides (only down is legal)
    obs = {
        'board': np.zeros((15, 15), dtype=np.float32),
        'my_head': np.array([7.0, 7.0], dtype=np.float32),
        'opp_head': np.array([10.0, 10.0], dtype=np.float32),
        'boosts': np.array([3.0, 3.0], dtype=np.float32),
        'turns': np.array([0.0], dtype=np.float32),
        'directions': np.array([0.0, 2.0], dtype=np.float32),
        'alive': np.array([1.0, 1.0], dtype=np.float32),
    }
    
    # Block three directions (up, right, left)
    obs['board'][6, 7] = 1.0  # up
    obs['board'][7, 8] = 1.0  # right
    obs['board'][7, 6] = 1.0  # left
    # down is free
    
    mask = policy._compute_legal_actions_mask(obs)
    
    # Action mapping: 0=up, 1=right, 2=down, 3=left
    assert mask[0] == 0.0, "Up should be blocked"
    assert mask[1] == 0.0, "Right should be blocked"
    assert mask[2] == 1.0, "Down should be legal"
    assert mask[3] == 0.0, "Left should be blocked"


@pytest.mark.unit
def test_act_returns_scalars(policy, sample_obs):
    """Test that act() returns scalar values."""
    action, logprob, value = policy.act(sample_obs)
    
    assert isinstance(action, int), f"Action should be int, got {type(action)}"
    assert 0 <= action < 4, f"Action should be in [0, 3], got {action}"
    assert isinstance(logprob, float), f"Logprob should be float, got {type(logprob)}"
    assert isinstance(value, float), f"Value should be float, got {type(value)}"
    assert np.isfinite(logprob), "Logprob should be finite"
    assert np.isfinite(value), "Value should be finite"


@pytest.mark.unit
def test_evaluate_actions_shapes(policy):
    """Test that evaluate_actions returns correct shapes."""
    batch_size = 16
    obs_tensors = torch.randn(batch_size, policy.feature_size)
    actions = torch.randint(0, 4, (batch_size,))
    action_masks = torch.ones(batch_size, 4)
    
    logprobs, values, entropy, kl = policy.evaluate_actions(
        obs_tensors, actions, action_masks
    )
    
    assert logprobs.shape == (batch_size,), f"Logprobs shape: {logprobs.shape}"
    assert values.shape == (batch_size,), f"Values shape: {values.shape}"
    assert entropy.shape == (batch_size,), f"Entropy shape: {entropy.shape}"
    assert kl.shape == (), f"KL shape: {kl.shape}"
    assert entropy.mean() > 0, "Entropy should be positive"


@pytest.mark.unit
def test_kl_guard_nan(policy):
    """Test that KL guard handles NaN in old_logprobs."""
    batch_size = 8
    obs_tensors = torch.randn(batch_size, policy.feature_size)
    actions = torch.randint(0, 4, (batch_size,))
    action_masks = torch.ones(batch_size, 4)
    
    # Create old_logprobs with NaN
    old_logprobs = torch.tensor([float('nan')] * batch_size)
    
    _, _, _, kl = policy.evaluate_actions(
        obs_tensors, actions, action_masks, old_logprobs=old_logprobs
    )
    
    # KL should be a large finite sentinel value, not NaN
    assert torch.isfinite(kl), f"KL should be finite, got {kl}"
    assert kl.item() > 0, f"KL sentinel should be positive, got {kl.item()}"


@pytest.mark.unit
def test_model_size_under_5mb(policy):
    """Test that model size is under 5MB."""
    total_params = sum(p.numel() for p in policy.parameters())
    model_size_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
    
    assert model_size_mb < 5.0, f"Model size {model_size_mb:.2f}MB exceeds 5MB limit"
